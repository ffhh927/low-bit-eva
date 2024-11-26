# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code file is partially based on https://github.com/google-research/google-research/tree/master/scalable_shampoo/pytorch/shampoo.py

"""Pytorch implementation of naive 4-bit Shampoo."""

import math
import itertools

from shampoo.matrix_functions import ComputePower
import numpy as np
import torch
import torch.optim as optim

import qtensor.functional as F
from qtensor.utils import QTensorDiagReal


CODE = 'linear-2'


class BlockPartitioner:
  """Partitions a tensor into smaller tensors for preconditioning.

    For example, if a variable has shape (4096, 512), we might split the
    4096 into 4 blocks, so we effectively have 4 variables of size
    (1024, 512) each.
  """

  def __init__(self, var, block_size):
    self._shape = var.shape
    self._splits = []
    self._split_sizes = []
    split_sizes = []
    # We split var into smaller blocks. Here we store the metadata to make
    # that split.
    for i, d in enumerate(var.shape):
      if block_size > 0 and d > block_size:
        # d-1, otherwise split appends a 0-size array.
        nsplit = (d-1) // block_size
        indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
        sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
        sizes[-1] = d - indices[-1]
        self._splits.append((i, indices))
        self._split_sizes.append((i, sizes))
        split_sizes.append(sizes)
      else:
        split_sizes.append(np.array([d], dtype=np.int32))
    self._num_splits = len(split_sizes)
    self._preconditioner_shapes = []
    for t in itertools.product(*split_sizes):
      self._preconditioner_shapes.extend([[d, d] for d in t])

  def shapes_for_preconditioners(self):
    return self._preconditioner_shapes

  def num_splits(self):
    return self._num_splits

  def partition(self, tensor):
    """Partition tensor into blocks."""

    assert tensor.shape == self._shape
    tensors = [tensor]
    for (i, sizes) in self._split_sizes:
      tensors_local = []
      for t in tensors:
        tensors_local.extend(
            torch.split(t, tuple(sizes), dim=i))
      tensors = tensors_local
    return tensors

  def merge_partitions(self, partitions):
    """Merge partitions back to original shape."""

    for (i, indices) in reversed(self._splits):
      n = len(indices) + 1
      partial_merged_tensors = []
      ind = 0
      while ind < len(partitions):
        partial_merged_tensors.append(
            torch.cat(partitions[ind:ind + n], axis=i))
        ind += n
      partitions = partial_merged_tensors
    assert len(partitions) == 1
    return partitions[0]


class Preconditioner:
    """Compute statistics/shape from gradients for preconditioning."""
    def __init__(self, var, name2qmap, 
                stat_decay=0.95, matrix_eps=1e-6, prec_maxorder=1200,
                prec_bits=32, min_lowbit_size=4096, quan_blocksize=64):
        self.stat_decay = stat_decay
        self.matrix_eps = matrix_eps

        self._original_shape = var.shape
        self._transformed_shape = [x for x in var.shape if x != 1]

        if len(self._transformed_shape) > 1:
            self._transformed_shape = (self._transformed_shape[0], -1)
     
        reshaped_var = torch.reshape(var, self._transformed_shape)

#        print("var.shape:", var.shape)
#        print("reshaped_var.shape:", reshaped_var.shape)
#        print("")

        self._partitioner = BlockPartitioner(reshaped_var, prec_maxorder)
        shapes = self._partitioner.shapes_for_preconditioners()
        rank = len(self._transformed_shape)
        device = var.get_device()
        if rank <= 1:
            self.statistics = []
            self.preconditioners = []
        else:
            eps = self.matrix_eps
            self.statistics = [QTensorDiagReal(eps * torch.eye(s[0], device=device), bits=prec_bits, 
                               name2qmap=name2qmap, code=CODE, blocksize=quan_blocksize, min_lowbit_size=min_lowbit_size) for s in shapes]
            self.preconditioners = [QTensorDiagReal(torch.eye(s[0], device=device), bits=prec_bits, 
                               name2qmap=name2qmap, code=CODE, blocksize=quan_blocksize, min_lowbit_size=min_lowbit_size) for s in shapes]

    def set_device(self, device):
        rank = len(self._transformed_shape)
        if rank <= 1: return
        for i in range(len(self.statistics)):
            self.statistics[i].set_device(device)
            self.preconditioners[i].set_device(device)

    def add_statistics(self, grad):
        """Compute statistics from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.statistics: return
        reshaped_grad = torch.reshape(grad, self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)

#        print("reshaped_grad:", reshaped_grad.shape, " len:", len(partitioned_grads))

        w1 = self.stat_decay
        w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
        rank = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            for i in range(rank):
                if i == 1:
                    stat = grad.T @ grad
                else:
                    stat = grad @ grad.T

                statistics_i = self.statistics[j*rank + i].dequantize()
                statistics_i.mul_(w1).add_(stat, alpha=w2)
                self.statistics[j*rank + i].quantize(statistics_i)

    def compute_preconditioners(self):
        """Compute L^{-1/exp} for each stats matrix L."""
        exp = 4
        eps = self.matrix_eps
        for statistics_i, preconditioners_i in zip(self.statistics, self.preconditioners):
            statistics_i_de = statistics_i.dequantize()
            preconditioners_i.quantize(ComputePower(statistics_i_de.float(), exp, ridge_epsilon=eps).to(statistics_i_de.dtype))

    def preconditioned_grad(self, grad):
        """Precondition the gradient.

        Args:
          grad: A gradient tensor to precondition.

        Returns:
          A preconditioned gradient.
        """
        if not self.preconditioners: return grad
        reshaped_grad = torch.reshape(grad, self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        preconditioned_partitioned_grads = []
        num_splits = self._partitioner.num_splits()

        for i, grad in enumerate(partitioned_grads):
            preconditioners_for_grad = self.preconditioners[i * num_splits:(i + 1) * num_splits]

            rank = len(grad.shape)
            precond_grad = grad
            for j in range(rank):
                preconditioner = preconditioners_for_grad[j].dequantize()
                preconditioner = preconditioner.to(grad.dtype)

                if j == 1:
                    precond_grad = precond_grad @ preconditioner
                else:
                    precond_grad = preconditioner @ precond_grad
            preconditioned_partitioned_grads.append(precond_grad)

        merged_grad = self._partitioner.merge_partitions(preconditioned_partitioned_grads)
        return torch.reshape(merged_grad, self._original_shape)


class ShampooSGD(optim.Optimizer):
    def __init__(self,
                params,
                lr=0.1,
                momentum=0.9,
                weight_decay=0.0,
                nesterov=False,
                start_prec_step=1,
                stat_compute_steps=100,
                prec_compute_steps=500,
                stat_decay=0.95,
                matrix_eps=1e-6,
                prec_maxorder=1200,
                prec_bits=32,
                min_lowbit_size=4096,
                quan_blocksize=64):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                        start_prec_step=start_prec_step, stat_compute_steps=stat_compute_steps, prec_compute_steps=prec_compute_steps,
                        stat_decay=stat_decay, matrix_eps=matrix_eps, prec_maxorder=prec_maxorder,
                        prec_bits=prec_bits, min_lowbit_size=min_lowbit_size, quan_blocksize=quan_blocksize)
        super(ShampooSGD, self).__init__(params, defaults)

        self.name2qmap = {}
        if prec_bits in [4, 8]:
            if CODE == 'dynamic':
                self.name2qmap[CODE] = F.create_dynamic_map(signed=True, total_bits=prec_bits, power=1)
            elif CODE == 'linear-2':
                self.name2qmap[CODE] = F.create_linear_map(signed=True, total_bits=prec_bits, power=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) != 0:
                    state['preconditioner'].set_device(p.device)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Shampoo does not support sparse yet')

                start_prec_step = group['start_prec_step']
                stat_compute_steps = group['stat_compute_steps']
                prec_compute_steps = group['prec_compute_steps']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['preconditioner'] = Preconditioner(p, self.name2qmap, 
                        stat_decay=group['stat_decay'], matrix_eps=group['matrix_eps'], prec_maxorder=group['prec_maxorder'],
                        prec_bits=group['prec_bits'], min_lowbit_size=group['min_lowbit_size'], quan_blocksize=group['quan_blocksize'])
                    state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                state['step'] += 1
                preconditioner = state['preconditioner']

                # Gather statistics, compute preconditioners
                if state['step'] % stat_compute_steps == 0:
                    preconditioner.add_statistics(grad)
                if state['step'] % prec_compute_steps == 0:
                    preconditioner.compute_preconditioners()

                # Precondition gradients
                shampoo_grad = grad
                if state['step'] >= start_prec_step:
                    shampoo_grad = preconditioner.preconditioned_grad(grad)
                    shampoo_grad.mul_(grad.norm() / (shampoo_grad.norm() + 1e-12))

                if group['weight_decay'] != 0.0:
                    shampoo_grad.add_(p.data, alpha=group['weight_decay'])

                state['momentum'].mul_(group['momentum']).add_(shampoo_grad)

                if group['nesterov']:
                    shampoo_grad = shampoo_grad.add(state['momentum'], alpha=group['momentum'])
                else:
                    shampoo_grad = state['momentum']

                p.data.add_(shampoo_grad, alpha=-group['lr'])


class ShampooAdamW(optim.Optimizer):
    def __init__(self,
                params,
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
                start_prec_step=1,
                stat_compute_steps=100,
                prec_compute_steps=500,
                stat_decay=0.95,
                matrix_eps=1e-6,
                prec_maxorder=1200,
                prec_bits=32,
                min_lowbit_size=4096,
                quan_blocksize=64):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        start_prec_step=start_prec_step, stat_compute_steps=stat_compute_steps, prec_compute_steps=prec_compute_steps,
                        stat_decay=stat_decay, matrix_eps=matrix_eps, prec_maxorder=prec_maxorder,
                        prec_bits=prec_bits, min_lowbit_size=min_lowbit_size, quan_blocksize=quan_blocksize)
        super(ShampooAdamW, self).__init__(params, defaults)

        self.name2qmap = {}
        if prec_bits in [4, 8]:
            if CODE == 'dynamic':
                self.name2qmap[CODE] = F.create_dynamic_map(signed=True, total_bits=prec_bits, power=1)
            elif CODE == 'linear-2':
                self.name2qmap[CODE] = F.create_linear_map(signed=True, total_bits=prec_bits, power=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) != 0:
                    state['preconditioner'].set_device(p.device)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Shampoo does not support sparse yet')

                start_prec_step = group['start_prec_step']
                stat_compute_steps = group['stat_compute_steps']
                prec_compute_steps = group['prec_compute_steps']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['preconditioner'] = Preconditioner(p, self.name2qmap, 
                        stat_decay=group['stat_decay'], matrix_eps=group['matrix_eps'], prec_maxorder=group['prec_maxorder'],
                        prec_bits=group['prec_bits'], min_lowbit_size=group['min_lowbit_size'], quan_blocksize=group['quan_blocksize'])
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                preconditioner = state['preconditioner']

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Gather statistics, compute preconditioners
                if state['step'] % stat_compute_steps == 0:
                    preconditioner.add_statistics(grad)
                if state['step'] % prec_compute_steps == 0:
                    preconditioner.compute_preconditioners()

                # Precondition gradients
                shampoo_grad = grad
                if state['step'] >= start_prec_step:
                    shampoo_grad = preconditioner.preconditioned_grad(grad)
                    shampoo_grad.mul_(grad.norm() / (shampoo_grad.norm() + 1e-12))

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(shampoo_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(shampoo_grad, shampoo_grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Gradient descent
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)


ShampooSGD.__doc__ = r"""
    Implements SGD+Shampoo (naive quantization).

    Args:
        start_prec_step (int): the step starting preconditioning
        stat_compute_steps (int): interval of updating preconditioners (T_1)
        prec_compute_steps (int): interval of updating inverse roots of preconditioners (T_2)
        stat_decay (float): exponential decay rate for preconditioners (beta)
        matrix_eps (float): dampening term (epsilon)
        prec_maxorder (int): maximum order for preconditioners
        prec_bits (int): bitwidth of a preconditioner
        min_lowbit_size (int): minimum tensor size required for quantization
        quan_blocksize (int): block size for block-wise quantization

    """


ShampooAdamW.__doc__ = r"""
    Implements AdamW+Shampoo (naive quantization).

    Args:
        start_prec_step (int): the step starting preconditioning
        stat_compute_steps (int): interval of updating preconditioners (T_1)
        prec_compute_steps (int): interval of updating inverse roots of preconditioners (T_2)
        stat_decay (float): exponential decay rate for preconditioners (beta)
        matrix_eps (float): dampening term (epsilon)
        prec_maxorder (int): maximum order for preconditioners
        prec_bits (int): bitwidth of a preconditioner
        min_lowbit_size (int): minimum tensor size required for quantization
        quan_blocksize (int): block size for block-wise quantization

    """
