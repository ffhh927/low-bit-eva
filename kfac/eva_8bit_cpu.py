import math
import torch
import torch.optim as optim
import numpy as np
#import horovod.torch as hvd
#import kfac.backend as backend
#backend.init("Horovod") 
import bitsandbytes.functional as B_F
from bitsandbytes.optim import SGD8bit 
import torch.profiler as profiler
from kfac.utils import get_vector_a, get_vector_g
import logging
logger = logging.getLogger()
from concurrent.futures import ThreadPoolExecutor, wait
import time

class KFAC(optim.Optimizer):
    """Accelerate Distributed K-FAC with Sublinear Memory Cost
    Args:
      model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.03)
      kl_clip (float): clipping parameter for gradient scaling (kl_clip > 0: kl-clip, kl_clip = 0: re-scale, kl-clip < 0: None)
      factor_decay (float): running average coefficient for KVs
      exclude_vocabulary_size: exclude the pre-softmax linear layer in the Transformer
      hook_enabled (bool): enable the hook events to save the immediate states (a and g)
      exclude_parts='': exclude CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor for time breakdowns
    """
    def __init__(self,
                 model,
                 lr=0.1,
                 damping=0.03,
                 fac_update_freq=1,
                 kfac_update_freq=1,
                 kfac_batch_size=16,
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 sgd_weight_decay = 5e-4,
                 sgd_momentum = 0.9,
                 sgd_dampening = 0,
                 sgd_nesterov = False,
                 exclude_parts=''):

        # For compatibility with `KFACParamScheduler`
        defaults = dict(lr=lr,
                        damping=damping,
                        fac_update_freq=fac_update_freq,
                        kfac_update_freq=kfac_update_freq,
                        weight_decay=sgd_weight_decay,
                        dampening = sgd_dampening,
                        nesterov = sgd_nesterov,
                        momentum = sgd_momentum) 

        super(KFAC, self).__init__(model.parameters(), defaults)

        self.fac_update_freq = fac_update_freq
        self.kfac_batch_size = kfac_batch_size
        self.kl_clip = kl_clip if (kl_clip is not None and kl_clip >= 0) else None
        self.factor_decay = factor_decay
        self.exclude_vocabulary_size = exclude_vocabulary_size
        self.hook_enabled = hook_enabled
        self.block_size = 2048
        
        # register hooks
        self.modules = []
        self.module_names = []
        self._register_module_hooks(model)

        # dictionaries keyed by `module` to storing KFs, inverse KFs, etc
        self.m_a, self.m_g = {}, {}
        # self.m_a_test, self.m_g_test = {}, {}
        self.quant_state_a, self.quant_state_g, self.quant_state_v = {},{},{}
        self.handles = []
        
        # scheduling results
        self.module_ranks = None

        self.steps = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.code = B_F.create_dynamic_map(signed=True).to(self.device)  # 创建动态量化映射表
        self.cpu_executor = ThreadPoolExecutor(max_workers=10)
        self.tasks = []
        
        self.optim = SGD8bit(model.parameters(), lr=lr, momentum=sgd_momentum)
        self.a_streams = torch.cuda.Stream() 
        self.a_event = torch.cuda.Event()
        self.g_streams = torch.cuda.Stream()
        self.g_event = torch.cuda.Event()
        # self.model = model 
          
    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode
        
    # @profile
    def _forward_hook_event(self, module, input):
        """Default: hook for saving input (a)"""
        # start = time.time()
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new_fp32 = get_vector_a(input[0].data[0:self.kfac_batch_size], module)
                # self.m_a_test[module] = new_fp32
                if module not in self.m_a:
                    new_fp8, self.quant_state_a[module] = B_F.quantize_blockwise(new_fp32[:-1], code=self.code, blocksize=self.block_size)
                    self.m_a[module] = [new_fp32[-1], new_fp8]
                else:
                    # weights
                    with torch.cuda.stream(self.a_streams):
                        # gpu -> cpu                
                        quant_state_a_absmax_cpu = self.quant_state_a[module].absmax.to("cpu", non_blocking=True)
                        new_fp32_cpu = new_fp32.to("cpu", non_blocking=True)
                        m_a_1_cpu = self.m_a[module][1].to("cpu", non_blocking=True)
                        m_a_0_cpu = self.m_a[module][0].to("cpu", non_blocking=True)
                        self.a_event.record() 
                                        
                    # dequantize
                    torch.cuda.current_stream().wait_event(self.a_event)
                    m_a_old = B_F.dequantize_blockwise(A=m_a_1_cpu, absmax=quant_state_a_absmax_cpu, blocksize=self.block_size, code=self.code)
                    
                    # compute
                    torch.lerp(m_a_0_cpu, new_fp32_cpu[-1], self.factor_decay, out=m_a_0_cpu)
                    new_fp32_cpu = new_fp32_cpu[:-1]
                    torch.lerp(m_a_old, new_fp32_cpu, self.factor_decay, out=new_fp32_cpu)
                    
                    # quantize
                    new_fp8_cpu, quant_state_a_cpu = B_F.quantize_blockwise(new_fp32_cpu, code=self.code, blocksize=self.block_size)
                        
                    # cpu -> gpu
                    with torch.cuda.stream(self.a_streams):
                        self.quant_state_a[module].absmax = quant_state_a_cpu.absmax.to(self.device, non_blocking=True)
                        self.m_a[module][0] = m_a_0_cpu.to(self.device, non_blocking=True)
                        self.m_a[module][1] = new_fp8_cpu.to(self.device, non_blocking=True)
                        
                            
    def _backward_hook_event(self, module, grad_input, grad_output):
        """Default: hook for saving gradient w.r.t output (g)"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new_fp32 = get_vector_g(grad_output[0].data[0:self.kfac_batch_size], module)  
                # print(new_fp32)        
                if module not in self.m_g:
                    new_fp8, self.quant_state_g[module] = B_F.quantize_blockwise(new_fp32[1:], code=self.code, blocksize=self.block_size)
                    self.m_g[module] = [new_fp32[0], new_fp8]
                else:
                    # gpu -> cpu
                    with torch.cuda.stream(self.g_streams):
                        quant_state_g_absmax_cpu = self.quant_state_g[module].absmax.to("cpu", non_blocking=True)
                        new_fp32_cpu = new_fp32.to("cpu", non_blocking=True)
                        m_g_1_cpu = self.m_g[module][1].to("cpu", non_blocking=True)
                        m_g_0_cpu = self.m_g[module][0].to("cpu", non_blocking=True)
                        self.g_event.record() 
                    
                    # dequantize
                    torch.cuda.current_stream().wait_event(self.g_event)
                    m_g_old = B_F.dequantize_blockwise(A=m_g_1_cpu, absmax=quant_state_g_absmax_cpu, blocksize=self.block_size, code=self.code)
                    
                    # compute
                    torch.lerp(m_g_0_cpu, new_fp32_cpu[0], self.factor_decay, out=m_g_0_cpu)
                    new_fp32_cpu = new_fp32_cpu[1:]
                    torch.lerp(m_g_old, new_fp32_cpu, self.factor_decay, out=new_fp32_cpu)
                    
                    # quantize
                    new_fp8_cpu, quant_state_g_cpu = B_F.quantize_blockwise(new_fp32_cpu, code=self.code, blocksize=self.block_size)
                        
                    # cpu -> gpu
                    with torch.cuda.stream(self.g_streams):
                        self.quant_state_g[module].absmax = quant_state_g_cpu.absmax.to(self.device, non_blocking=True)
                        self.m_g[module][0] = m_g_0_cpu.to(self.device, non_blocking=True)
                        self.m_g[module][1] = new_fp8_cpu.to(self.device, non_blocking=True)
         
    def _register_module_hooks(self, model):
        """Register forard/backward hooks to supported modules"""
        supported_modules = {'Linear', 'Conv2d'}
        name_idx = 0
        for module in model.modules():
            classname = module.__class__.__name__
            if classname in supported_modules:
                if self.exclude_vocabulary_size is not None and classname == 'Linear' and module.out_features == self.exclude_vocabulary_size:
                    continue # exclude the pre-softmax linear layer in the Transformer model
                self.modules.append(module)
                module.register_forward_pre_hook(self._forward_hook_event)
                module.register_backward_hook(self._backward_hook_event)  # used in pytorch1.4, and pytorch1.8 (full_backward_hook is not fired when its grad_input is None)
                #module.register_full_backward_hook(self._backward_hook_event)  # used in pytorch1.10
                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                name_idx += 1
        #if backend.comm.rank() == 0:
         #   logger.info("#register modules: %s", len(self.modules))

	### Precondition gradients
    def _precondition_grads(self):
        """Compute preconditioned gradients via Eva"""
        g_sum = 0
        v_sum = 0
        vg_sum = 0
        
        # wait(self.tasks)
        # self.tasks.clear()

        for module in self.modules:
            ma = torch.cat((B_F.dequantize_blockwise(self.m_a[module][1], self.quant_state_a[module], blocksize=self.block_size), self.m_a[module][0].unsqueeze(0)), dim=0).view(-1, 1)
            mg = torch.cat((self.m_g[module][0].unsqueeze(0), B_F.dequantize_blockwise(self.m_g[module][1], self.quant_state_g[module], blocksize=self.block_size)), dim=0).view(-1, 1)

            # print("ma:",ma)
            # print("mg:",mg)
            grad = self._get_grad(module)
            
            # compute intermediate states
            a = (ma.T @ ma).item()
            g = (mg.T @ mg).item()
            ag = (mg.T @ grad @ ma).item()

            # compute preconditioned grads
            v = (mg @ ma.T).mul_(-ag/(a * g + self.damping))           
            v.add_(grad)
            v.div_(self.damping)
            

            #print(v)
            # weight and bias
            if module.bias is not None:
                weight = v[:, :-1].view(module.weight.grad.data.size())
                bias = v[:, -1:].view(module.bias.grad.data.size())
                # transform preconditioned gradient into gradient scale
                if self.kl_clip is not None:
                    if self.kl_clip > 0:
                        vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                        vg_sum += (bias * module.bias.grad.data * self.lr ** 2).sum().item()
                    else:
                        v_sum += (weight * weight).sum().item()
                        v_sum += (bias * bias).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                        g_sum += (module.bias.grad.data * module.bias.grad.data).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
                module.bias.grad.data.copy_(bias)
                del grad
            else:
                weight = v.view(module.weight.grad.data.size())
                if self.kl_clip is not None:
                    if self.kl_clip > 0:
                        vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                    else:
                        v_sum += (weight * weight).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                # copy
                module.weight.grad.data.copy_(weight)
            #print(v)
            del v

        # scale preconditioned gradient
        if self.kl_clip is not None:
            if self.kl_clip > 0: # kl-clip
                nu = min(1.0, math.sqrt(self.kl_clip / vg_sum)) if vg_sum > 0 else 1.0
            else: # re-scale
                nu = math.sqrt(g_sum / v_sum)

            for module in self.modules:
                module.weight.grad.data.mul_(nu)
                if module.bias is not None:
                    module.bias.grad.data.mul_(nu)
     

    def _get_grad(self, module):
        """Get gradient with shape [output_dim, input_dim] for module"""
        if module.__class__.__name__ == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            grad = module.weight.grad.data.view(module.weight.grad.data.size(0), -1)
        else:
            grad = module.weight.grad.data
        if module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad    


    ### Perform one K-FAC step
    @torch.no_grad()
    def step(self, closure=None, epoch=None):
        """Perform one K-FAC step"""

        # update params, used for compatibilty with `KFACParamScheduler`
        group = self.param_groups[0]
        self.lr = group['lr']
        self.damping = group['damping']
        self.fac_update_freq = group['fac_update_freq']
        self.kfac_update_freq = group['kfac_update_freq']

       # if self.steps % self.fac_update_freq == 0 and backend.comm.size() > 1:
      #      for handle in self.handles:
       #         backend.comm.synchronize(handle)
      #      self.handles = []
        
        self._precondition_grads()
        # self._sgd()
        self.optim.step()

        self.steps += 1

    def _sgd(self, closure=None, epoch=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = torch.clone(d_p).detach()                       
                    else:
                        buf = B_F.dequantize_blockwise(param_state['momentum_buffer'],param_state['quant_momentum_max'], blocksize=self.block_size)                  
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)        
                    param_state['momentum_buffer'], param_state['quant_momentum_max']= B_F.quantize_blockwise(d_p, code=self.code, blocksize=self.block_size)
                    # if 'momentum_buffer' not in param_state:
                    #     buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    # else:
                    #     buf = param_state['momentum_buffer']
                    #     buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])            

        return loss