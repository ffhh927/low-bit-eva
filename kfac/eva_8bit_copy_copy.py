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
# 配置日志
logging.basicConfig(
    filename="./logs/eva8bit_cpu.log",       # 日志文件名
    level=logging.INFO,        # 日志级别
    format="%(levelname)s - %(message)s",  # 日志格式
)
from concurrent.futures import ThreadPoolExecutor, wait
import time
def batch_transfer_to_cpu_inplace(tensor_list):
    """
    将张量列表中的每个张量合并到缓冲区，一次性传输到 CPU，并原地修改列表内容。
    
    Args:
        tensor_list (list of torch.Tensor): 输入张量列表，所有张量需位于相同设备。
        
    Returns:
        None: 原地修改张量列表的内容。
    """
    # 检查张量是否位于同一设备
    device = tensor_list[0].device
    for tensor in tensor_list:
        if tensor.device != device:
            raise ValueError("All tensors in the list must be on the same device.")
    
    # 计算缓冲区大小和偏移量
    offsets = [0]
    total_size = 0
    for tensor in tensor_list:
        total_size += tensor.numel()
        offsets.append(total_size)
    
    # 创建缓冲区并拷贝数据
    buf = torch.empty(total_size, dtype=tensor_list[0].dtype, device=device)
    for tensor, start, end in zip(tensor_list, offsets[:-1], offsets[1:]):
        buf[start:end] = tensor.flatten()
    
    # 一次性传输到 CPU
    buf_cpu = buf.to("cpu", non_blocking=True)
    
    # 从缓冲区还原张量并原地修改
    for i, (tensor, start, end) in enumerate(zip(tensor_list, offsets[:-1], offsets[1:])):
        tensor_list[i] = buf_cpu[start:end].view_as(tensor)

def batch_transfer_to_gpu_inplace(tensor_list, device):
    """
    将张量列表中的每个张量合并到缓冲区，一次性传输到 GPU，并原地修改列表内容。
    
    Args:
        tensor_list (list of torch.Tensor): 输入张量列表，所有张量需位于同一设备。
        device (torch.device): 目标设备（如 'cuda:0' 或 'cuda'）。
        
    Returns:
        None: 原地修改张量列表的内容。
    """
    # 检查张量是否位于同一设备
    for tensor in tensor_list:
        if tensor.device != 'cpu':  # 可以接收 CPU 张量
            raise ValueError("All tensors in the list must be on CPU.")
    
    # 计算缓冲区大小和偏移量
    offsets = [0]
    total_size = 0
    for tensor in tensor_list:
        total_size += tensor.numel()
        offsets.append(total_size)
    
    # 创建缓冲区并拷贝数据
    buf = torch.empty(total_size, dtype=tensor_list[0].dtype, device='cpu')
    for tensor, start, end in zip(tensor_list, offsets[:-1], offsets[1:]):
        buf[start:end] = tensor.flatten()
    
    # 一次性传输到 GPU
    buf_gpu = buf.to(device, non_blocking=True)
    
    # 从缓冲区还原张量并原地修改
    for i, (tensor, start, end) in enumerate(zip(tensor_list, offsets[:-1], offsets[1:])):
        tensor_list[i] = buf_gpu[start:end].view_as(tensor)
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
        # self.model = model 
          
    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

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
                    # print("start")
                    # start = time.time()
                    self.tasks.append(self.cpu_executor.submit(
                        self._cpu_quantization_taskA, 
                        module, new_fp32[:-1]
                    ))
                    
                    # self.m_a[module][0].mul_(1-self.factor_decay).add_(new_fp32[-1], alpha=self.factor_decay)
                    torch.lerp(self.m_a[module][0], new_fp32[-1], self.factor_decay, out=self.m_a[module][0])
                    # print(time.time()-start)
                # del new_fp32
        # print(time.time()-start)     

                            
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
                    # print("test")
                    # new_fp32_cpu = new_fp32[1:].cpu()
                    # with profiler.profile(
                    #     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                    #     on_trace_ready=profiler.tensorboard_trace_handler('./log')
                    # ) as prof:
                    #     self.tasks.append(self.cpu_executor.submit(
                    #         self._cpu_quantization_taskG, 
                    #         module, new_fp32[1:]
                    #     ))
                    # print(prof.key_averages().table(sort_by="cuda_time_total"))
                    # self.m_g[module][0].mul_(1-self.factor_decay).add_(new_fp32[0], alpha=self.factor_decay)
                    # for param in self.model.parameters():
                    #     print(param)
                    self.tasks.append(self.cpu_executor.submit(
                            self._cpu_quantization_taskG, 
                            module, new_fp32[1:]
                    ))
                    torch.lerp(self.m_g[module][0], new_fp32[0], self.factor_decay, out=self.m_g[module][0])
                del new_fp32
    
    
    def _cpu_quantization_taskA(self, module, new_fp32):
        # 解量化 -> 合并 -> 重新量化  
        self.quant_state_a[module].absmax = self.quant_state_a[module].absmax.cpu()
        new_fp32 = new_fp32.cpu()
        m_a_old = B_F.dequantize_blockwise(self.m_a[module][1].cpu(), self.quant_state_a[module], blocksize=self.block_size)
        # print(f"[{threading.current_thread().name}] m_g: {self.quant_state_g[module].absmax}")
        # m_g_old = B_F.dequantize_blockwise(self.m_g[module][1], self.quant_state_g[module], blocksize=self.block_size)
        # print(f"[{threading.current_thread().name}] a m_g_old: {m_g_old}")
        # print(f"[{threading.current_thread().name}] g new_fp32: {new_fp32}")
        torch.lerp(m_a_old, new_fp32, self.factor_decay, out=new_fp32)
        # print(f"[{threading.current_thread().name}] g new_fp32: {new_fp32}")
        new_fp8, self.quant_state_a[module] = B_F.quantize_blockwise(new_fp32, code=self.code, blocksize=self.block_size)
        # self.batch_transfer_to_gpu_inplace([self.quant_state_g[module].absmax, self.m_g[module][1]])
        self.quant_state_a[module].absmax = self.quant_state_a[module].absmax.to(self.device, non_blocking=True)
        self.quant_state_a[module].code = self.code
        self.m_a[module][1] = new_fp8.to(self.device, non_blocking=True)
            
    def _cpu_quantization_taskG(self, module, new_fp32):
            # 解量化 -> 合并 -> 重新量化
        # batch_transfer_to_cpu_inplace([self.quant_state_g[module].absmax, new_fp32, self.m_g[module][1]])
        # torch.cuda.synchronize()
        # self.quant_state_g[module].absmax = restored_tensors
        self.quant_state_g[module].absmax = self.quant_state_g[module].absmax.cpu()
        new_fp32 = new_fp32.cpu()
        m_g_old = B_F.dequantize_blockwise(self.m_g[module][1].cpu(), self.quant_state_g[module], blocksize=self.block_size)
        # print(f"[{threading.current_thread().name}] m_g: {self.quant_state_g[module].absmax}")
        # m_g_old = B_F.dequantize_blockwise(self.m_g[module][1], self.quant_state_g[module], blocksize=self.block_size)
        # print(f"[{threading.current_thread().name}] a m_g_old: {m_g_old}")
        # print(f"[{threading.current_thread().name}] g new_fp32: {new_fp32}")
        torch.lerp(m_g_old, new_fp32, self.factor_decay, out=new_fp32)
        # print(f"[{threading.current_thread().name}] g new_fp32: {new_fp32}")
        new_fp8, self.quant_state_g[module] = B_F.quantize_blockwise(new_fp32, code=self.code, blocksize=self.block_size)
        # self.batch_transfer_to_gpu_inplace([self.quant_state_g[module].absmax, self.m_g[module][1]])
        self.quant_state_g[module].absmax = self.quant_state_g[module].absmax.to(self.device, non_blocking=True)
        self.quant_state_g[module].code = self.code
        self.m_g[module][1] = new_fp8.to(self.device, non_blocking=True)
        # print(f"[{threading.current_thread().name}] g new_fp8: {new_fp8}")
        # logging.info(f"Logging a :\n{new_fp8}")
    
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
        
        wait(self.tasks)
        self.tasks.clear()
        # print("test")
        # for param in self.model.parameters():
        #     print(param)
        for module in self.modules:
            # print("test")
            # print(self.m_a[module][0])
            # print(self.m_a[module][1])
            ma = torch.cat((B_F.dequantize_blockwise(self.m_a[module][1], self.quant_state_a[module], blocksize=self.block_size), self.m_a[module][0].unsqueeze(0)), dim=0).view(-1, 1)
            mg = torch.cat((self.m_g[module][0].unsqueeze(0), B_F.dequantize_blockwise(self.m_g[module][1], self.quant_state_g[module], blocksize=self.block_size)), dim=0).view(-1, 1)
            # logging.info(f"Logging a :\n{ma}")
            # logging.info(f"Logging g :\n{mg}")
            # print(ma)
            # print(self.quant_state_g[module].absmax)
            # print(mg)
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