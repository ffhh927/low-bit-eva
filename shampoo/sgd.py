# SGD implementation based on offical pytorch code

import torch
import torch.optim as optim


class SGD(optim.Optimizer):
    def __init__(self,
                params,
                lr=0.1,
                momentum=0.9,
                weight_decay=0.0,
                nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SGD does not support sparse yet')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if group['weight_decay'] != 0.0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state['momentum'].mul_(group['momentum']).add_(grad)

                if group['nesterov']:
                    grad = grad.add(state['momentum'], alpha=group['momentum'])
                else:
                    grad = state['momentum']

                p.data.add_(grad, alpha=-group['lr'])
