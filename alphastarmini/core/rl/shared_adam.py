#!/usr/bin/env python
# -*- coding: utf-8 -*-

" shared adam by MorvanZhou and ikostrikov"

import math

import torch
import torch.optim as optim


class MorvanZhouSharedAdam(optim.Adam):
    '''
    https://github.com/MorvanZhou/pytorch-A3C/blob/master/shared_adam.py
    '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(MorvanZhouSharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class IkostrikovSharedAdam(optim.Adam):
    """
    https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py
    Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(IkostrikovSharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

                del state

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

                del state

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                del grad, state, exp_avg, exp_avg_sq, beta1, beta2
                del denom, bias_correction1, bias_correction2, step_size

        return loss


def show_grads(model, shared_model, debug):
    '''
    https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
    '''
    print('show_grads') if debug else None
    for param, shared_param in zip(model.named_parameters(),
                                   shared_model.named_parameters()):
        name = param[0]
        if "winloss_baseline.cumulatscore_fc" in name:
            print('param name', param[0]) if debug else None
            print('param device', param[1].device) if debug else None
            print('param grad[0]', param[1].grad[0]) if debug else None
            # print('shared_param name', shared_param[0]) if debug else None
            # print('shared_param device', shared_param[1].device) if debug else None
            # print('shared_param grad', shared_param[1].grad) if debug else None  
            # break
        del param, shared_param


def show_datas(model, shared_model, debug):
    '''
    https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
    '''
    print('show_datas') if debug else None
    for param, shared_param in zip(model.named_parameters(),
                                   shared_model.named_parameters()):
        name = param[0]
        if "winloss_baseline.cumulatscore_fc" in name:
            print('param name', param[0]) if debug else None
            print('param device', param[1].device) if debug else None
            print('param data[0]', param[1].data[0]) if debug else None
            # print('shared_param name', shared_param[0]) if debug else None
            # print('shared_param device', shared_param[1].device) if debug else None
            # print('shared_param data', shared_param[1].data) if debug else None
            # break
        del param, shared_param


def ensure_shared_grads(model, shared_model, debug):
    '''
    https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
    '''
    for param, shared_param in zip(model.named_parameters(),
                                   shared_model.named_parameters()):
        if shared_param[1].grad is not None:
            print('shared_param is not None', shared_param[0]) if debug else None
            return
        if param[1].grad is not None:
            shared_param[1]._grad = param[1].grad.to(shared_param[1].device)
        else:
            pass
            print('param grad is None', param[0]) if debug else None

        del param, shared_param
