# -*- coding: utf-8 -*-
# File   : batchnorm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.

import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm

from .sync_manager import SyncManager

__all__ = ['SynchronizedBatchNorm1d']


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_manager = SyncManager(self)

        self._is_parallel = False
        self._parallel_id = None
        self._child_registry = None

    def compute_mean_std(self, sum_, ssum, size):
        assert size > 1
        mean = sum_ / size
        sumvar = ssum - sum_ * mean 
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, (bias_var + self.eps) ** -0.5

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_manager.collect(input_sum, input_ssum, sum_size)
        else:
            mean, inv_std = self._child_registry.get(input_sum, input_ssum, sum_size)

        output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        if self.affine:
            output = output * _unsqueeze_ft(self.weight) + _unsqueeze_ft(self.bias)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, current_id):
        self._is_parallel = True
        self._parallel_id = current_id
        if self._parallel_id == 0:
            ctx.sync_manager = self._sync_manager
        else:
            self._child_registry = ctx.sync_manager.register(current_id)


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)
