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

        return mean, torch.sqrt(bias_var + self.eps)

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        input_shape = input.size()
        input = input.view(-1, self.num_features, -1)
        total_size = input.size(0) * input.size(2)
        input_sum = input.sum(dim=0).sum(dim=-1)
        input_ssum = (input ** 2).sum(dim=0).sum(dim=-1)
        if self._parallel_id == 0:
            mean, std = self._sync_manager.collect(input_sum, input_ssum, total_size)
        else:
            mean, std = self._child_registry.get(input_sum, input_ssum, total_size)

        mean = mean.unsqueeze(0).unsqueeze(-1)
        std = std.unsqueeze(0).unsqueeze(-1)

        output = ((input - mean) / std)
        if self.affine:
            output = output * self.weight.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
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
