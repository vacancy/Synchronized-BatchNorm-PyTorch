# -*- coding: utf-8 -*-
# File   : replicate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.

from torch.nn.parallel.data_parallel import DataParallel

__all__ = ['CallbackContext', 'execute_callbacks', 'DataParallelWithCallback']


class CallbackContext(object):
    pass


def execute_callbacks(modules):
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_callbacks(modules)
        return modules

