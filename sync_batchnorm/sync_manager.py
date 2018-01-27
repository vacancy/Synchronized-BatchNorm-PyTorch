# -*- coding: utf-8 -*-
# File   : sync_manager.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.

import queue
import collections
import threading

from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

__all__ = ['FutureResult', 'ChildRegistry', 'SyncManager']


class FutureResult(object):
    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_ManagerRegistry = collections.namedtuple('ManagerRegistry', ['result'])
_ChildRegistryBase = collections.namedtuple('_ChildRegistryBase', ['queue', 'result'])


class ChildRegistry(_ChildRegistryBase):
    def get(self, sum_, ssum, total_size):
        self.queue.put((sum_, ssum, total_size))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncManager(object):
    def __init__(self, batch_norm):
        self._batch_norm = batch_norm
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def register(self, identifier):
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _ManagerRegistry(future)
        return ChildRegistry(self._queue, future)

    def collect(self, mine_sum, mine_ssum, mine_size):
        self._activated = True

        intermediate = [(mine_sum, mine_ssum, mine_size)]
        to_reduce = [mine_sum, mine_ssum]
        for i in range(self.nr_children):
            intermediate.append(self._queue.get())
            to_reduce.extend(intermediate[-1][:2])

        target_gpus = [i[0].get_device() for i in intermediate]

        sum_, ssum = ReduceAddCoalesced.apply(mine_sum.get_device(), 1, *to_reduce)
        total_size = sum([i[2] for i in intermediate])
        mean, std = self._batch_norm.compute_mean_std(sum_, ssum, total_size)

        broadcasted = Broadcast.apply(target_gpus, mean, std)

        for v, (mean, std) in zip(self._registry.values(), broadcasted[1:]):
            v.result.put((mean, std))
        for i in range(self.nr_children):
            assert self._queue.get() is True

        return broadcasted[0]

    @property
    def nr_children(self):
        return len(self._registry)
