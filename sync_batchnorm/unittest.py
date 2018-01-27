# -*- coding: utf-8 -*-
# File   : unittest.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
# 
# This file is part of Synchronized-BatchNorm-PyTorch.

import unittest

import numpy as np
from torch.autograd import Variable


def as_numpy(v):
    if isinstance(v, Variable):
        v = v.data
    return v.cpu().numpy()


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, a, b, atol=1e-6):
        self.assertTrue(np.allclose(as_numpy(a), as_numpy(b), atol=atol),
                        'Tensor close check failed\n{}\n{}'.format(a, b))
