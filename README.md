# Synchronized-BatchNorm-PyTorch

Synchronized Batch Normalization implementation in PyTorch.

This module differs from the built-in PyTorch BatchNorm as the mean and
standard-deviation are reduced across all devices during training.

For example, when one uses `nn.DataParallel` to wrap the network during
training, PyTorch's implementation normalize the tensor on each device using
the statistics only on that device, which accelerated the computation and
is also easy to implement, but the statistics might be inaccurate.
Instead, in this synchronized version, the statistics will be computed
over all training samples distributed on multiple devices.

Note that, for one-GPU or CPU-only case, this module behaves exactly same
as the built-in PyTorch implementation.

This module is currently only a prototype version for research usages. As mentioned below,
it has its limitations and may even suffer from some design problems. If you have any
questions or suggestions, please feel free to
[open an issue](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues) or 
[submit a pull request](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues).

## Why Synchronized BatchNorm?

Although the typical implementation of BatchNorm working on multiple devices (GPUs)
is fast (with no communication overhead), it inevitably reduces the size of batch size,
which potentially degenerates the performance. This is not a significant issue in some
standard vision tasks such as ImageNet classification (as the batch size per device
is usually large enough to obtain good statistics). However, it will hurt the performance
in some tasks that the batch size is usually very small (e.g., 1 per GPU).

For example, the importance of synchronized batch normalization in object detection has been recently proved with a
an extensive analysis in the paper [MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/abs/1711.07240).

## Usage

To use the Synchronized Batch Normalization, we add a data parallel replication callback. This introduces a slight
difference with typical usage of the `nn.DataParallel`.

Use it with a provided, customized data parallel wrapper:

```python
from sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback

sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
```

Or, if you are using a customized data parallel module, you can use this library as a monkey patching.

```python
from torch.nn import DataParallel  # or your customized DataParallel module
from sync_batchnorm import SynchronizedBatchNorm1d, patch_replication_callback

sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
patch_replication_callback(sync_bn)  # monkey-patching
```

You can use `convert_model` to convert your model to use Synchronized BatchNorm easily.

```python
import torch.nn as nn
from torchvision import models
from sync_batchnorm import convert_model
# m is a standard pytorch model
m = models.resnet18(True)
m = nn.DataParallel(m)
# after convert, m is using SyncBN
m = convert_model(m)
```

See also `tests/test_sync_batchnorm.py` for numeric result comparison.

## Implementation details and highlights

If you are interested in how batch statistics are reduced and broadcasted among multiple devices, please take a look
at the code with detailed comments. Here we only emphasize some highlights of the implementation:

- This implementation is in pure-python. No C++ extra extension libs.
- Easy to use as demonstrated above.
- It is completely compatible with PyTorch's implementation. Specifically, it uses unbiased variance to update the
moving average, and use `sqrt(max(var, eps))` instead of `sqrt(var + eps)`.
- The implementation requires that each module on different devices should invoke the `batchnorm` for exactly SAME
amount of times in each forward pass. For example, you can not only call `batchnorm` on GPU0 but not on GPU1. The `#i
(i = 1, 2, 3, ...)` calls of the `batchnorm` on each device will be viewed as a whole and the statistics will be reduced.
This is tricky but is a good way to handle PyTorch's dynamic computation graph. Although sounds complicated, this
will usually not be the issue for most of the models.

## Known issues

#### Runtime error on backward pass.

Due to a [PyTorch Bug](https://github.com/pytorch/pytorch/issues/3883), using old PyTorch libraries will trigger an `RuntimeError` with messages like:

```
Assertion `pos >= 0 && pos < buffer.size()` failed.
```

This has already been solved in the newest PyTorch repo, which, unfortunately, has not been pushed to the official and anaconda binary release. Thus, you are required to build the PyTorch package from the source according to the
 instructions [here](https://github.com/pytorch/pytorch#from-source).

#### Numeric error.

Because this library does not fuse the normalization and statistics operations in C++ (nor CUDA), it is less
numerically stable compared to the original PyTorch implementation. Detailed analysis can be found in
`tests/test_sync_batchnorm.py`.

## Authors and License:

Copyright (c) 2018-, [Jiayuan Mao](https://vccy.xyz).

**Contributors**: [Tete Xiao](https://tetexiao.com), [DTennant](https://github.com/DTennant).

Distributed under **MIT License** (See LICENSE)

