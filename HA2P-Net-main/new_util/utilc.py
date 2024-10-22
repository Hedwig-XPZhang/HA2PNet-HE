# @auther:guopeng
# @time:2023/9/11 11:02
# @file:utilc.py
# @description:

from string import Template
from collections import namedtuple
import torch
import cupy

Stream = namedtuple('Stream', ['ptr'])

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)