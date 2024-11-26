# code is based the implementation of "bitsandbytes"
# see https://github.com/TimDettmers/bitsandbytes/tree/main/bitsandbytes

import torch
import platform
import ctypes as ct
from pathlib import Path


DYNAMIC_LIBRARY_SUFFIX = {"Windows": ".dll", "Linux": ".so"}.get(platform.system())


def get_cuda_version():
    major, minor = map(int, torch.version.cuda.split("."))
    return f'{major}{minor}'


def get_lib():
    cuda_version_string = get_cuda_version()
    binary_name = f"libqtensor_cuda{cuda_version_string}{DYNAMIC_LIBRARY_SUFFIX}"
    binary_path = str(Path(__file__).parent / binary_name)
    return ct.cdll.LoadLibrary(binary_path.replace('\\', '/'))
