# code is based the implementation of "bitsandbytes"
# see https://github.com/TimDettmers/bitsandbytes/tree/main/bitsandbytes

import ctypes as ct
import torch
from torch import Tensor
from typing import Optional, Tuple
from qtensor.cextension import get_lib


lib = get_lib()

lib_quan = {
    8 : (
        lib.cquantize_blockwise_8bit_fp32,
        lib.cquantize_blockwise_8bit_bf16,
    ),
    4 : (
        lib.cquantize_blockwise_4bit_fp32,
        lib.cquantize_blockwise_4bit_bf16,
    ),
}

lib_dequan = {
    8 : (
        lib.cdequantize_blockwise_8bit_fp32,
        lib.cdequantize_blockwise_8bit_bf16,
    ),
    4 : (
        lib.cdequantize_blockwise_4bit_fp32,
        lib.cdequantize_blockwise_4bit_bf16,
    ),
}

lib_quan_diagreal = {
    8 : (
        lib.cquantize_blockwise_diagreal_8bit_fp32,
        lib.cquantize_blockwise_diagreal_8bit_bf16,
    ),
    4 : (
        lib.cquantize_blockwise_diagreal_4bit_fp32,
        lib.cquantize_blockwise_diagreal_4bit_bf16,
    ),
}

lib_dequan_diagreal = {
    8 : (
        lib.cdequantize_blockwise_diagreal_8bit_fp32,
        lib.cdequantize_blockwise_diagreal_8bit_bf16,
    ),
    4 : (
        lib.cdequantize_blockwise_diagreal_4bit_fp32,
        lib.cdequantize_blockwise_diagreal_4bit_bf16,
    ),
}


def get_ptr(A: Tensor) -> ct.c_void_p:
    """
    Get the ctypes pointer from a PyTorch Tensor.

    Parameters
    ----------
    A : torch.tensor
        The PyTorch tensor.

    Returns
    -------
    ctypes.c_void_p
    """
    if A is None:
        return None
    else:
        return ct.c_void_p(A.data.data_ptr())


def create_dynamic_map(signed=True, total_bits=8, power=1):
    data = []
    max_exponent_bits = total_bits - 1
    for i in range(max_exponent_bits):
        fraction_items = int((2 ** i + 1 if signed else 2 ** (i + 1) + 1))
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)
    data.sort()

    data = torch.Tensor(data)
    return data.sign() * data.abs().pow(power)


def create_linear_map(signed=True, total_bits=8, power=2):
    data = torch.linspace(-1, 1, (2 ** total_bits))
    data[2 ** (total_bits-1) - 1] = 0

    return data.sign() * data.abs().pow(power)


def quantize_blockwise(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 256,
    bits: int = 8,
) -> Tuple[Tensor, Tensor]:
#    assert bits in [8, 4]

    if absmax is None:
        blocks = order // blocksize
        blocks += 1 if order % blocksize > 0 else 0
        blocks *= order
        absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)

    if out is None:
        n = order * order
        m = 8 // bits
        out_numel = n // m
        out_numel += 1 if n % m > 0 else 0
        out = torch.empty(out_numel, device=A.device, dtype=torch.uint8)

    if A.device.type != 'cpu':
        assert blocksize in [2048, 1024, 512, 256, 128, 64]
        assert code.shape == torch.Size([2 ** bits])
        cblocksize = ct.c_int32(blocksize)
        corder = ct.c_int32(order)

        if A.dtype == torch.float32:
            quan_func = lib_quan[bits][0]
        elif A.dtype == torch.bfloat16:
            quan_func = lib_quan[bits][1]
        else:
            raise ValueError(f"data type of A is not supported: {A.dtype}")

        quan_func(get_ptr(A), get_ptr(code), corder, get_ptr(absmax), get_ptr(out), cblocksize)
    else:
        raise NotImplementedError('quantize_blockwise on cpu is not supported')

    return out, absmax


def dequantize_blockwise(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Tensor,
    outdtype = torch.float32,
    out: Optional[Tensor] = None,
    blocksize: int = 256,
    bits: int = 8,
) -> Tensor:
#    assert bits in [8, 4]

    if out is None:
        out = torch.empty((order, order), device=A.device, dtype=outdtype)

    if A.device.type != 'cpu':
        assert blocksize in [2048, 1024, 512, 256, 128, 64]
        assert code.shape == torch.Size([2 ** bits])
        cblocksize = ct.c_int32(blocksize)
        corder = ct.c_int32(order)

        if out.dtype == torch.float32:
            dequan_func = lib_dequan[bits][0]
        elif out.dtype == torch.bfloat16:
            dequan_func = lib_dequan[bits][1]
        else:
            raise ValueError(f"data type of out is not supported: {out.dtype}")

        dequan_func(get_ptr(A), get_ptr(code), corder, get_ptr(absmax), get_ptr(out), cblocksize)
    else:
        raise NotImplementedError('dequantize_blockwise on cpu is not supported')

    return out


def quantize_blockwise_diagreal(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Optional[Tensor] = None,
    diag: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 256,
    bits: int = 8,
) -> Tuple[Tensor, Tensor, Tensor]:
#    assert bits in [8, 4]

    if absmax is None:
        blocks = order // blocksize
        blocks += 1 if order % blocksize > 0 else 0
        blocks *= order
        absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)

    if diag is None:
        diag = torch.empty((order,), device=A.device, dtype=torch.float32)

    if out is None:
        n = order * order
        m = 8 // bits
        out_numel = n // m
        out_numel += 1 if n % m > 0 else 0
        out = torch.empty(out_numel, device=A.device, dtype=torch.uint8)

    if A.device.type != 'cpu':
        assert blocksize in [2048, 1024, 512, 256, 128, 64]
        assert code.shape == torch.Size([2 ** bits])
        cblocksize = ct.c_int32(blocksize)
        corder = ct.c_int32(order)

        if A.dtype == torch.float32:
            quan_func = lib_quan_diagreal[bits][0]
        elif A.dtype == torch.bfloat16:
            quan_func = lib_quan_diagreal[bits][1]
        else:
            raise ValueError(f"data type of A is not supported: {A.dtype}")

        quan_func(get_ptr(A), get_ptr(code), corder, get_ptr(absmax), get_ptr(diag), get_ptr(out), cblocksize)
    else:
        raise NotImplementedError('quantize_blockwise_diagreal on cpu is not supported')

    return out, absmax, diag


def dequantize_blockwise_diagreal(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Tensor,
    diag: Tensor,
    outdtype = torch.float32,
    out: Optional[Tensor] = None,
    blocksize: int = 256,
    bits: int = 8,
) -> Tensor:
#    assert bits in [8, 4]

    if out is None:
        out = torch.empty((order, order), device=A.device, dtype=outdtype)

    if A.device.type != 'cpu':
        assert blocksize in [2048, 1024, 512, 256, 128, 64]
        assert code.shape == torch.Size([2 ** bits])
        cblocksize = ct.c_int32(blocksize)
        corder = ct.c_int32(order)

        if out.dtype == torch.float32:
            dequan_func = lib_dequan_diagreal[bits][0]
        elif out.dtype == torch.bfloat16:
            dequan_func = lib_dequan_diagreal[bits][1]
        else:
            raise ValueError(f"data type of out is not supported: {out.dtype}")

        dequan_func(get_ptr(A), get_ptr(code), corder, get_ptr(absmax), get_ptr(diag), get_ptr(out), cblocksize)
    else:
        raise NotImplementedError('dequantize_blockwise_diagreal on cpu is not supported')

    return out


@torch.no_grad()
def compute_power(Vt, S, p, iter_count=4, ridge_epsilon=1e-6):
    for j in range(iter_count):
        Vt = 1.5 * Vt - 0.5 * Vt @ Vt.T @ Vt
    rho = ridge_epsilon * S.max()

    return Vt.T @ (1 / (S + rho).pow(1 / p)).diag() @ Vt
