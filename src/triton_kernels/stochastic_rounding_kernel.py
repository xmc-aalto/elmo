import torch
import os
#os.environ['TRITON_INTERPRET'] = '1'
import triton
import triton.language as tl
from typing import Optional
from torch import Tensor


@triton.jit
def stochastic_rounding_to_bf16(source, seed, offs_out):
    rand = tl.randint(seed, offs_out) & 65535
    out = source.to(tl.int32, bitcast=True) + rand
    out = out.to(tl.uint32, bitcast=True)
    out = out & 4294901760
    out = out.to(tl.float32, bitcast=True) 
    out = out.to(tl.bfloat16)
    return out

@triton.jit
def stochastic_rounding_to_fp8(source, seed, offs_out):
    rand = tl.randint(seed, offs_out) & 1048575
    out = source.to(tl.int32, bitcast=True) + rand
    out = out.to(tl.uint32, bitcast=True)
    out = out & 4293918720
    out = out.to(tl.float32, bitcast=True) 
    out = out.to(tl.float8e4nv)
    return out


@triton.jit
def _seeded_stochastic_rounding(
    x_ptr,
    output_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if (output_ptr.dtype.element_ty == tl.float8e4nv):
        output = stochastic_rounding_to_fp8(x, seed, offsets)
    else:
        output = stochastic_rounding_to_bf16(x, seed, offsets)
    tl.store(output_ptr + offsets, output, mask=mask)



def stochastic_rounding(x, output, seed):
    assert x.is_contiguous()
    assert output.is_contiguous()
    assert x.shape == output.shape
    assert x.dtype == torch.float32
    assert output.dtype != torch.float32
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_stochastic_rounding[grid](x, output, n_elements, seed, BLOCK_SIZE=1024)

@triton.jit
def _sgd(
    weights_ptr,
    gradient_ptr,
    n_elements,
    lr,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data
    mask = offsets < n_elements
    W = tl.load(weights_ptr + offsets, mask=mask)
    grad = tl.load(gradient_ptr + offsets, mask=mask)
    W = W.to(tl.float32)
    W = W - grad*lr

    if (weights_ptr.dtype.element_ty == tl.float8e4nv):
        W = stochastic_rounding_to_fp8(W, seed, offsets)
    else:
        W = stochastic_rounding_to_bf16(W, seed, offsets)
    tl.store(weights_ptr + offsets, W, mask=mask)

def sgd_update(weights: Tensor, gradient: Tensor, compensation: Optional[Tensor],
               learning_rate: float, weight_decay: float, stochastic_rounding: bool, seed: int):
    assert weights.is_contiguous()
    assert weights.dtype != torch.float32
    assert gradient.is_contiguous()
    assert gradient.shape == weights.shape
    n_elements = weights.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _sgd[grid](weights, gradient, n_elements, learning_rate, seed, BLOCK_SIZE=1024)
    
def test_stochastic_rounding():
    x = torch.randn(3, 5)
    x = x.cuda()
    fp8_x = stochastic_rounding(x, 123, torch.float8_e4m3fn)
    bf16_x = stochastic_rounding(x, 123, torch.bfloat16)

    print(x)
    print(fp8_x)
    print(bf16_x)    
    

# test_stochastic_rounding()
