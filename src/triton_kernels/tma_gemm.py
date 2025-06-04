import triton
import triton.language as tl
import numpy as np
import torch
from triton_kernels.triton_util import check_tensors_gpu_ready

@triton.jit
def _dropout(x, offsets, seed, p):
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    return output.to(tl.float8e4nv)

@triton.jit
def gemm_kernel_tma(a_desc_ptr, b_desc_ptr, c_desc_ptr,  #
                      prob_m, prob_n, prob_k, drop_p, seed, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr, apply_sigmoid: tl.constexpr, apply_dropout: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(prob_m, block_m)
    num_pid_k = tl.cdiv(prob_k, block_k)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * block_m
    offs_bn = pid_n * block_n
    offs_k = 0

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for kk in range(0, num_pid_k):

        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [block_m, block_k], tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [block_n, block_k], tl.float8e4nv)
        
        if apply_dropout:
            offs_0 = offs_am + tl.arange(0, block_m)
            offs_1 = offs_k + tl.arange(0, block_k)
            offs_a = prob_k * offs_0[:, None] + offs_1[None, :]
            a = _dropout(a, offs_a, seed, drop_p)
        accumulator = tl.dot(a, b.T, acc=accumulator, out_dtype=tl.float32)
        offs_k += block_k
    if apply_sigmoid:
        accumulator = tl.sigmoid(accumulator)
    accumulator = accumulator.to(tl.bfloat16)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


def fp8_tma_matmul(a, b, bs=16, apply_sigmoid=False, apply_dropout=False, seed=40, drop_p=0.1, config=None):

    m, _ = a.shape
    n, k = b.shape
    assert a.shape[1] == b.shape[1]
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn

    if config:
        block_m = config["block_m"]
        block_n = config["block_n"]
        block_k = config["block_k"]
        num_warps = config["num_warps"]
        num_stages = config["num_stages"]
    
    block_m = bs*4
    block_n = bs*4
    block_k = bs*16
    num_warps = 4
    num_stages = 4
    TMA_SIZE = 512

  
    desc_a = torch.empty(TMA_SIZE, device="cpu")
    desc_b = torch.empty(TMA_SIZE, device="cpu")
    desc_c = torch.empty(TMA_SIZE, device="cpu")

    c = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    check_tensors_gpu_ready(a, b, c)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), m, k, block_m, block_k, a.element_size(), desc_a.data_ptr())
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), n, k, block_n, block_k, b.element_size(), desc_b.data_ptr())
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(c.data_ptr(), m, n, block_m, block_n, c.element_size(), desc_c.data_ptr())
    desc_a = desc_a.cuda()
    desc_b = desc_b.cuda()
    desc_c = desc_c.cuda()
    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    
    grid = (total_blocks_m * total_blocks_n, 1, 1)
    k = gemm_kernel_tma[grid](
        desc_a, desc_b, desc_c,
        m, n, k,
        drop_p, seed,
        block_m,
        block_n,
        block_k,
        apply_sigmoid=apply_sigmoid,
        apply_dropout=apply_dropout,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # with open('tma_fp8.ttgir', 'w') as f:
    #      print(k.asm['ttgir'], file=f)

    # with open('tma_fp8.ptx', 'w') as f:
    #      print(k.asm['ptx'], file=f)

    return c


# if __name__ == '__main__':

#     M = 128
#     N = 4096
#     K = 4096

#     a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
#     b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
#     b = b.T.contiguous()

#     c = fp8_tma_matmul(a, b, apply_dropout=True, seed=10, drop_p=0.2)
#     print(c[10:12, 20:30])
#     bf16 = torch.mm(a.to(torch.bfloat16), b.T.to(torch.bfloat16))
#     print(bf16[10:12, 20:30])

