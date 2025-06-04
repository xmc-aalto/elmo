import torch
import os
#os.environ['TRITON_INTERPRET'] = '1'
import triton
import triton.language as tl
from triton_kernels.triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready



@triton.jit
def stochastic_rounding_to_bf16(source, seed, offs_out):
    rand = tl.randint(seed, offs_out) & 65535
    out = source.to(tl.int32, bitcast=True) + rand
    out = out & -65536
    out = out.to(tl.float32, bitcast=True) 
    out = out.to(tl.bfloat16)
    return out

# @triton.jit
# def stochastic_rounding_to_fp8(source, seed, offs_out):
#     rand = tl.randint(seed, offs_out) & 1048575
#     out = source.to(tl.int32, bitcast=True) + rand
#     out = out & -1048576
#     out = out.to(tl.float32, bitcast=True) 
#     out = out.to(tl.float8e4nv)
#     return out

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
def stochastic_rounding_to_fakefp8(source, seed, offs_out):
    rand = tl.randint(seed, offs_out) & 1048575
    out = source.to(tl.int32, bitcast=True) + rand
    out = out & -1048576
    out = out.to(tl.float32, bitcast=True) 
    return out.to(tl.bfloat16)

@triton.jit
def get_1d_offset(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)

@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1): 
    return tl.expand_dims(offs_0, 1)*stride_0 + tl.expand_dims(offs_1, 0)*stride_1

@triton.jit
def get_1d_mask(offs, max):
    return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)


@triton.jit
def naive_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    lr, seed,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0).to(tl.int64), tl.program_id(1)
    # chunks along m/n/k dimensions
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
    rk = get_1d_offset(size=bk, n_prev_chunks=0)
    # relevant offsets of a, b
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
    # initialize and iteratively update accumulator
    offs_k = tl.arange(0, bk)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for kk in range(0, k, bk):
        a_mask = tl.expand_dims(offs_k, 0) + kk < k
        b_mask = tl.expand_dims(offs_k, 1) + kk < k

        a = tl.load(offs_a, mask=a_mask, other=0.0)
        a = a.to(tl.float32).to(tl.bfloat16)

        b = tl.load(offs_b, mask=b_mask, other=0.0)
        b = b.to(tl.float32).to(tl.bfloat16)

        acc += tl.dot(a, b, allow_tf32=False) # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offets, so next iteration loads next chunks
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    c_offs = get_2d_offset(rm, rn, stride_cm, stride_cn)
    c_ptrs = c_ptr + c_offs
    mask_c = get_2d_mask(rm, rn, m, n)
 
    weights = tl.load(c_ptrs, mask=mask_c, other=0.0)
    weights = weights.to(tl.float32)

    weights = weights - lr*acc

    c_offs = c_offs.to(tl.int32)
    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        weights = stochastic_rounding_to_fp8(weights, seed, c_offs)
    else:
        weights = stochastic_rounding_to_bf16(weights, seed, c_offs)

    tl.store(c_ptrs, weights, mask=mask_c)


def matmul_update(a, b, c, lr, seed, bs=16):
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    check_tensors_gpu_ready(a, b, c)
    (m, k), (_, n) = a.shape, b.shape
    assert c.shape == (m, n)
    assert c.device == a.device
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    naive_matmul_k[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        lr, seed,
        bm=bs*8, bn=bs*4, bk=bs*4, # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    )


@triton.jit
def fakefp8_matmul_update_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    lr, seed,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0).to(tl.int64), tl.program_id(1)
    # chunks along m/n/k dimensions
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
    rk = get_1d_offset(size=bk, n_prev_chunks=0)
    # relevant offsets of a, b
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
    # initialize and iteratively update accumulator
    offs_k = tl.arange(0, bk)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for kk in range(0, k, bk):
        a_mask = tl.expand_dims(offs_k, 0) + kk < k
        b_mask = tl.expand_dims(offs_k, 1) + kk < k

        a = tl.load(offs_a, mask=a_mask, other=0.0)
        a = a.to(tl.bfloat16)

        b = tl.load(offs_b, mask=b_mask, other=0.0)
        b = b.to(tl.bfloat16)

        acc += tl.dot(a, b, allow_tf32=False) # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offets, so next iteration loads next chunks
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    c_offs = get_2d_offset(rm, rn, stride_cm, stride_cn)
    c_ptrs = c_ptr + c_offs
    mask_c = get_2d_mask(rm, rn, m, n)
 
    weights = tl.load(c_ptrs, mask=mask_c, other=0.0)
    weights = weights.to(tl.float32)

    weights = weights - lr*acc

    c_offs = c_offs.to(tl.int32)
    
    weights = stochastic_rounding_to_fakefp8(weights, seed, c_offs)

    tl.store(c_ptrs, weights, mask=mask_c)

def fakefp8_matmul_update(a, b, c, lr, seed, bs=16):
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    check_tensors_gpu_ready(a, b, c)
    (m, k), (_, n) = a.shape, b.shape
    assert c.shape == (m, n)
    assert c.device == a.device
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    fakefp8_matmul_update_k[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        lr, seed,
        bm=bs*8, bn=bs*4, bk=bs*4, # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    )


@triton.jit
def kahan_matmul_update_k(
    a_ptr, b_ptr, c_ptr, d_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    lr, seed,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0).to(tl.int64), tl.program_id(1)
    # chunks along m/n/k dimensions
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
    rk = get_1d_offset(size=bk, n_prev_chunks=0)
    # relevant offsets of a, b
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
    # initialize and iteratively update accumulator
    offs_k = tl.arange(0, bk)
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for kk in range(0, k, bk):
        a_mask = tl.expand_dims(offs_k, 0) + kk < k
        b_mask = tl.expand_dims(offs_k, 1) + kk < k

        a = tl.load(offs_a, mask=a_mask, other=0.0)
        a = a.to(tl.float32).to(tl.bfloat16)

        b = tl.load(offs_b, mask=b_mask, other=0.0)
        b = b.to(tl.float32).to(tl.bfloat16)

        acc += tl.dot(a, b, allow_tf32=False) # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offets, so next iteration loads next chunks
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    c_offs = get_2d_offset(rm, rn, stride_cm, stride_cn)
    c_ptrs = c_ptr + c_offs
    mask_c = get_2d_mask(rm, rn, m, n)
 
    weights = tl.load(c_ptrs, mask=mask_c, other=0.0)
    weights = weights.to(tl.float32)
    weights_f32_old = weights

    # Kahan correction
    d_offs = get_2d_offset(rm, rn, stride_cm, stride_cn)
    d_ptrs = d_ptr + d_offs
    mask_d = get_2d_mask(rm, rn, m, n)

    kahan = tl.load(d_ptrs, mask=mask_d, other=0.0)
    weights = weights + kahan
    weights = weights - lr*acc

    c_offs = c_offs.to(tl.int32)
    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        weights = stochastic_rounding_to_fp8(weights, seed, c_offs)
    else:
        weights = stochastic_rounding_to_bf16(weights, seed, c_offs)
    
    weights_f32_new = weights.to(tl.float32)
    kahan = kahan + weights_f32_old - weights_f32_new - lr * acc
    if (d_ptr.dtype.element_ty == tl.float8e5):
        kahan = kahan.to(tl.float8e5)
    else:
        kahan = kahan.to(tl.bfloat16)

    tl.store(d_ptrs, kahan, mask=mask_d)
    tl.store(c_ptrs, weights, mask=mask_c)

def kahan_matmul_update(a, b, c, d, lr, seed, bs=16):
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    check_tensors_gpu_ready(a, b, c, d)
    (m, k), (_, n) = a.shape, b.shape
    assert c.shape == (m, n)
    assert d.shape == (m, n)
    assert c.device == a.device
    assert d.device == a.device
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    kahan_matmul_update_k[grid](
        a, b, c, d,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        lr, seed,
        bm=bs*8, bn=bs*4, bk=bs*4, # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    )

def test_matmul():
    lr = 0.01
    seed = 1024

    m = 13
    n = 19
    batch_size = 17

    x = torch.randn(m, batch_size)
    x = x.to(torch.bfloat16)
    x = x.cuda()
    

    w = torch.randn(batch_size, n)
    w = w.to(torch.bfloat16) # (torch.float8_e4m3fn)
    w = w.cuda()

    weights = torch.randn((m, n), device=x.device, dtype=torch.bfloat16)
    weights = weights.to(torch.float8_e4m3fn)

    bf16_out = weights.to(torch.float32) - lr * torch.mm(x.to(torch.float32), w.to(torch.float32))
    bf16_out = bf16_out.to(torch.bfloat16)

    matmul_update(x, w, weights, lr, seed)

    print(bf16_out - weights.to(torch.bfloat16))
    print(torch.all(weights.to(torch.bfloat16) == bf16_out))

#test_matmul()
