import torch
import os
#os.environ['TRITON_INTERPRET'] = '1'
import triton
import triton.language as tl
from triton_kernels.triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready

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
def fp8_naive_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr,
    transpose: tl.constexpr, apply_sigmoid: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0).to(tl.int64), tl.program_id(1)
    # chunks along m/n/k dimensions
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
    rk = get_1d_offset(size=bk, n_prev_chunks=0)
    # relevant offsets of a, b
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
    offs_k = tl.arange(0, bk)
    # initialize and iteratively update accumulator
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for kk in range(0, k, bk):
        a_mask = tl.expand_dims(offs_k, 0) + kk < k
        b_mask = tl.expand_dims(offs_k, 1) + kk < k

        a = tl.load(offs_a, mask=a_mask, other=0.0)
        #a = a.to(tl.bfloat16)

        b = tl.load(offs_b, mask=b_mask, other=0.0)
        #b = b.to(tl.bfloat16)

        acc += tl.dot(a, b, allow_tf32=False) # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offets, so next iteration loads next chunks
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    
    if apply_sigmoid == True:
        acc = tl.sigmoid(acc)
    acc = acc.to(tl.bfloat16)
    
    offs_c = get_2d_offset(rm, rn, stride_cm, stride_cn) 
    
    mask = get_2d_mask(rm, rn, m, n)
    
    if transpose == True:
        acc = tl.trans(acc)
        offs_c = tl.trans(offs_c)
        mask = tl.trans(mask)

    tl.store(c_ptr + offs_c, acc, mask=mask)

@triton.jit
def bf16_fp8_naive_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr,
    transpose: tl.constexpr, apply_sigmoid: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    # chunks along m/n/k dimensions
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
    rk = get_1d_offset(size=bk, n_prev_chunks=0)
    # relevant offsets of a, b
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
    # initialize and iteratively update accumulator
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for kk in range(0, k, bk):
        a_mask = tl.expand_dims(rk, 0) + kk < k
        b_mask = tl.expand_dims(rk, 1) + kk < k

        a = tl.load(offs_a, mask=a_mask, other=0.0)
        a = a.to(tl.bfloat16)

        b = tl.load(offs_b, mask=b_mask, other=0.0)
        b = b.to(tl.bfloat16)

        acc += tl.dot(a, b, allow_tf32=False) # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offets, so next iteration loads next chunks
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    
    if apply_sigmoid == True:
        acc = tl.sigmoid(acc)
    acc = acc.to(tl.bfloat16)
    
    offs_c = get_2d_offset(rm, rn, stride_cm, stride_cn) 
    mask = get_2d_mask(rm, rn, m, n)
    
    if transpose == True:
        acc = tl.trans(acc)
        offs_c = tl.trans(offs_c)
        mask = tl.trans(mask)

    tl.store(c_ptr + offs_c, acc, mask=mask)

@triton.jit
def _fast_matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    acc_dtype: tl.constexpr,  #
    input_precision: tl.constexpr,  #
    fp8_fast_accum: tl.constexpr,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  #
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,  #
    apply_sigmoid: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, allow_tf32=False)
        else:
            acc += tl.dot(a, b, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    if apply_sigmoid == True:
        acc = tl.sigmoid(acc)
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)

@triton.jit
def large_k_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    bk: tl.constexpr, bm: tl.constexpr, bn: tl.constexpr,
):
    pid_k, pid_m, pid_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # chunks along m/n/k dimensions
    
    offs_k = pid_k * bk + tl.arange(0, bk)  
    offs_m = pid_m * bm + tl.arange(0, bm)
    offs_n = pid_n * bn + tl.arange(0, bn)

 
    offs_a = m * offs_k[:,None] + offs_m[None, :]
    offs_b = n * offs_k[:,None] + offs_n[None, :]
    offs_c = n * offs_m[:,None] + offs_n[None, :] 

    mask_k = offs_k < k
    mask_m = offs_m < m
    mask_n = offs_n < n

    mask_a = mask_k[:,None] & mask_m[None,:]
    mask_b = mask_k[:,None] & mask_n[None,:]
    mask_c = mask_m[:,None] & mask_n[None,:]
   
    a = tl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
    a = a.to(tl.float32).to(tl.bfloat16)

    b = tl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
    b = b.to(tl.float32).to(tl.bfloat16)

    accumulator = tl.zeros((bm, bn), dtype=tl.float32)
    accumulator = tl.dot(a.trans(), b, acc=accumulator, out_dtype=tl.float32, allow_tf32=False)

    tl.atomic_add(c_ptr + offs_c, accumulator, mask=mask_c)

def fast_bf16_matmul(a, b, bs=16, apply_sigmoid=False):
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    assert a.dtype == torch.bfloat16
    assert b.dtype == torch.bfloat16
    (M, K), (_, N) = a.shape, b.shape 
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    check_tensors_gpu_ready(a, b, c)
    input_precision = "tf32"
    ab_dtype = tl.bfloat16
    acc_dtype = tl.float32
    fp8_fast_accum = False
    block_k = bs
    split_k = 4
    block_m = bs*4
    block_n = bs
    even_k = K % (block_k * split_k) == 0
    grid = lambda META: (
        cdiv(M, META["BLOCK_M"]) * cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    _fast_matmul_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,  #
            a.stride(0),
            a.stride(1),  #
            b.stride(0),
            b.stride(1),  #
            c.stride(0),
            c.stride(1),  #
            acc_dtype=acc_dtype,  #
            input_precision=input_precision,  #
            fp8_fast_accum=fp8_fast_accum,  #
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,  #
            GROUP_M=8,
            SPLIT_K=split_k,
            EVEN_K=even_k,
            AB_DTYPE=ab_dtype,
            apply_sigmoid=apply_sigmoid,
        )
    return c

def fp8_matmul(a, b, bs=16, transpose=False, apply_sigmoid=False):
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    (m, k), (_, n) = a.shape, b.shape  
    if transpose:
        c = torch.empty((n, m), device=a.device, dtype=torch.bfloat16)
        stride_c_0 = c.stride(1)
        stride_c_1 = c.stride(0)
    else:
        c = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        stride_c_0 = c.stride(0)
        stride_c_1 = c.stride(1)
    check_tensors_gpu_ready(a, b, c)
    
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    fp8_naive_matmul_k[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        stride_c_0, stride_c_1,
        bm=bs, bn=bs, bk=bs,
        transpose=transpose,
        apply_sigmoid=apply_sigmoid # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    )
    return c
 
def bf16_fp8_matmul(a, b, bs=16, transpose=False, apply_sigmoid=False):
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    (m, k), (_, n) = a.shape, b.shape  
    if transpose:
        c = torch.empty((n, m), device=a.device, dtype=torch.bfloat16)
        stride_c_0 = c.stride(1)
        stride_c_1 = c.stride(0)
    else:
        c = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        stride_c_0 = c.stride(0)
        stride_c_1 = c.stride(1)
    check_tensors_gpu_ready(a, b, c)
    
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    bf16_fp8_naive_matmul_k[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        stride_c_0, stride_c_1,
        bm=bs, bn=bs, bk=64,
        transpose=transpose,
        apply_sigmoid=apply_sigmoid # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    )
    return c

def large_k_matmul(a, b, bs):
    assert a.shape[0] == b.shape[0], "matrix dims not compatible for large k matmul"
    (k, m), (_, n) = a.shape, b.shape  
    c = torch.zeros((m, n), device=a.device, dtype=torch.float32)
    check_tensors_gpu_ready(a, b, c)
    grid = lambda meta: (triton.cdiv(k, meta['bk']), triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    large_k_matmul_kernel[grid](
        a, b, c,
        m, n, k,
        bk=bs*16, bm=bs*4, bn=bs*4,
    )
    return c

def test_fp8matmul():
    num_labels = 65536
    batch_size = 768
    x = torch.randn(num_labels, batch_size) #2796192
    x = x.to(torch.float8_e4m3fn)
    x = x.cuda()

    w = torch.randn(batch_size, 32)
    w = w.to(torch.float8_e4m3fn) # (torch.float8_e4m3fn)
    w = w.cuda()

    transpose = True
    apply_sigmoid = False

    out = fp8_matmul(x, w, bs=16, transpose=transpose, apply_sigmoid=apply_sigmoid)
        
        
    #print(out[0:3, 5:10])

    bf16_out = torch.mm(x.to(torch.bfloat16), w.to(torch.bfloat16))
    if apply_sigmoid:
        torch.sigmoid(bf16_out, out=bf16_out)
    # bf16_out = bf16_out.to(torch.float8_e4m3fn)
    if transpose:
        print(torch.all(out == bf16_out.t()))
    else:
        print(bf16_out[0:3, 5:10])
        print(out[0:3, 5:10])
        print(torch.all(out == bf16_out))

def test_bf16fp8matmul():
    num_labels = 25
    batch_size = 19
    x = torch.randn(num_labels, batch_size) #2796192
    #x = x.to(torch.bfloat16)
    x = x.cuda()

    w = torch.randn(batch_size, 17)
    #w = w.to(torch.float8_e4m3fn) # (torch.float8_e4m3fn)
    w = w.cuda()

    transpose = False
    apply_sigmoid = False

    out = bf16_fp8_matmul(x, w, bs=16, transpose=transpose, apply_sigmoid=apply_sigmoid)
        
        
    #print(out[0:3, 5:10])

    bf16_out = torch.mm(x.to(torch.bfloat16), w.to(torch.bfloat16))
    if apply_sigmoid:
        torch.sigmoid(bf16_out, out=bf16_out)
    # bf16_out = bf16_out.to(torch.float8_e4m3fn)
    if transpose:
        print(torch.all(out == bf16_out.t()))
    else:
        print(bf16_out[0:3, 5:10])
        print(out[0:3, 5:10])
        print(torch.all(out == bf16_out))

def test_bf16fp8largekmatmul():
    num_labels = 1000000
    batch_size = 5
    x = torch.randn(num_labels, batch_size) #2796192
    x = x.to(torch.bfloat16)
    x = x.cuda()

    w = torch.randn(num_labels, 6)
    w = w.to(torch.float8_e4m3fn) # (torch.float8_e4m3fn)
    w = w.cuda()

    out = large_k_matmul(x, w, bs=16)
        
    

    bf16_out = torch.mm(x.to(torch.float32).t(), w.to(torch.float32))
   
    print(bf16_out)
    print(out)
    print(bf16_out-out)
    print(torch.all(out == bf16_out))

def test_fast_fp8_matmul():
    num_labels = 19
    batch_size = 5
    x = torch.randn(batch_size, num_labels) #2796192
    x = x.to(torch.float8_e4m3fn)
    x = x.cuda()

    w = torch.randn(num_labels, 17)
    w = w.to(torch.float8_e4m3fn) # (torch.float8_e4m3fn)
    w = w.cuda()

    out = fast_fp8_matmul(x, w)
        
    

    bf16_out = torch.mm(x.to(torch.bfloat16), w.to(torch.bfloat16))
   
    print(bf16_out)
    print(out)
    print(bf16_out-out)
    print(torch.all(out == bf16_out))

def test_fast_bf16_matmul():
    num_labels = 19
    batch_size = 5
    x = torch.randn(batch_size, num_labels) #2796192
    x = x.to(torch.bfloat16)
    x = x.cuda()

    w = torch.randn(num_labels, 17)
    w = w.to(torch.bfloat16) # (torch.float8_e4m3fn)
    w = w.cuda()

    out = fast_bf16_matmul(x, w)
        
    

    bf16_out = torch.mm(x, w)
   
    print(bf16_out)
    print(out)
    print(bf16_out-out)
    print(torch.all(out == bf16_out))



# test_bf16fp8largekmatmul()
