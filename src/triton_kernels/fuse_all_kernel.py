import torch
import os
#os.environ['TRITON_INTERPRET'] = '1'
import triton
import triton.language as tl
from triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready



@triton.jit
def stochastic_rounding_to_bf16(source, seed, offs_out):
    rand = tl.randint(seed, offs_out) & 65535
    out = source.to(tl.int32, bitcast=True) + rand
    out = out & -65536
    out = out.to(tl.float32, bitcast=True) 
    out = out.to(tl.bfloat16)
    return out

@triton.jit
def stochastic_rounding_to_fp8(source, seed, offs_out):
    rand = tl.randint(seed, offs_out) & 1048575
    out = source.to(tl.int32, bitcast=True) + rand
    out = out & -1048576
    out = out.to(tl.float32, bitcast=True) 
    out = out.to(tl.float8e4nv)
    return out

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
    a_ptr, b_ptr, labels_ptr, b_grad_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    lr, seed,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    # chunks along m/n/k dimensions
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
    rk = get_1d_offset(size=bk, n_prev_chunks=0)
    # relevant offsets of a, b
    offs_a = get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = get_2d_offset(rk, rn, stride_bk, stride_bn)
    # initialize and iteratively update accumulator

    logits = tl.zeros((bm, bn), dtype=tl.float32)
    for kk in range(0, k, bk):
        a_mask = tl.expand_dims(tl.arange(0, bk), 0) + kk < k
        b_mask = tl.expand_dims(tl.arange(0, bk), 1) + kk < k

        a = tl.load(a_ptr + offs_a, mask=a_mask, other=0.0) # weights (block_num_labels, k/num_phases)
        b = tl.load(b_ptr + offs_b, mask=b_mask, other=0.0) # embed （k/num_phases, block_num_datapoints)

        logits += tl.dot(a, b) # (block_num_labels, block_num_datapoints)
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk

    c_offs = get_2d_offset(rm, rn, stride_cm, stride_cn)
    c_mask = get_2d_mask(rm, rn, m, n)

    true_labels = tl.load(labels_ptr + c_offs, mask=c_mask, other=0.0) # (block_num_labels, block_num_datapoints)
    logits = tl.sigmoid(logits) # (block_num_labels, block_num_datapoints)
    logits = logits - true_labels.to(tl.float32)

    logits = logits.to(tl.bfloat16)
    offs_a = get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = get_2d_offset(rk, rn, stride_bk, stride_bn)
    for kk in range(0, k, bk):
        a_mask = tl.expand_dims(tl.arange(0, bk), 0) + kk < k
        b_mask = tl.expand_dims(tl.arange(0, bk), 1) + kk < k

        offs_k = kk + tl.arange(0, bk)
        
        weights = tl.load(a_ptr + offs_a, mask=a_mask, other=0.0) # weights (block_num_labels, k/num_phases)
        
        b = tl.load(b_ptr + offs_b, mask=b_mask, other=0.0) # encoder embed transpose (k/num_phases, block_num_datapoints)
        b = b.to(tl.bfloat16)

        b_grad = tl.zeros((bn, bk), dtype=tl.float32)
        b_grad += tl.dot(logits.T, weights.to(tl.bfloat16)) # (block_num_datapoints, k/num_phases)

        
        b_grad_offs = k * rn[:,None] + offs_k[None, :]

        mask_k = offs_k < k
        mask_n = rn < n
        mask_b_grad = mask_n[:,None] & mask_k[None,:]

        tl.atomic_add(b_grad_ptr + b_grad_offs, b_grad, mask=mask_b_grad)

        w_grad = tl.dot(logits, b.T) #(block_num_labels, k/num_phases)

        weights = weights.to(tl.float32)
        weights = weights - lr*w_grad
        if (a_ptr.dtype.element_ty == tl.float8e4nv):
            weights = stochastic_rounding_to_fp8(weights, seed, offs_a)
        else:
            weights = stochastic_rounding_to_bf16(weights, seed, offs_a)
        tl.store(a_ptr + offs_a, weights, mask=a_mask)

        offs_a += bk * stride_ak
        offs_b += bk * stride_bk




def matmul_update(a, b, labels, b_grad, lr, seed, bs=16):
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    assert b.shape[0] == b_grad.shape[1]
    assert b.shape[1] == b_grad.shape[0]
    check_tensors_gpu_ready(a, b, labels, b_grad)
    (m, k), (_, n) = a.shape, b.shape
    assert labels.shape == (m, n)
    assert labels.device == a.device
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    naive_matmul_k[grid](
        a, b, labels, b_grad,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        labels.stride(0), labels.stride(1),
        lr, seed,
        bm=bs*4, bn=bs*2, bk=bs*2, # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    )


def test_matmul():
    lr = 0.01
    
    num_labels = 11320
    dim = 32
    batch_size = 16

    weight = torch.randn(num_labels, dim)
    weight = weight.to(torch.bfloat16) # (torch.float8_e4m3fn)
    weight = weight.cuda()


    x = torch.randn(batch_size, dim)
    x = x.to(torch.bfloat16)
    x = x.cuda()

    x_labels = torch.zeros((num_labels, batch_size), dtype=torch.bfloat16)
    x_labels = x_labels.cuda()
    rows = [0, 1, 0, 3, 2, 3, 2]
    cols = [0, 10, 10, 11, 10, 15, 12]
    x_labels[cols, rows] = 1

    logits = torch.mm(weight, x.t())
    logits.sigmoid_()
    logits -= x_labels

    x_grad = torch.mm(logits.T, weight)

    w_grad = torch.mm(logits, x)

   
    bf16_out = weight.to(torch.float32) - lr * w_grad

    tmp_x_grad = torch.zeros_like(x, dtype=torch.float32)
    matmul_update(weight, x.T.contiguous(), x_labels, tmp_x_grad, lr, 1024)

    print(bf16_out.to(torch.bfloat16) - weight.to(torch.bfloat16))
    print(x_grad.to(torch.bfloat16) - tmp_x_grad.to(torch.bfloat16))
    print(torch.sum(torch.abs(weight.to(torch.bfloat16) - bf16_out.to(torch.bfloat16))))

# test_matmul()

