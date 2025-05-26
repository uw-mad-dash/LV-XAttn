import os
import math

from einops import rearrange
import argparse

import pytest
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
#from torch.profiler import profile, record_function, ProfilerActivity

import triton
import triton.language as tl
import time
import numpy as np
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

try:
    # from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    # from flash_attn.flash_attn_triton import _flash_attn_backward
    from .flash_attn_kernels import _flash_attn_backward
except:
    pass

from .async_communication import (
        wait_async_handles,  reset_global_memory_buffer,
        maybe_get_set_global_memory_buffer, maybe_get_set_global_memory_buffer_bwd, initialize_distributed, get_sequence_parallel_size, get_sequence_parallel_rank,
        send_recv_fwd_q, send_recv_fwd_Lo, send_recv_bwd_comp_reqs, send_recv_bwd_rescale_reqs)

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

@triton.jit
def _rescale_kernel(
    # peer_m,
    # m,
    # peer_l,
    # l,
    peer_o,
    o,
    peer_L,
    L,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LAST_STEP: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    o_offset = off_hz * stride_oh
    peer_o_block_ptr = tl.make_block_ptr(
        base=peer_o + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    o_block_ptr = tl.make_block_ptr(
        base=o + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # peer_m_ptrs = peer_m + off_hz * N_CTX + offs_m
    # m_ptrs = m + off_hz * N_CTX + offs_m
    # peer_l_ptrs = peer_l + off_hz * N_CTX + offs_m
    # l_ptrs = l + off_hz * N_CTX + offs_m
    
    # # peer_m_i = tl.load(peer_m_ptrs) 
    # # peer_m_i = peer_m_i.to(tl.float32)
    # m_i = tl.load(m_ptrs) 
    # m_i = m_i.to(tl.float32)
    # # peer_l_i = tl.load(peer_l_ptrs) 
    # # peer_l_i = peer_l_i.to(tl.float32)
    # l_i = tl.load(l_ptrs) 
    # l_i = l_i.to(tl.float32)
    
    peer_L_ptrs = peer_L + off_hz * N_CTX + offs_m
    L_ptrs = L + off_hz * N_CTX + offs_m
    
    old_L_i = tl.load(peer_L_ptrs)
    old_L_i = old_L_i.to(tl.float32)
    cur_L_i = tl.load(L_ptrs)
    cur_L_i = cur_L_i.to(tl.float32)
    
    # Update L_i to log(exp(old_L_i) + exp(cur_L_i))
    new_L_i = tl.math.log(tl.math.exp(old_L_i) + tl.math.exp(cur_L_i))
    
    old_acc = tl.load(peer_o_block_ptr)
    old_acc = old_acc.to(tl.float32)
    cur_acc = tl.load(o_block_ptr)
    cur_acc = cur_acc.to(tl.float32)
    
    # Update acc to acc * exp(old_L_i) / exp(new_L_i) + peer_acc * exp(cur_L_i) / exp(new_L_i)
    new_acc = old_acc * tl.math.exp(old_L_i[:, None]) / tl.math.exp(new_L_i[:, None]) + cur_acc * tl.math.exp(cur_L_i[:, None]) / tl.math.exp(new_L_i[:, None])
    
    tl.store(L_ptrs, new_L_i)
    tl.store(o_block_ptr, new_acc.to(tl.float16))
    
    

    # peer_acc = tl.load(peer_o_block_ptr)
    # peer_acc = peer_acc.to(tl.float32)
    # acc = tl.load(o_block_ptr) 
    # acc = acc.to(tl.float32)
    # lo = 0
    # hi = N_CTX
    # m_i_sync = tl.maximum(m_i, peer_m_i)
    # alpha = tl.math.exp2(m_i - m_i_sync)
    # peer_alpha = tl.math.exp2(peer_m_i - m_i_sync)
    # # -- scale and update acc --
    # acc_scale = l_i * 0 + alpha  # workaround some compiler bug
    # peer_acc_scale = peer_l_i * 0 + peer_alpha  # workaround some compiler bug
    
    # acc *= acc_scale[:, None]
    # peer_acc *= peer_acc_scale[:, None]
    # acc += peer_acc
    # l_i = l_i * acc_scale + peer_l_i * peer_acc_scale
    # # write back O, l, m
    # tl.store(m_ptrs, m_i_sync)
    # tl.store(l_ptrs, l_i)
    # if LAST_STEP: # should not happen
    #     acc = acc / l_i[:, None]
    #     L_ptrs = L + off_hz * N_CTX + offs_m
    #     tl.store(L_ptrs, m_i_sync / 1.44269504 + tl.math.log(l_i))
    # tl.store(o_block_ptr, acc.to(tl.float16))

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    m,
    l,
    O,
    L,
    Bias, # shape (Z, H, L_Q, L_KV)
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_bz, stride_bh, stride_bm, stride_bn,
    Z, H, L_Q, L_KV,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    LAST_STEP: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_vh
    bias_offset = off_hz * stride_bh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(L_Q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, L_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(L_KV, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + q_offset,
        shape=(L_Q, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    if ADD_BIAS:
        Bias_block_ptr = tl.make_block_ptr(
            base=Bias + bias_offset,
            shape=(L_Q, L_KV),
            strides=(stride_bm, stride_bn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l -> load from provided pointer
    m_ptrs = m + off_hz * L_Q + offs_m
    l_ptrs = l + off_hz * L_Q + offs_m
    m_i = tl.load(m_ptrs) 
    m_i = m_i.to(tl.float32)
    l_i = tl.load(l_ptrs) 
    l_i = l_i.to(tl.float32)
    acc = tl.load(O_block_ptr) 
    acc = acc.to(tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # qk_scale = sm_scale * 1.44269504
    qk_scale = sm_scale
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else L_KV
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)
        if ADD_BIAS:
            bias = tl.load(Bias_block_ptr)
            qk += bias
        qk *= 1.44269504
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if ADD_BIAS:
            Bias_block_ptr = tl.advance(Bias_block_ptr, (0, BLOCK_N))
    # write back original l and m
    # tl.store(m_ptrs, m_i)
    # tl.store(l_ptrs, l_i)
    # write back O, L
    # if LAST_STEP: # should not happen
    if True:
        acc = acc / l_i[:, None]
        L_ptrs = L + off_hz * L_Q + offs_m
        tl.store(L_ptrs, m_i / 1.44269504 + tl.math.log(l_i))
    tl.store(O_block_ptr, acc.to(tl.float16))

# for gqa/mqa to expand kv heads
def maybe_repeat_kv_fwd(nqh, kv):
    bs, nkvh, slen, hdim = kv.shape
    n_rep = nqh // nkvh
    if n_rep == 1:
        return kv
    kv_expand = kv[:, :, None, :, :].expand(bs, nkvh, n_rep, slen, hdim)
    return kv_expand.reshape(bs, nkvh * n_rep, slen, hdim)

def maybe_repeat_kv_bwd(nqh, kv):
    bs, slen, nkvh, hdim = kv.shape
    n_rep = nqh // nkvh
    if n_rep == 1:
        return kv
    kv_expand = kv[:, :, :, None, :].expand(bs, slen, nkvh, n_rep, hdim)
    return kv_expand.reshape(bs, slen, nkvh * n_rep, hdim)

# kv grad has shape bs, slen, nqh, hdim
def maybe_reduce_dkv(nkvh, dkv):
    bs, slen, nqh, hdim = dkv.shape
    n_rep = nqh // nkvh
    if n_rep == 1:
        return dkv
    dkv_reshape = dkv.view(bs, slen, nkvh, n_rep, hdim)
    return torch.sum(dkv_reshape, dim=3)


def _attn_forward(q, k, v, causal, sm_scale, comm_mode, bias_func=None, 
                    local_bias_args=[], remote_bias_args=[], no_overlap=False):
    # bias_args is a tuple or list of arguments to the bias_func

    # shape constraints
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    Lq, Lk, Lv = q.shape[-2], k.shape[-2], v.shape[-2]
    # Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv
    assert Lk == Lv

    if bias_func is None:
        assert len(local_bias_args) == 0 and len(remote_bias_args) == 0

    # assert Lk in {16, 32, 64, 128}
    BLOCK_M = 32
    BLOCK_N = 32
   
    bsz, nh, seq_len, hdim = q.shape

    m = torch.full((bsz * nh, seq_len), fill_value=-float("inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros_like(m)
    L = torch.zeros_like(m) # to be overwritten
    o = torch.zeros_like(q)
    
    grid = (triton.cdiv(seq_len, BLOCK_M), bsz * nh, 1)
    num_warps = 4 if Lk <= 64 else 8
    # num_warps = 4
    # print(f"num_warps: {num_warps}")
    
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()

    # Initialize all buffers
    peer_q, peer_k, peer_v, _peer_m, peer_L, peer_o, peer_bias_args = maybe_get_set_global_memory_buffer(q, k, v, m, L, o, remote_bias_args)
    
    fwd_launch_helper = lambda q, k, v, m, l, o, L, bias, IS_CAUSAL, LAST_STEP: _fwd_kernel[grid](
                q, k, v, sm_scale,
                m,
                l,
                o,
                L,
                bias,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                bias.stride(0) if bias is not None else 0, 
                bias.stride(1) if bias is not None else 0,
                bias.stride(2) if bias is not None else 0, 
                bias.stride(3) if bias is not None else 0,
                q.shape[0], q.shape[1], q.shape[2], k.shape[2],
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Dk,
                IS_CAUSAL=IS_CAUSAL,
                LAST_STEP=LAST_STEP,
                ADD_BIAS=bias is not None,
                num_warps=num_warps,
                num_stages=4)
    
    peer_q[0] = q.clone().detach() # need to clone to avoid overwrite q
    peer_L[0] = L
    peer_o[0] = o
    for i in range(len(remote_bias_args)):
        peer_bias_args[0][i] = remote_bias_args[i].clone().detach()
    q_reqs = []
    mlo_reqs = []
    torch.cuda.synchronize()
    for time_step in range(seq_world_size):
        send_buffer_idx = time_step % 2
        recv_buffer_idx = (time_step - 1) % 2

        wait_async_handles(q_reqs)
        if time_step < seq_world_size - 1:
            q_reqs = send_recv_fwd_q(
                q_send=peer_q[send_buffer_idx], 
                q_recv=peer_q[recv_buffer_idx], 
                bias_args_send=peer_bias_args[send_buffer_idx],
                bias_args_recv=peer_bias_args[recv_buffer_idx],
                comm_mode=comm_mode)
            if no_overlap:
                wait_async_handles(q_reqs)
        else:
            q_reqs = []

        # computation
        m_delta = torch.full_like(m, fill_value=-float("inf"))
        l_delta = torch.zeros_like(l)
        o_delta = torch.zeros_like(o)
        L_delta = torch.zeros_like(L)
        bias = None
        if bias_func is not None:
            bias = bias_func(*(local_bias_args + peer_bias_args[send_buffer_idx])) # [B, 1, L_Q, L_KV]
            bias = bias.expand(bsz, nh, Lq, Lk) # [B, H, L_Q, L_KV]
        fwd_launch_helper(
            q=peer_q[send_buffer_idx],
            k=maybe_repeat_kv_fwd(peer_q[send_buffer_idx].shape[1], k),
            v=maybe_repeat_kv_fwd(peer_q[send_buffer_idx].shape[1], v),
            m=m_delta,
            l=l_delta,
            o=o_delta,
            L=L_delta, # not used, as not last step
            bias=bias,
            IS_CAUSAL=causal,
            LAST_STEP=False)
    
        # ensure mlo is ready
        wait_async_handles(mlo_reqs)
        # rescale
        _rescale_kernel[grid](
            peer_o=o_delta,
            o=peer_o[send_buffer_idx],
            peer_L=L_delta,
            L=peer_L[send_buffer_idx],
            stride_oz=peer_o[send_buffer_idx].stride(0),
            stride_oh=peer_o[send_buffer_idx].stride(1),
            stride_om=peer_o[send_buffer_idx].stride(2),
            stride_on=peer_o[send_buffer_idx].stride(3),
            Z=peer_o[send_buffer_idx].shape[0],
            H=peer_o[send_buffer_idx].shape[1],
            N_CTX=peer_o[send_buffer_idx].shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Dk,
            LAST_STEP=False
        )
        # send mlo
        if seq_world_size > 1:
            mlo_reqs = send_recv_fwd_Lo(
                L_send=peer_L[send_buffer_idx],
                o_send=peer_o[send_buffer_idx],
                L_recv=peer_L[recv_buffer_idx],
                o_recv=peer_o[recv_buffer_idx],
                comm_mode=comm_mode)
            if no_overlap:
                wait_async_handles(mlo_reqs)
    
    # get back the last mlo
    wait_async_handles(mlo_reqs)
    torch.cuda.synchronize()

    return q, k, v, o, L

def _attn_backward(do, q, k, v, o, L, causal, sm_scale, comm_mode, backward_engine, bias_func=None, local_bias_args=[], remote_bias_args=[], no_overlap=False):
    bsz, nh, seq_len, hdim = q.shape
    Lq, Lk, Lv = q.shape[-2], k.shape[-2], v.shape[-2]

    q, k, v, o, do = [rearrange(_x, 'b h s d -> b s h d').contiguous() for _x in [q, k, v, o, do]]
    L = rearrange(L, '(b h) s -> b h s', b=q.shape[0])

    # pad L to have length (s) divisible by 128
    BLOCK = 128
    L = torch.cat([L, torch.zeros(bsz, nh, (BLOCK - L.shape[-1] % BLOCK) % BLOCK, device=L.device, dtype=L.dtype)], dim=-1)
    
    dq = torch.empty_like(q) # to be overwritten
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # maybe gqa
    nqh = q.shape[2]
    nkvh = k.shape[2]
    is_gqa = (nqh > nkvh)

    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
   
    # Initialize all backward buffers
    peer_dq_delta, _dk_delta, _dv_delta, _dk_delta_from_peer, _dv_delta_from_peer, \
            peer_q, peer_L, _peer_k, _peer_v, peer_o, peer_do, \
                 peer_bias_args = maybe_get_set_global_memory_buffer_bwd(dq, dk, dv, q, L, k, v, o, do, remote_bias_args)
    
    peer_q[0] = q.clone().detach()
    peer_L[0] = L.clone().detach()
    peer_o[0] = o.clone().detach()
    peer_do[0] = do.clone().detach()
    peer_dq_delta[0] = torch.zeros_like(dq)
    for i in range(len(remote_bias_args)):
        peer_bias_args[0][i] = remote_bias_args[i].clone().detach()

    comp_reqs = [] # d, L, o, dO
    rescale_reqs = [] # dq_delta
    torch.cuda.synchronize()
    for time_step in range(seq_world_size):
        send_buffer_idx = time_step % 2
        recv_buffer_idx = (time_step + 1) % 2

        wait_async_handles(comp_reqs)
        if time_step < seq_world_size - 1:
            comp_reqs = send_recv_bwd_comp_reqs(
                q_send=peer_q[send_buffer_idx],
                L_send=peer_L[send_buffer_idx],
                o_send=peer_o[send_buffer_idx],
                do_send=peer_do[send_buffer_idx],
                bias_args_send=peer_bias_args[send_buffer_idx],
                q_recv=peer_q[recv_buffer_idx],
                L_recv=peer_L[recv_buffer_idx],
                o_recv=peer_o[recv_buffer_idx],
                do_recv=peer_do[recv_buffer_idx],
                bias_args_recv=peer_bias_args[recv_buffer_idx],
                comm_mode=comm_mode
            )
            if no_overlap:
                wait_async_handles(comp_reqs)
        else:
            comp_reqs = []
        
        # computation
        dq_delta = torch.empty_like(dq)
        dk_delta = torch.empty_like(dk)
        dv_delta = torch.empty_like(dv)
        bias=None
        if bias_func is not None:
            bias = bias_func(*(local_bias_args + peer_bias_args[send_buffer_idx])) # [B, 1, L_Q, L_KV]
            bias = bias.expand(bsz, nh, Lq, Lk) # [B, H, L_Q, L_KV]
            
        _flash_attn_backward(
            do=peer_do[send_buffer_idx],
            q=peer_q[send_buffer_idx],
            k=k,
            v=v,
            o=peer_o[send_buffer_idx],
            lse=peer_L[send_buffer_idx],
            dq=dq_delta,
            dk=dk_delta,
            dv=dv_delta,
            softmax_scale=sm_scale,
            causal=causal,
            bias=bias
        )
        dk += dk_delta
        dv += dv_delta

        wait_async_handles(rescale_reqs)
        peer_dq_delta[send_buffer_idx] += dq_delta

        # send dq_delta
        if seq_world_size > 1:
            rescale_reqs = send_recv_bwd_rescale_reqs(
                dq_delta_send=peer_dq_delta[send_buffer_idx],
                dq_delta_recv=peer_dq_delta[recv_buffer_idx],
                comm_mode=comm_mode)
            if no_overlap:
                wait_async_handles(rescale_reqs)
        
    # get back the last dq_delta
    wait_async_handles(rescale_reqs)
    recv_buffer_idx = seq_world_size % 2 if seq_world_size > 1 else 0
    dq = peer_dq_delta[recv_buffer_idx]
                
    dq, dk, dv = [rearrange(_x, 'b h s d -> b s h d') for _x in [dq, dk, dv]]
    torch.cuda.synchronize()
    return dq, dk, dv