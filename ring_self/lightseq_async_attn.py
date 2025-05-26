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
    from .flash_attn_kernels import _flash_attn_forward, _flash_attn_backward
except:
    pass

from .async_communication import (is_last_time, is_compute_for_local_query, is_sync_from_remote, is_idle, print_and_reset_comm_stats, 
        launch_async_handles, wait_async_handles, maybe_send_recv_fwd_qkvo, maybe_send_recv_bwd_kv, maybe_send_recv_bwd_delta, reset_global_memory_buffer,
        maybe_get_set_global_memory_buffer, maybe_get_set_global_memory_buffer_bwd, initialize_distributed, get_sequence_parallel_size, get_sequence_parallel_rank)

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
    tl.store(m_ptrs, m_i)
    tl.store(l_ptrs, l_i)
    # write back O, L
    if LAST_STEP: # should not happen
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


def _lightseq_forward(q, k, v, causal, sm_scale, comm_mode, bias_func=None, 
                    local_bias_args=[], remote_bias_args=[], no_overlap=False):
    # bias_args is a tuple or list of arguments to the bias_func

    # maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    # shape constraints
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    Lq, Lk, Lv = q.shape[-2], k.shape[-2], v.shape[-2]
    # Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv
    assert Lk == Lv
    # assert Lk in {16, 32, 64, 128}
    if bias_func is None:
        assert len(local_bias_args) == 0 and len(remote_bias_args) == 0

    BLOCK_M = 32
    BLOCK_N = 32

   
    bsz, nh, seq_len, hdim = q.shape

    m = torch.full((bsz * nh, seq_len), fill_value=-float("inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros_like(m)
    L = torch.empty_like(m) # to be overwritten
    o = torch.zeros_like(q)
    
    grid = (triton.cdiv(seq_len, BLOCK_M), bsz * nh, 1)
    num_warps = 4 if Lk <= 64 else 8
    # num_warps = 4
    # print(f"num_warps: {num_warps}")
    
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()

    # Initialize all buffers
    peer_q, peer_k, peer_v, peer_m, peer_l, peer_o, peer_bias_args = maybe_get_set_global_memory_buffer(q, k, v, m, l, o, remote_bias_args)
    
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
    
    peer_k[1] = k.clone().detach()
    peer_v[1] = v.clone().detach()
    for i in range(len(remote_bias_args)):
        peer_bias_args[1][i] = remote_bias_args[i].clone().detach()
    torch.cuda.synchronize()
    for time_step in range(seq_world_size):
        # This is important for cuda scheduler to execute nccl calls first.
        # torch.cuda.synchronize()
        # Communication uses recv_buffer_idx, and compute uses send_buffer_idx, which effectively are contents from the last time step.
        send_buffer_idx = (time_step + 1) % 2
        recv_buffer_idx = time_step % 2

        if time_step == seq_world_size-1:
            reqs = []
        else:
            reqs = maybe_send_recv_fwd_qkvo(
                q_send=None, q_recv=None, 
                k_send=peer_k[send_buffer_idx], k_recv=peer_k[recv_buffer_idx], 
                v_send=peer_v[send_buffer_idx], v_recv=peer_v[recv_buffer_idx], 
                bias_args_send=peer_bias_args[send_buffer_idx], 
                bias_args_recv=peer_bias_args[recv_buffer_idx],
                comm_mode=comm_mode)
            if no_overlap:
                wait_async_handles(reqs)
        
        bias = None
        if bias_func is not None:
            bias = bias_func(*(local_bias_args + peer_bias_args[send_buffer_idx])) # [B, 1, L_Q, L_KV]
            bias = bias.expand(bsz, nh, seq_len, Lk) # [B, H, L_Q, L_KV]

        fwd_launch_helper(
            q=q, 
            k=maybe_repeat_kv_fwd(q.shape[1], peer_k[send_buffer_idx]), 
            v=maybe_repeat_kv_fwd(q.shape[1], peer_v[send_buffer_idx]), 
            m=m, 
            l=l, 
            o=o, 
            L=L, 
            bias=bias,
            IS_CAUSAL=causal,
            LAST_STEP=time_step == seq_world_size - 1)

        wait_async_handles(reqs)
        # sync between statistics get from other ranks and the local ones
    torch.cuda.synchronize()
    return q, k, v, o, L

def _lightseq_backward(do, q, k, v, o, L, causal, sm_scale, comm_mode, backward_engine, bias_func=None, local_bias_args=[], remote_bias_args=[], no_overlap=False):
    bsz, nh, seq_len, hdim = q.shape
    Lq, Lk, Lv = q.shape[-2], k.shape[-2], v.shape[-2]

    q, k, v, o, do = [rearrange(_x, 'b h s d -> b s h d').contiguous() for _x in [q, k, v, o, do]]
    # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}, o shape: {o.shape}, do shape: {do.shape}")
    L = rearrange(L, '(b h) s -> b h s', b=q.shape[0])
    
    dq = torch.zeros_like(q)
    dk = torch.empty_like(k) # to be overwritten
    dv = torch.empty_like(v) # to be overwritten

    # pad L to have length (s) divisible by 128
    BLOCK = 128
    L = torch.cat([L, torch.zeros(bsz, nh, (BLOCK - L.shape[-1] % BLOCK) % BLOCK, device=L.device, dtype=L.dtype)], dim=-1)

    # maybe gqa
    nqh = q.shape[2]
    nkvh = k.shape[2]
    is_gqa = (nqh > nkvh)

    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
   
    # Initialize all backward buffers
    _dq_delta, dk_delta, dv_delta, _dk_delta_from_peer, _dv_delta_from_peer, \
            _peer_q, _peer_L, peer_k, peer_v, _peer_o, _peer_do, peer_bias_args = \
                maybe_get_set_global_memory_buffer_bwd(dq, dk, dv, q, L, k, v, o, do, remote_bias_args)
    
    dk_delta[0] = torch.zeros_like(k)
    dv_delta[0] = torch.zeros_like(v)
    peer_k[0] = k.clone().detach()
    peer_v[0] = v.clone().detach()
    for i in range(len(remote_bias_args)):
        peer_bias_args[0][i] = remote_bias_args[i].clone().detach()

    comp_reqs = []
    rescale_reqs = []
    torch.cuda.synchronize()
    for time_step in range(seq_world_size):
        # torch.cuda.synchronize()
        send_buffer_idx = time_step % 2
        recv_buffer_idx = (time_step+1) % 2
        
        wait_async_handles(comp_reqs)
        if time_step != seq_world_size - 1:
            comp_reqs = maybe_send_recv_bwd_kv(
                k_send=peer_k[send_buffer_idx],  # send
                k_recv=peer_k[recv_buffer_idx], # recv
                v_send=peer_v[send_buffer_idx],  # send
                v_recv=peer_v[recv_buffer_idx], # recv
                bias_args_send=peer_bias_args[send_buffer_idx],
                bias_args_recv=peer_bias_args[recv_buffer_idx],
                comm_mode=comm_mode)
            if no_overlap:
                wait_async_handles(comp_reqs)

        dq_delta_cur = torch.empty_like(dq)
        dk_delta_cur = torch.empty_like(dk)
        dv_delta_cur = torch.empty_like(dv)
        # print(f"calc using peer_k[send_buffer_idx]: {peer_k[send_buffer_idx]}")
        # _flash_attn_backward(
        #     do, 
        #     q, 
        #     peer_k[send_buffer_idx], 
        #     peer_v[send_buffer_idx], 
        #     o, 
        #     L, 
        #     dq_delta_cur,
        #     dk_delta_cur,
        #     dv_delta_cur,
        #     0.0, 
        #     sm_scale, 
        #     False, 
        #     (-1, -1), 
        #     None, 
        #     False, 
        #     None)
        bias = None
        if bias_func is not None:
            bias = bias_func(*(local_bias_args + peer_bias_args[send_buffer_idx])) # [B, 1, L_Q, L_KV]
            bias = bias.expand(bsz, nh, Lq, Lk) # [B, H, L_Q, L_KV]

        _flash_attn_backward(
            do=do,
            q=q,
            k=peer_k[send_buffer_idx],
            v=peer_v[send_buffer_idx],
            o=o,
            lse=L,
            dq=dq_delta_cur,
            dk=dk_delta_cur,
            dv=dv_delta_cur,
            softmax_scale=sm_scale,
            causal=causal,
            bias=bias,
        )
        dq += dq_delta_cur
        
        wait_async_handles(rescale_reqs)
        dk_delta[send_buffer_idx] += dk_delta_cur
        dv_delta[send_buffer_idx] += dv_delta_cur
        # print(f"updating dk_delta[send_buffer_idx]: {dk_delta[send_buffer_idx]}")

        rescale_reqs = maybe_send_recv_bwd_delta(
            dk_delta_send=dk_delta[send_buffer_idx],
            dk_delta_recv=dk_delta[recv_buffer_idx],
            dv_delta_send=dv_delta[send_buffer_idx],
            dv_delta_recv=dv_delta[recv_buffer_idx],
            time_step=time_step, comm_mode=comm_mode)
        if no_overlap:
            wait_async_handles(rescale_reqs)
        
        if time_step == seq_world_size - 1:
            wait_async_handles(rescale_reqs)
            dk = dk_delta[recv_buffer_idx]
            dv = dv_delta[recv_buffer_idx]
        
                
    dq, dk, dv = [rearrange(_x, 'b h s d -> b s h d') for _x in [dq, dk, dv]]
    torch.cuda.synchronize()
    return dq, dk, dv
