import time
import argparse
import pytest
import torch
import torch.distributed as dist

import contextlib
import json
import csv
import os

from torch.profiler import profile, record_function, ProfilerActivity
from lv_xattn import _lightseq_forward as _lv_xattn_forward, \
                            _lightseq_backward as _lv_xattn_backward, \
                            initialize_distributed as _lv_xattn_initialize_distributed, \
                            reset_global_memory_buffer as _lv_xattn_reset_global_memory_buffer

from ring import _lightseq_forward as _ring_forward, \
                            _lightseq_backward as _ring_backward, \
                            initialize_distributed as _ring_initialize_distributed, \
                            reset_global_memory_buffer as _ring_reset_global_memory_buffer


def initialize_distributed(attention_mode):
    assert attention_mode in ['lv_xattn', 'ring']
    if attention_mode == 'lv_xattn':
        _lv_xattn_initialize_distributed()
    elif attention_mode == 'ring':
        _ring_initialize_distributed()

def reset_global_memory_buffer(attention_mode):
    assert attention_mode in ['lv_xattn', 'ring']
    if attention_mode == 'lv_xattn':
        _lv_xattn_reset_global_memory_buffer()
    elif attention_mode == 'ring':
        _ring_reset_global_memory_buffer()

def create_attention_class(attention_mode):
    assert attention_mode in ['lv_xattn', 'ring']
    class _attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, causal, sm_scale, bias_func=None, local_bias_args=[], remote_bias_args=[]):
            comm_mode = 'lightseq'
            backward_engine = 'flash'

            if attention_mode == 'lv_xattn':
                forward_func = _lv_xattn_forward
            elif attention_mode == 'ring':
                forward_func = _ring_forward

            q, k, v, o, L = forward_func(q, k, v, causal, sm_scale, comm_mode, bias_func=bias_func, local_bias_args=local_bias_args, remote_bias_args=remote_bias_args)

            ctx.save_for_backward(q, k, v, o, L, torch.tensor(len(local_bias_args)), *local_bias_args, *remote_bias_args)
            ctx.sm_scale = sm_scale
            ctx.comm_mode = comm_mode
            ctx.backward_engine = backward_engine
            ctx.causal = causal
            ctx.bias_func = bias_func
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, o, L, local_bias_args_len, *bias_args = ctx.saved_tensors
            local_bias_args = bias_args[:local_bias_args_len]
            remote_bias_args = bias_args[local_bias_args_len:]
            sm_scale = ctx.sm_scale
            bias_func = ctx.bias_func

            if attention_mode == 'lv_xattn':
                backward_func = _lv_xattn_backward
            elif attention_mode == 'ring':
                backward_func = _ring_backward
                
            dq, dk, dv = backward_func(do, q, k, v, o, L, ctx.causal, sm_scale, ctx.comm_mode, ctx.backward_engine, bias_func=bias_func, local_bias_args=local_bias_args, remote_bias_args=remote_bias_args)
            return dq, dk, dv, None, None, None, None, None

    return _attention


def benchmark(attention_mode, Z, H, L_Q, L_KV, D_HEAD, causal, dtype=torch.float16,
              ref_bias=None, bias_func=None, local_bias_args=None, remote_bias_args=None,
              total_iter=20, warmup_iter=10,
              forward_only=True, verbose=False,
              assert_close=True,
              atol=1e-2, rtol=1e-2):
    """
    Z: batch size
    H: num of heads
    D_HEAD: hidden dimension
    """
    assert total_iter > warmup_iter
    assert attention_mode in ['lv_xattn', 'ring']
    if ref_bias is None:
        assert bias_func is None and local_bias_args is None and remote_bias_args is None
    # create attention
    attention = create_attention_class(attention_mode).apply

    forward_time = []
    backward_time = []
    for i in range(total_iter):
        torch.manual_seed(i+1)

        q = torch.empty((Z, H, L_Q, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        k = torch.empty((Z, H, L_KV, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        v = torch.empty((Z, H, L_KV, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
        sm_scale = 0.5
        dout = torch.randn_like(q)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # ---------------------- reference ----------------------
        if assert_close:
            # reference implementation
            M = torch.ones((L_Q, L_KV), device="cuda")
            if causal:
                M = torch.tril(torch.ones((L_Q, L_KV), device="cuda"))
                # if L_Q <= L_KV:
                #     M = torch.tril(torch.ones((L_Q, L_KV), device="cuda"))
                #     # for i in range(L_Q):
                #     #     for j in range(L_KV - L_Q+i+1, L_KV):
                #     #         M[i][j] = 0
                # else:
                #     for i in range(L_Q):
                #         for j in range(L_KV):
                #             if i < L_Q - L_KV + j:
                #                 M[i, j] = 0

            p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
            if ref_bias is not None:
                p = p + ref_bias
            # add bias of all 1
            # bias = torch.ones((L_Q, L_KV), device="cuda")
            # bias = torch.zeros((L_Q, L_KV), device="cuda")
            # add bias of 0 1 0 1 each row
            # bias = torch.zeros((L_Q, L_KV), device="cuda")
            # bias[:, 1::2] = 1
            # p = p + bias

            # row max
            # m = torch.max(p, dim=-1, keepdim=True)
            # # rowsum(e^{p-m})
            # l = torch.exp(p - m.values).sum(dim=-1, keepdim=True)
            # # logsumexp
            # L = m.values + torch.log(l)
            # print(f"L is {L}")
            # assert causal
            p[:, :, M == 0] = float("-inf")
            p = torch.softmax(p.float(), dim=-1)
            p = p.type_as(q)
            full_ref_out = torch.matmul(p, v)
            p = None
            full_ref_out.backward(dout)
            full_ref_dq, q.grad = q.grad.clone(), None
            full_ref_dk, k.grad = k.grad.clone(), None
            full_ref_dv, v.grad = v.grad.clone(), None

            # clean up
            M = None
            p = None

        # ---------------------- distributed ----------------------
        ## prep input and answer ========================
        q_seq_per_rank = L_Q // world_size
        kv_seq_per_rank = L_KV // world_size
        
        if attention_mode in ['lv_xattn', 'ring']:
            if assert_close:
                ref_out = full_ref_out[:,:, rank * q_seq_per_rank: (rank + 1) * q_seq_per_rank, :]
                ref_dq = full_ref_dq[:,:, rank * q_seq_per_rank: (rank + 1) * q_seq_per_rank, :]
            input_q = q[:,:, rank * q_seq_per_rank: (rank + 1) * q_seq_per_rank, :].view(Z, H, -1, D_HEAD).contiguous().clone().detach().requires_grad_(True)
            input_do = dout[:,:, rank * q_seq_per_rank: (rank + 1) * q_seq_per_rank, :].view(Z, H, -1, D_HEAD).contiguous().clone().detach().requires_grad_(True)

        if assert_close:
            ref_dk = full_ref_dk[:,:, rank * kv_seq_per_rank: (rank + 1) * kv_seq_per_rank, :]
            ref_dv = full_ref_dv[:,:, rank * kv_seq_per_rank: (rank + 1) * kv_seq_per_rank, :]
        input_k = k[:,:, rank * kv_seq_per_rank: (rank + 1) * kv_seq_per_rank, :].view(Z, H, -1, D_HEAD).contiguous().clone().detach().requires_grad_(True)
        input_v = v[:,:, rank * kv_seq_per_rank: (rank + 1) * kv_seq_per_rank, :].view(Z, H, -1, D_HEAD).contiguous().clone().detach().requires_grad_(True)

        ## forward ========================
        torch.cuda.synchronize()
        start_time = time.time()
        if ref_bias is not None:
            tri_out = attention(input_q, input_k, input_v, causal, sm_scale, bias_func, local_bias_args, remote_bias_args).half()
        else:
            tri_out = attention(input_q, input_k, input_v, causal, sm_scale).half()
        torch.cuda.synchronize()
        end_time = time.time()
        forward_time.append(end_time - start_time)

        if assert_close:
            assert torch.allclose(tri_out, ref_out, atol=atol, rtol=rtol), f"rank {rank} tri_out: {tri_out}, ref_out: {ref_out}"
            if verbose:
                print(f" *** (iter {i}) rank {rank} passes forward")

        ## backward ========================
        if not forward_only:
            torch.cuda.synchronize()
            start_time = time.time()
            tri_out.backward(input_do)
            torch.cuda.synchronize()
            end_time = time.time()
            backward_time.append(end_time - start_time)

            tri_dq, input_q.grad = input_q.grad.clone(), None
            tri_dk, input_k.grad = input_k.grad.clone(), None
            tri_dv, input_v.grad = input_v.grad.clone(), None

            if assert_close:
                assert torch.allclose(tri_dq, ref_dq, atol=atol, rtol=rtol), f"rank {rank} tri_dq: {tri_dq}, ref_dq: {ref_dq}"
                assert torch.allclose(tri_dk, ref_dk, atol=atol, rtol=rtol), f"rank {rank} tri_dk: {tri_dk}, ref_dk: {ref_dk}"
                assert torch.allclose(tri_dv, ref_dv, atol=atol, rtol=rtol), f"rank {rank} tri_dv: {tri_dv}, ref_dv: {ref_dv}"

                if verbose:
                    print(f" *** (iter {i}) rank {rank} passes backward")
            
            print(f" *** (iter {i}) rank {rank} finished")
        

    forward_time = forward_time[warmup_iter:]
    avg_forward_time = sum(forward_time) / len(forward_time)
    if not forward_only:
        backward_time = backward_time[warmup_iter:]
        avg_backward_time = sum(backward_time) / len(backward_time)
    else:
        avg_backward_time = None

    median_forward_time = sorted(forward_time)[len(forward_time) // 2]
    if not forward_only:
        median_backward_time = sorted(backward_time)[len(backward_time) // 2]

    if verbose:
        print(f" *** {attention_mode} forward time: {avg_forward_time}")
        print(f" *** {attention_mode} backward time: {avg_backward_time}")

    return avg_forward_time, avg_backward_time

if __name__ == "__main__":
    Z = 1
    H = 8
    D_HEAD = 128
    causal=False
    total_iter = 15
    warmup_iter = 5
    forward_only = False
    verbose = True
    is_profile = False
    atol = 1e-2
    rtol = 1e-3
    assert_close = True
    add_bias = False
    trace_dir = "trace/"
    result_csv = "benchmark_results.csv"

    attention_modes = ['ring', 'lv_xattn']
    L_Qs = [1024]
    L_KVs = [2**12]

    results = {}

    for attention_mode in attention_modes:
        initialize_distributed(attention_mode)
        for L_Q in L_Qs:
            for L_KV in L_KVs:
                if dist.get_rank() == 0:
                    print(f" *** attention_mode: {attention_mode}, L_Q: {L_Q}, L_KV: {L_KV}")

                if is_profile:
                    context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
                else:
                    context = contextlib.nullcontext()

                ref_bias = bias_func = local_bias_args = remote_bias_args = None
                if add_bias:
                    # ref_bias = torch.ones((L_Q, L_KV), device="cuda")
                    ref_bias = torch.zeros((L_Q, L_KV), device="cuda")
                    ref_bias[:, 1::2] = 1
                    def bias_func(shape, val):
                        bias = torch.zeros(list(shape), device="cuda")
                        bias[:, :, :, 1::2] = val
                        return bias
                        # return torch.ones(list(shape), device="cuda")
                    if attention_mode in ['lv_xattn', 'ring', 'lv_xattn_gather', 'ring_gather', 'lv_xattn_optimized', 'ring_optimized']:
                        local_bias_args = [torch.tensor([Z, H, L_Q // dist.get_world_size(), L_KV // dist.get_world_size()], device="cuda")]
                        remote_bias_args = [torch.tensor(1, device="cuda")]
                    elif attention_mode in ['dupq']:
                        local_bias_args = [torch.tensor([Z, H, L_Q, L_KV // dist.get_world_size()], device="cuda")]
                        remote_bias_args = [torch.tensor(1, device="cuda")]

                reset_global_memory_buffer(attention_mode)
                with context:
                    forward_time, backward_time = benchmark(
                        attention_mode=attention_mode,
                        Z=Z,
                        H=H,
                        L_Q=L_Q,
                        L_KV=L_KV,
                        D_HEAD=D_HEAD,
                        ref_bias=ref_bias,
                        bias_func=bias_func,
                        local_bias_args=local_bias_args,
                        remote_bias_args=remote_bias_args,
                        causal=causal,
                        dtype=torch.float16,
                        total_iter=total_iter,
                        warmup_iter=warmup_iter,
                        forward_only=forward_only,
                        assert_close=assert_close,
                        atol=atol,
                        rtol=rtol,
                        verbose=verbose
                    )
                    # clear gpu memory
                    # torch.cuda.empty_cache()

                if dist.get_rank() == 0:
                    # print time in ms (times 1000 and round to int)
                    results[(attention_mode, L_Q, L_KV)] = (forward_time*1000, backward_time*1000 if not forward_only else None)
                    print(f"{attention_mode}, {L_Q}, {L_KV}, {results[(attention_mode, L_Q, L_KV)]}")
                    if is_profile:
                        if not os.path.exists(trace_dir):
                            os.makedirs(trace_dir)
                        context.export_chrome_trace(f"{trace_dir}/{attention_mode}_{L_Q}_{L_KV}.json")

    # write results to file
    if dist.get_rank() == 0:
         # write to csv
        with open(result_csv, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["attention_mode", "L_Q", "L_KV", "forward_time", "backward_time"])
            for key, value in results.items():
                writer.writerow([key[0], key[1], key[2], value[0], value[1]])

        print(f"finished writing to benchmark_results.csv")
