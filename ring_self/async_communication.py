import threading
import math
import os

import torch
import torch.distributed as dist
from torch.distributed import batch_isend_irecv, P2POp, isend, irecv

# Sequence parallel group that the current rank belongs to.
_SEQUENCE_PARALLEL_GROUP = None

# These values enable us to change the sequence parallel sizes on the fly.
_SEQUENCE_PARALLEL_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

# Global buffer for P2P
_PEER_Q = None
_PEER_K = None
_PEER_V = None
_PEER_M = None
_PEER_L = None
_PEER_O = None
_PEER_BIAS_ARGS = None
_PEER_Q_BWD = None
_PEER_K_BWD = None
_PEER_V_BWD = None
_PEER_O_BWD = None
_PEER_BIAS_ARGS_BWD = None

_DELTA_DQ = None
_PEER_L = None
_DELTA_DK = None
_DELTA_DV = None
_DK_DELTA_FROM_PEER = None
_DV_DELTA_FROM_PEER = None
_PEER_DO = None


def initialize_distributed():
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
    else:
        if int(os.environ["RANK"]) == 0:
            print("Initializing Torch distributed.")
        dist.init_process_group(
            backend="nccl",
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        )
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    destroy_sequence_parallel()
    _initialize_sequence_parallel()
   # create_nccl_communicators()

def _initialize_sequence_parallel(sequence_parallel_size=None):
    # Get world size and rank. Ensure some consistencies.
    assert sequence_parallel_size is None, "Multiple sequence parallel group not implemented."
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if sequence_parallel_size is None:
        sequence_parallel_size = world_size
    else:
        assert world_size % sequence_parallel_size == 0
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size

    rank = torch.distributed.get_rank()

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_RANK
    global _SEQUENCE_PARALLEL_SIZE

    assert (
        _SEQUENCE_PARALLEL_GROUP is None
    ), 'sequence parallel group is already initialized'
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
            _SEQUENCE_PARALLEL_RANK = ranks.index(rank)
            _SEQUENCE_PARALLEL_SIZE = len(ranks)

    if dist.get_rank() == 0:
        print("************ Finish sequence parallel group Initialization. ***********")
    # _set_global_memory_buffer()

def maybe_get_set_global_memory_buffer(q, k, v, m, l, o, bias_args):
    global _PEER_Q, _PEER_K, _PEER_V, _PEER_M, _PEER_L, _PEER_O, _PEER_BIAS_ARGS
    # if _PEER_Q is None:
    if _PEER_Q is None or q.shape != _PEER_Q[0].shape or k.shape != _PEER_K[0].shape:
        _PEER_Q = None
        _PEER_K = [torch.empty_like(k) for _ in range(2)]
        _PEER_V = [torch.empty_like(v) for _ in range(2)]
        _PEER_M = None
        _PEER_L = None
        _PEER_O = None

        # bias_args is a tuple of tensors
        _PEER_BIAS_ARGS = [[torch.empty_like(b) for b in bias_args] for _ in range(2)]
        
    return _PEER_Q, _PEER_K, _PEER_V, _PEER_M, _PEER_L, _PEER_O, _PEER_BIAS_ARGS

def maybe_get_set_global_memory_buffer_bwd(dq, dk, dv, q, L, k, v, o, do, bias_args):
    global _DELTA_DQ, _DELTA_DK, _DELTA_DV, _DK_DELTA_FROM_PEER, _DV_DELTA_FROM_PEER,_PEER_Q_BWD, _PEER_L, _PEER_K_BWD, _PEER_V_BWD, _PEER_O_BWD, _PEER_DO, _PEER_BIAS_ARGS_BWD
    # if _DELTA_DQ is None:
    if _DELTA_DK is None or k.shape != _PEER_K_BWD[0].shape:
        _DELTA_DQ = None 
        _DELTA_DK = [torch.empty_like(dk) for _ in range(2)]
        _DELTA_DV = [torch.empty_like(dv) for _ in range(2)]
        _PEER_L = None
        
        _DK_DELTA_FROM_PEER = None
        _DV_DELTA_FROM_PEER = None

        # may already be initailized in the forward call.
        # current forward and backward needs a transpose in q's format
        _PEER_Q_BWD = None
        _PEER_K_BWD = [torch.empty_like(k) for _ in range(2)]
        _PEER_V_BWD = [torch.empty_like(v) for _ in range(2)]
        _PEER_O_BWD = None
            
        _PEER_DO = None
        _PEER_BIAS_ARGS_BWD = [[torch.empty_like(b) for b in bias_args] for _ in range(2)]

    return _DELTA_DQ, _DELTA_DK, _DELTA_DV, _DK_DELTA_FROM_PEER, _DV_DELTA_FROM_PEER,  _PEER_Q_BWD, _PEER_L, _PEER_K_BWD, _PEER_V_BWD, _PEER_O_BWD, _PEER_DO, _PEER_BIAS_ARGS_BWD

def reset_global_memory_buffer():
    global _PEER_Q, _PEER_K, _PEER_V, _PEER_M, _PEER_L, _PEER_O, _DELTA_DQ, _PEER_L, _DELTA_DK, _DELTA_DV, _DK_DELTA_FROM_PEER, _DV_DELTA_FROM_PEER, _PEER_DO, _PEER_BIAS_ARGS, _PEER_BIAS_ARGS_BWD
    global _PEER_K_BWD, _PEER_V_BWD
    _PEER_Q = None
    _PEER_K = None
    _PEER_V = None
    _PEER_M = None
    _PEER_L = None
    _PEER_O = None
    _PEER_BIAS_ARGS = None

    _DELTA_DQ = None
    _PEER_L = None
    _DELTA_DK = None
    _DELTA_DV = None
    _DK_DELTA_FROM_PEER = None
    _DV_DELTA_FROM_PEER = None
    _PEER_DO = None
    _PEER_BIAS_ARGS_BWD = None

    _PEER_K_BWD = None
    _PEER_V_BWD = None

# Pytorch defers the creation of nccl communicators to the first P2P call,
# We manually create them so the first isend does not hang without an irecv.
# reference: https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/nccl.cpp#L138
# Only support even number of GPUs.
def create_nccl_communicators():
    seq_rank = get_sequence_parallel_rank()
    seq_group = get_sequence_parallel_group()

    empty_tensor = torch.empty(1,).cuda()
    empty_tensor_2 = torch.empty(1,).cuda()
    if torch.distributed.get_rank() % 2 == 0:
        # sender
        op1 = P2POp(op=isend, tensor=torch.empty(1,).cuda(), peer=seq_rank+1, group=seq_group)
        op2 = P2POp(op=irecv, tensor=torch.empty(1,).cuda(), peer=seq_rank+1, group=seq_group)
        #req = torch.distributed.isend(tensor=empty_tensor, dst=seq_rank + 1, group=seq_group)
        dist.batch_isend_irecv([op1, op2])
    else:
        # receiver
        op1 = P2POp(op=irecv, tensor=torch.empty(1,).cuda(), peer=seq_rank-1, group=seq_group)
        op2 = P2POp(op=isend, tensor=torch.empty(1,).cuda(), peer=seq_rank-1, group=seq_group)
        #req = torch.distributed.isend(tensor=empty_tensor, dst=seq_rank + 1, group=seq_group)
        handles = dist.batch_isend_irecv([op1, op2])
        #req = torch.distributed.irecv(tensor=empty_tensor, src=seq_rank - 1, group=seq_group)
    dist.all_reduce(empty_tensor, group=seq_group)

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    #global _SEQUENCE_PARALLEL_GROUP
    assert (
        _SEQUENCE_PARALLEL_GROUP is not None
    ), 'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_rank():
    """Return my rank for the sequence  parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_parallel_group())

def get_sequence_parallel_size():
    """Return my rank for the sequence  parallel group."""
    global _SEQUENCE_PARALLEL_SIZE
    if _SEQUENCE_PARALLEL_SIZE is not None:
        return _SEQUENCE_PARALLEL_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())

def destroy_sequence_parallel():
    """Set the groups to none."""
    global _SEQUENCE_PARALLEL_GROUP
    _SEQUENCE_PARALLEL_GROUP = None

def maybe_send_recv_fwd_qkvo(q_send: torch.Tensor, q_recv: torch.Tensor,
                             k_send: torch.Tensor, k_recv: torch.Tensor,
                             v_send: torch.Tensor, v_recv: torch.Tensor,
                             bias_args_send: list, bias_args_recv: list,
                             comm_mode) -> torch.Tensor:

    seq_group = get_sequence_parallel_group()
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()

    all_handles = []
    send_rank = (seq_rank + 1) % seq_world_size
    recv_rank = (seq_rank - 1) % seq_world_size
    

    all_handles.append(P2POp(op=isend, tensor=k_send, peer=send_rank, group=seq_group))
    all_handles.append(P2POp(op=isend, tensor=v_send, peer=send_rank, group=seq_group))
    for i in range(len(bias_args_send)):
        all_handles.append(P2POp(op=isend, tensor=bias_args_send[i], peer=send_rank, group=seq_group))
    
    all_handles.append(P2POp(op=irecv, tensor=k_recv, peer=recv_rank, group=seq_group))
    all_handles.append(P2POp(op=irecv, tensor=v_recv, peer=recv_rank, group=seq_group))
    for i in range(len(bias_args_recv)):
        all_handles.append(P2POp(op=irecv, tensor=bias_args_recv[i], peer=recv_rank, group=seq_group))

    #return reqs
    all_reqs = launch_async_handles(all_handles, comm_mode)
    return [all_reqs]

# delta: may be you are using it for your local compute or as a distributed buffer to send to others
# .. Sorry for the bad naming..
def maybe_send_recv_bwd_kv(k_send: torch.Tensor, k_recv: torch.Tensor, 
                           v_send: torch.Tensor, v_recv: torch.Tensor, 
                           bias_args_send: list, bias_args_recv: list,
                           comm_mode):
     
    seq_group = get_sequence_parallel_group()
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()

    all_handles = []
    send_rank = (seq_rank + 1) % seq_world_size
    recv_rank = (seq_rank - 1) % seq_world_size
    
    # send kv
    all_handles.append(P2POp(op=isend, tensor=k_send, peer=send_rank, group=seq_group))
    all_handles.append(P2POp(op=isend, tensor=v_send, peer=send_rank, group=seq_group))
    for i in range(len(bias_args_send)):
        all_handles.append(P2POp(op=isend, tensor=bias_args_send[i], peer=send_rank, group=seq_group))

    all_handles.append(P2POp(op=irecv, tensor=k_recv, peer=recv_rank, group=seq_group))
    all_handles.append(P2POp(op=irecv, tensor=v_recv, peer=recv_rank, group=seq_group))
    for i in range(len(bias_args_recv)):
        all_handles.append(P2POp(op=irecv, tensor=bias_args_recv[i], peer=recv_rank, group=seq_group))

    all_reqs = launch_async_handles(all_handles, comm_mode)
    return [all_reqs]

def maybe_send_recv_bwd_delta(dk_delta_send: torch.Tensor, dk_delta_recv: torch.Tensor,
                              dv_delta_send: torch.Tensor, dv_delta_recv: torch.Tensor,
                              time_step, comm_mode, debug=False):
    seq_group = get_sequence_parallel_group()
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()
    
    all_handles = []

    send_rank = (seq_rank + 1) % seq_world_size
    recv_rank = (seq_rank - 1) % seq_world_size

    all_handles.append(P2POp(op=isend, tensor=dk_delta_send, peer=send_rank, group=seq_group))
    all_handles.append(P2POp(op=isend, tensor=dv_delta_send, peer=send_rank, group=seq_group))
    if debug:
        _bwd_send_volume += torch.numel(dk_delta_send) * dk_delta_send.element_size()
        _bwd_send_volume += torch.numel(dv_delta_send) * dv_delta_send.element_size()

    all_handles.append(P2POp(op=irecv, tensor=dk_delta_recv, peer=recv_rank, group=seq_group))
    all_handles.append(P2POp(op=irecv, tensor=dv_delta_recv, peer=recv_rank, group=seq_group))
    if debug:
        _bwd_recv_volume += torch.numel(dk_delta_recv) * dk_delta_recv.element_size()
        _bwd_recv_volume += torch.numel(dv_delta_recv) * dv_delta_recv.element_size()
    
    all_reqs = launch_async_handles(all_handles, comm_mode)
     
    return [all_reqs]

def launch_async_handles(handles, comm_mode):
    global _args
    if comm_mode == "nocomm":
        #print("skipping communication for ablation")
        return []
    if len(handles) > 0:
        return dist.batch_isend_irecv(handles)
    return []

def wait_async_handles(reqs):
    if len(reqs) > 0:
        for req in reqs:
            for r in req:
                r.wait()