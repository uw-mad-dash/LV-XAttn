NUM_NODES=1
NUM_PROC_PER_NODE=4
RDZV_HOST="nid001052"
RDZV_PORT=29500

torchrun --nproc_per_node=${NUM_PROC_PER_NODE} \
        --nnodes=${NUM_NODES} \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
        ../test_llama3.py