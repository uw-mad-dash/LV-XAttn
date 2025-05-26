NUM_NODES=2
NUM_PROC_PER_NODE=3
HOST="c301-003"
RANK=${PMI_RANK}

torchrun --nproc_per_node=${NUM_PROC_PER_NODE} \
        --nnodes=${NUM_NODES} \
        --master_addr=${HOST} \
        --node_rank=${RANK} \
        --master_port=12345 \
        ../test_llama3.py