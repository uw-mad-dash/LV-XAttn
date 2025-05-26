NUM_NODES=1
NUM_PROC_PER_NODE=3
HOST="c301-002"
RANK=${PMI_RANK}

torchrun --nproc_per_node=${NUM_PROC_PER_NODE} \
        --nnodes=${NUM_NODES} \
        --nproc_per_node ${NUM_PROC_PER_NODE} \
        --master_addr=${HOST} \
        --node_rank=${RANK} \
        --master_port=12344 \
        ../test.py


