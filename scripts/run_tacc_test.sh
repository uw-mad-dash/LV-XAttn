NUM_NODES=1
NUM_PROC_PER_NODE=3
HOST="c301-002"
RANK=${PMI_RANK}
# torchrun --nproc_per_node=1 --nnodes=${NUM_NODES} --master_addr=10.10.1.1 --node_rank=${RANK} --master_port=12344 ${MODE}/lightseq/lightseq_async_attn.py --comm-mode lightseq --debug --bs 1 --n_heads 32 --run-mode test
# torchrun --nproc_per_node=${NUM_PROC_PER_NODE} --nnodes=${NUM_NODES} --nproc_per_node ${NUM_PROC_PER_NODE} --master_addr=127.0.0.1 --node_rank=${RANK} --master_port=12344 test.py
torchrun --nproc_per_node=${NUM_PROC_PER_NODE} --nnodes=${NUM_NODES} --nproc_per_node ${NUM_PROC_PER_NODE} --master_addr=${HOST} --node_rank=${RANK} --master_port=12344 ../test.py

