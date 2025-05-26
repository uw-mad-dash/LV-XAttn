# python -m torch.distributed.run --nproc_per_node=1 \
#          lightseq/train_lightseq_no_trainer.py \
#         --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#         --data_path fake.pkl \
#         --bf16 \
#         --output_dir outputs \
#         --num_train_epochs 3    \
#         --per_device_train_batch_size 1 \
#         --per_device_eval_batch_size 1  \
#         --gradient_accumulation_steps 1 \
#         --evaluation_strategy no \
#         --save_strategy steps \
#         --save_steps 1000  \
#         --save_total_limit 1 \
#         --learning_rate 2e-5 \
#         --weight_decay 0.  \
#         --warmup_ratio 0.03  \
#         --lr_scheduler_type "cosine" \
#         --logging_steps 1  \
#         --fsdp "full_shard auto_wrap" \
#         --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#         --tf32 False  \
#         --model_max_length 16384  \
#         --gradient_checkpointing True  \
#         --lazy_preprocess True

NUM_NODES=2
NUM_PROC_PER_NODE=3
HOST="c301-003"
# RANK=0
RANK=${PMI_RANK}
echo $RANK
MODEL_SIZE="1b"
# torchrun --nproc_per_node=1 --nnodes=${NUM_NODES} --master_addr=10.10.1.1 --node_rank=${RANK} --master_port=12344 ${MODE}/lightseq/lightseq_async_attn.py --comm-mode lightseq --debug --bs 1 --n_heads 32 --run-mode test
# torchrun --nproc_per_node=${NUM_PROC_PER_NODE} --nnodes=${NUM_NODES} --nproc_per_node ${NUM_PROC_PER_NODE} --master_addr=127.0.0.1 --node_rank=${RANK} --master_port=12344 test.py
# torchrun --nproc_per_node=${NUM_PROC_PER_NODE} --nnodes=${NUM_NODES} --nproc_per_node ${NUM_PROC_PER_NODE} --master_addr=${HOST} --node_rank=${RANK} --master_port=12345 test_flamingo.py --fsdp --fsdp_use_orig_params
torchrun --nproc_per_node=${NUM_PROC_PER_NODE} \
        --nnodes=${NUM_NODES} \
        --master_addr=${HOST} \
        --node_rank=${RANK} \
        --master_port=12345 \
        test_llama3.py
        # test_llama3.py \
        # --fsdp \
        # --fsdp_use_orig_params \
        # --model_size ${MODEL_SIZE}
        # --xattn_no_overlap
