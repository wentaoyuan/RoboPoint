#!/bin/bash

DATA=$1
MODEL=$2
VISION=$3
OUT_DIR=$4
NUM_GPUS=${5:-8}
NUM_NODES=${6:-1}
SAVE_FREQ=${7:-500}
if [[ -z $SLURM_JOB_NODELIST ]]; then
    LAUNCHER="deepspeed --num_gpus $NUM_GPUS"
else
    nodelist=""
    for node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
        nodelist+="$node slots=$NUM_GPUS\n"
    done
    echo $nodelist > /tmp/hostfile
    LAUNCHER="deepspeed --hostfile=/tmp/hostfile --num_nodes $NUM_NODES --num_gpus $NUM_GPUS"
fi
$LAUNCHER robopoint/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL \
    --version plain \
    --data_path $DATA/chat.json \
    --image_folder $DATA/images \
    --vision_tower $VISION \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_FREQ \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb