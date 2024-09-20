#!/bin/bash#!/bin/bash

DATA_FILE=$1
OUT_DIR=$2
MODEL_SIZE=${3:-"13b"}
PROJ_DIR=llava-v1.5-mlp2x-336px-pretrain-vicuna-$MODEL_SIZE-v1.5
BATCH_PER_GPU=${4:-16}
GRAD_ACC_STEPS=${5:-1}
LR=${6:-"2e-5"}
NUM_EPOCHS=${7:-1}
NUM_GPUS=${8:-8}
NUM_NODES=${9:-1}
SAVE_FREQ=${10:-500}
EFF_BATCH=$(($NUM_NODES * $NUM_GPUS * $BATCH_PER_GPU * $GRAD_ACC_STEPS))
echo "Effective batch size: $EFF_BATCH"
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
    --deepspeed scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-$MODEL_SIZE-v1.5 \
    --version v1 \
    --data_path $DATA_FILE \
    --image_folder '' \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $PROJ_DIR/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_GPU \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_FREQ \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb
