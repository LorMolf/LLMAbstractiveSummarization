#/bin/bash


CACHE_DIR="cache"
OUT_DIR="results"

SEED=42

# Config
MAX_TRAIN_SAMPLES=10
MAX_TEST_SAMPLES=5

MAX_STEPS=100
WARMUP_STEPS=25


TRAIN_BS=1
EVAL_BS=1

MODEL_NAME="HuggingFaceH4/zephyr-7b-beta"
model_type='zephyr'

# MODEL_NAME="NousResearch/Llama-2-7b-chat-hf"
MODEL_NAME="meta-llama/Llama-2-13b-hf"
MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
model_type="llama2"


# Data
DATASET_NAME="samsum"
TEXT_COL="dialogue"
SUM_COL="summary"


MAX_SOURCE_LENGTH=1024
MAX_TARGET_LENGTH=200

LR='1e-4'

# GEN PARAMS
TOP_K=30
TOP_P=0.9
TEMPERATURE=0.8


#CUDA_VISIBLE_DEVICES=1 python3 main.py 
CUDA_VISIBLE_DEVICES=0 python3 main_sft.py --dataset_name $DATASET_NAME \
                                       --text_column $TEXT_COL \
                                       --summary_column $SUM_COL \
                                       --val_max_target_length $MAX_TARGET_LENGTH \
                                       --max_source_length $MAX_SOURCE_LENGTH \
                                       --model_name_or_path $MODEL_NAME \
                                       --cache_dir $CACHE_DIR \
                                       --output_dir $OUT_DIR \
                                       --per_device_train_batch_size $TRAIN_BS \
                                       --per_device_eval_batch_size $EVAL_BS \
                                       --eval_accumulation_steps 30\
                                       --pad_to_max_length \
                                       --seed $SEED \
                                       --max_train_samples $MAX_TRAIN_SAMPLES \
                                       --use_peft \
                                       --fp16 \
                                       --max_steps $MAX_STEPS \
                                       --warmup_steps $WARMUP_STEPS \
                                       --neftune_noise_alpha '5'\
                                       --optim "paged_adamw_8bit" \
                                       --lr_scheduler_type "cosine" \
                                       --learning_rate $LR \
                                       --do_sample \
                                       --top_k $TOP_K \
                                       --top_p $TOP_P \
                                       --temperature $TEMPERATURE \
                                       --do_predict \
                                       --do_train \
                                       --report_to "wandb"
                                       #--max_predict_samples $MAX_TEST_SAMPLES \
                                       #--max_eval_samples $MAX_TEST_SAMPLES \
                                       #--do_eval \
                                       #--predict_with_generate \
                                       