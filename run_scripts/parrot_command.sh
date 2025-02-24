

train_path=src/run_clm_lora.py
model_path=meta-llama/Llama-3.2-3B-Instruct
model_save=output/models/parrot-hint-lora-7b

# HOST_NUM will be 1
torchrun --nnodes 1 --node_rank $INDEX --nproc_per_node 1 \
    ${train_path} \
    --model_name_or_path ${model_path} \
    --train_file data/data_parrot_hf.json \
    --use_lora True \
    --lora_config train/lora_config.json \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1.5 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --block_size 512 \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --fp16 True \
    --fp16_full_eval True \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing True \
    --output_dir ${model_save}