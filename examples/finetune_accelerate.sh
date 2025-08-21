export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NO_TORCH_COMPILE=1
export OMP_NUM_THREADS=4 #各プロセスが使うスレッド数（OpenMPによる並列スレッド数）を指定する環境変数
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/zero3-fp16-warmlr-act_ckpt.json \
    finetune.py \
        --launcher accelerate \
        --output_dir output/moshiko-finetuned \
        --train_data_files processed_data/spokenwoz_sample/train-*.parquet \
        --model_dir init_models/moshiko-both_streams-float32 \
        --model_dtype float32 \
        --model_user_stream \
        --max_length 2048 \
        --min_length 128 \
        --num_train_epochs 1000 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --num_warmup_steps 500 \
        --activation_checkpointing \
        --logging_steps 100 \
        --report_to wandb \
        --project_name moshi-finetuning \
        --save_steps 100

