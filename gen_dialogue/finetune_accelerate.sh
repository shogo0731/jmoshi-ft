export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NO_TORCH_COMPILE=1
export OMP_NUM_THREADS=4 #各プロセスが使うスレッド数（OpenMPによる並列スレッド数）を指定する環境変数
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="disabled" # wandbを無効にする

# 引数の説明
# --model_dtype                 # モデルのデータ型（float32/float16/bfloat16）
# --model_user_stream           # 両方の音声を予測対象
# --max_length                  # 入力シーケンスの最大長　（約41秒）
# --min_length                  # 入力シーケンスの最小長 （約10秒）
# --num_train_epochs            # 訓練エポック数
# --per_device_train_batch_size # デバイス毎の訓練バッチサイズ
# --num_warmup_steps            # 学習率を段階的に上げるウォームアップフェーズのステップ数
# --seed                        # 再現性のためのシード値

accelerate launch \
    --main_process_port 0 \
    --num_processes 4 \
    --num_machines 1 \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/zero3-fp16-act_ckpt_cpu-off-load.json \
    finetune.py \
        --launcher accelerate \
        --output_dir output/moshiko-finetuned \
        --train_data_files processed_data/gen_dialogue/train-*.parquet \
        --model_dir init_models/moshiko-both_streams-bfloat16 \
        --model_dtype bfloat16 \
        --moshi_speakers A \
        --model_user_stream \
        --max_length 1028 \
        --min_length 64 \
        --num_train_epochs 1000 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_warmup_steps 500 \
        --activation_checkpointing \
        --logging_steps 100 \
        --report_to wandb \
        --project_name moshi-finetuning \
        --save_steps 100 \
        --seed 42