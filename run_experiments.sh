for seed in 66 77 88 99; do # 42; do
    CUDA_VISIBLE_DEVICES=0,6,8,9 python Proxy_training.py \
    --tokenizer_name bert-base-uncased \
    --model_name_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.2-class.exclusive.seed_${seed}/ \
    --high_level_model_type_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.2-class.exclusive.seed_${seed}/ \
    --task_name CEBaB \
    --dataset_name ./datasets/Proxy.CEBaB.sa.2-class.exclusive \
    --do_train \
    --do_eval \
    --mode align \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 8e-5 \
    --num_train_epochs 60 \
    --output_dir ./proxy_training_results/ \
    --cache_dir ./train_cache/ \
    --seed ${seed} \
    --report_to wandb \
    --wandb_metadata wuzhengx:Causal-Proxy-Model \
    --logging_steps 1 \
    --alpha 1.0 \
    --beta 1.0 \
    --gemma 3.0 \
    --overwrite_output_dir \
    --intervention_h_dim 128
done
