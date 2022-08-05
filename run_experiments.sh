# BERT, true counterfactuals, layer 12, hdim 192, different ks
for h_dim in 192; do
    for seed in 42; do # 42 66 77 88 99
        for k in 10 100 500 1000 3000 6000 9848 19684; do  
            CUDA_VISIBLE_DEVICES=6,7,8,9 python Proxy_training.py \
            --model_name_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.5-class.exclusive.seed_42/ \
            --task_name CEBaB \
            --dataset_name CEBaB/CEBaB \
            --do_train \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 2 \
            --learning_rate 8e-5 \
            --num_train_epochs 30 \
            --output_dir ./proxy_training_results/bert-true-counterfactuals/ \
            --cache_dir ./train_cache/ \
            --seed 42 \
            --report_to none \
            --wandb_metadata wuzhengx:Causal-Proxy-Model \
            --logging_steps 1 \
            --alpha 1.0 \
            --beta 1.0 \
            --gemma 3.0 \
            --overwrite_output_dir \
            --intervention_h_dim ${h_dim} \
            --k ${k} \
            --counterfactual_type true \
            --interchange_hidden_layer 12
        done
    done
done