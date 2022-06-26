# scripts for gpt2 model main results.
for h_dim in 192; do
    for class_num in 2 3 5; do
        for seed in 42 66 77 88 99; do
            CUDA_VISIBLE_DEVICES=6,7,8,9 python Proxy_training.py \
            --tokenizer_name gpt2 \
            --model_name_or_path ./saved_models/gpt2.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
            --high_level_model_type_or_path ./saved_models/gpt2.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
            --task_name CEBaB \
            --dataset_name ./datasets/Proxy.CEBaB.sa.${class_num}-class.exclusive \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --learning_rate 8e-5 \
            --num_train_epochs 60 \
            --output_dir ./proxy_training_results/gpt2-results/ \
            --cache_dir ./train_cache/ \
            --seed ${seed} \
            --report_to wandb \
            --wandb_metadata wuzhengx:Causal-Proxy-Model \
            --logging_steps 1 \
            --alpha 1.0 \
            --beta 1.0 \
            --gemma 3.0 \
            --overwrite_output_dir \
            --intervention_h_dim ${h_dim} \
            --classifier_dropout 0.1 \
            --encoder_dropout 0.1
        done
    done
done

# scripts for gpt2 model control condition without IIT training objective.
for h_dim in 192; do
    for class_num in 2 3 5; do
        for seed in 42 66 77 88 99; do
            CUDA_VISIBLE_DEVICES=6,7,8,9 python Proxy_training.py \
            --tokenizer_name gpt2 \
            --model_name_or_path ./saved_models/gpt2.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
            --high_level_model_type_or_path ./saved_models/gpt2.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
            --task_name CEBaB \
            --dataset_name ./datasets/Proxy.CEBaB.sa.${class_num}-class.exclusive \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --learning_rate 8e-5 \
            --num_train_epochs 60 \
            --output_dir ./proxy_training_results/gpt2-control-results/ \
            --cache_dir ./train_cache/ \
            --seed ${seed} \
            --report_to wandb \
            --wandb_metadata wuzhengx:Causal-Proxy-Model \
            --logging_steps 1 \
            --alpha 1.0 \
            --beta 0.0 \
            --gemma 0.0 \
            --overwrite_output_dir \
            --intervention_h_dim ${h_dim} \
            --classifier_dropout 0.1 \
            --encoder_dropout 0.1
        done
    done
done

# scripts for LSTM model main results.
# for h_dim in 75; do
#     for class_num in 2 3 5; do
#         for seed in 42 66 77 88 99; do
#             CUDA_VISIBLE_DEVICES=1,2,3,4 python Proxy_training.py \
#             --tokenizer_name bert-base-uncased \
#             --model_name_or_path ./saved_models/lstm.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --high_level_model_type_or_path ./saved_models/lstm.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --task_name CEBaB \
#             --dataset_name ./datasets/Proxy.CEBaB.sa.${class_num}-class.exclusive \
#             --do_train \
#             --do_eval \
#             --max_seq_length 128 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 32 \
#             --learning_rate 8e-5 \
#             --num_train_epochs 60 \
#             --output_dir ./proxy_training_results/RoBERTa-results/ \
#             --cache_dir ./train_cache/ \
#             --seed ${seed} \
#             --report_to wandb \
#             --wandb_metadata wuzhengx:Causal-Proxy-Model \
#             --logging_steps 1 \
#             --alpha 1.0 \
#             --beta 1.0 \
#             --gemma 3.0 \
#             --overwrite_output_dir \
#             --intervention_h_dim ${h_dim} \
#             --classifier_dropout 0.1 \
#             --encoder_dropout 0.1
#         done
#     done
# done

# scripts for LSTM model control condition without IIT training objective.
# for h_dim in 75; do
#     for class_num in 2 3 5; do
#         for seed in 42 66 77 88 99; do
#             CUDA_VISIBLE_DEVICES=1,2,3,4 python Proxy_training.py \
#             --tokenizer_name bert-base-uncased \
#             --model_name_or_path ./saved_models/lstm.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --high_level_model_type_or_path ./saved_models/lstm.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --task_name CEBaB \
#             --dataset_name ./datasets/Proxy.CEBaB.sa.${class_num}-class.exclusive \
#             --do_train \
#             --do_eval \
#             --max_seq_length 128 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 32 \
#             --learning_rate 8e-5 \
#             --num_train_epochs 60 \
#             --output_dir ./proxy_training_results/RoBERTa-control-results/ \
#             --cache_dir ./train_cache/ \
#             --seed ${seed} \
#             --report_to wandb \
#             --wandb_metadata wuzhengx:Causal-Proxy-Model \
#             --logging_steps 1 \
#             --alpha 1.0 \
#             --beta 0.0 \
#             --gemma 0.0 \
#             --overwrite_output_dir \
#             --intervention_h_dim ${h_dim} \
#             --classifier_dropout 0.1 \
#             --encoder_dropout 0.1
#         done
#     done
# done

# scripts for RoBERTa model main results.
# for h_dim in 192; do
#     for class_num in 2 3 5; do
#         for seed in 42 66 77 88 99; do
#             CUDA_VISIBLE_DEVICES=0,6,8,9 python Proxy_training.py \
#             --tokenizer_name roberta-base \
#             --model_name_or_path ./saved_models/roberta-base.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --high_level_model_type_or_path ./saved_models/roberta-base.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --task_name CEBaB \
#             --dataset_name ./datasets/Proxy.CEBaB.sa.${class_num}-class.exclusive \
#             --do_train \
#             --do_eval \
#             --max_seq_length 128 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 32 \
#             --learning_rate 8e-5 \
#             --num_train_epochs 60 \
#             --output_dir ./proxy_training_results/RoBERTa-results/ \
#             --cache_dir ./train_cache/ \
#             --seed ${seed} \
#             --report_to wandb \
#             --wandb_metadata wuzhengx:Causal-Proxy-Model \
#             --logging_steps 1 \
#             --alpha 1.0 \
#             --beta 1.0 \
#             --gemma 3.0 \
#             --overwrite_output_dir \
#             --intervention_h_dim ${h_dim} \
#             --classifier_dropout 0.1 \
#             --encoder_dropout 0.1
#         done
#     done
# done

# scripts for RoBERTa model control condition without IIT training objective.
# for h_dim in 192; do
#     for class_num in 3 5; do
#         for seed in 42 66 77 88 99; do
#             CUDA_VISIBLE_DEVICES=1,2,8,9 python Proxy_training.py \
#             --tokenizer_name roberta-base \
#             --model_name_or_path ./saved_models/roberta-base.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --high_level_model_type_or_path ./saved_models/roberta-base.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --task_name CEBaB \
#             --dataset_name ./datasets/Proxy.CEBaB.sa.${class_num}-class.exclusive \
#             --do_train \
#             --do_eval \
#             --max_seq_length 128 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 32 \
#             --learning_rate 8e-5 \
#             --num_train_epochs 60 \
#             --output_dir ./proxy_training_results/RoBERTa-control-results/ \
#             --cache_dir ./train_cache/ \
#             --seed ${seed} \
#             --report_to wandb \
#             --wandb_metadata wuzhengx:Causal-Proxy-Model \
#             --logging_steps 1 \
#             --alpha 1.0 \
#             --beta 0.0 \
#             --gemma 0.0 \
#             --overwrite_output_dir \
#             --intervention_h_dim ${h_dim} \
#             --classifier_dropout 0.1 \
#             --encoder_dropout 0.1
#         done
#     done
# done

# scripts for BERT model main results.
# for h_dim in 192; do
#     for class_num in 2 3 5; do
#         for seed in 66 77 88 99; do # 42
#             CUDA_VISIBLE_DEVICES=0,6,8,9 python Proxy_training.py \
#             --tokenizer_name bert-base-uncased \
#             --model_name_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --high_level_model_type_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --task_name CEBaB \
#             --dataset_name ./datasets/Proxy.CEBaB.sa.${class_num}-class.exclusive \
#             --do_train \
#             --do_eval \
#             --max_seq_length 128 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 32 \
#             --learning_rate 8e-5 \
#             --num_train_epochs 60 \
#             --output_dir ./proxy_training_results/BERT-results/ \
#             --cache_dir ./train_cache/ \
#             --seed ${seed} \
#             --report_to wandb \
#             --wandb_metadata wuzhengx:Causal-Proxy-Model \
#             --logging_steps 1 \
#             --alpha 1.0 \
#             --beta 1.0 \
#             --gemma 3.0 \
#             --overwrite_output_dir \
#             --intervention_h_dim ${h_dim} \
#             --classifier_dropout 0.1 \
#             --encoder_dropout 0.1
#         done
#     done
# done

# scripts for BERT model control condition without IIT training objective.
# for h_dim in 192; do
#     for class_num in 2 3 5; do
#         for seed in 66 77 88 99; do
#             CUDA_VISIBLE_DEVICES=0,6,8,9 python Proxy_training.py \
#             --tokenizer_name bert-base-uncased \
#             --model_name_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --high_level_model_type_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#             --task_name CEBaB \
#             --dataset_name ./datasets/Proxy.CEBaB.sa.${class_num}-class.exclusive \
#             --do_train \
#             --do_eval \
#             --max_seq_length 128 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 32 \
#             --learning_rate 8e-5 \
#             --num_train_epochs 60 \
#             --output_dir ./proxy_training_results/BERT-control-results/ \
#             --cache_dir ./train_cache/ \
#             --seed ${seed} \
#             --report_to wandb \
#             --wandb_metadata wuzhengx:Causal-Proxy-Model \
#             --logging_steps 1 \
#             --alpha 1.0 \
#             --beta 0.0 \
#             --gemma 0.0 \
#             --overwrite_output_dir \
#             --intervention_h_dim ${h_dim} \
#             --classifier_dropout 0.1 \
#             --encoder_dropout 0.1
#         done
#     done
# done

# scripts for interchange dimension experiments
# as our results are rather with low variance, we 
# did not run with multi-seeds in this ablation study.
# for h_dim in 1 4 16 32 64 128 192; do
#     for class_num in 2 3 5; do
#         CUDA_VISIBLE_DEVICES=0,6,8,9 python Proxy_training.py \
#         --tokenizer_name bert-base-uncased \
#         --model_name_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#         --high_level_model_type_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.${class_num}-class.exclusive.seed_${seed}/ \
#         --task_name CEBaB \
#         --dataset_name ./datasets/Proxy.CEBaB.sa.${class_num}-class.exclusive \
#         --do_train \
#         --do_eval \
#         --max_seq_length 128 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --learning_rate 8e-5 \
#         --num_train_epochs 60 \
#         --output_dir ./proxy_training_results/ \
#         --cache_dir ./train_cache/ \
#         --seed ${seed} \
#         --report_to wandb \
#         --wandb_metadata wuzhengx:Causal-Proxy-Model \
#         --logging_steps 1 \
#         --alpha 1.0 \
#         --beta 1.0 \
#         --gemma 3.0 \
#         --overwrite_output_dir \
#         --intervention_h_dim ${h_dim} \
#         --classifier_dropout 0.1 \
#         --encoder_dropout 0.1
#     done
# done