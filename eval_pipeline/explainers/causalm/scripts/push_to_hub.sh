#!/bin/bash

for task_name in opentable_binary opentable_ternary opentable_5_way; do
    for model_architecture in lstm bert-base-uncased roberta-base gpt2; do
        for tc in None ambiance food noise service; do
            for seed in 42 43 44 45 46; do
                # control concept
                if [[ ${tc} == "None" ]]; then
                    cc=None;
                elif [[ ${tc} == "food" ]]; then
                    cc=service;
                else
                    cc=food;
                fi

                # is causalm model
                if [[ ${tc} == "None" ]]; then
                    python utils/push_to_hub.py \
                    \
                    --push_config True \
                    --push_model True \
                    --task_name ${task_name} \
                    --model_architecture ${model_architecture} \
                    --tc ${tc} \
                    --cc ${cc} \
                    --seed ${seed}
                else
                    python utils/push_to_hub.py \
                    --is_causalm_model True \
                    \
                    --push_config True \
                    --push_model True \
                    --task_name ${task_name} \
                    --model_architecture ${model_architecture} \
                    --tc ${tc} \
                    --cc ${cc} \
                    --seed ${seed}
                fi

            done
        done
    done
done