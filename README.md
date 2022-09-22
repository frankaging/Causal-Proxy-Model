![Python 3.7](https://img.shields.io/badge/python-3.7-blueviolet.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-MIT-05b502.svg?style=plastic)

# Causal Proxy Models For Concept-Based Model Explanations (CPM)
<p align="center">
  <b><a href="https://nlp.stanford.edu/~wuzhengx/">Zhengxuan Wu</a>*, <a href="https://www.kareldoosterlinck.com/">Karel D'Oosterlinck</a>*, <a href="https://atticusg.github.io/">Atticus Geiger</a>*, <a href="https://www.linkedin.com/in/amir-zur-a924ba187/">Amir Zur</a>, <a href="https://web.stanford.edu/~cgpotts/">Christopher Potts</a></b></span>
</p>

The codebase contains some implementations of our preprint [Causal Proxy Models For Concept-Based Model Explanations](https://nlp.stanford.edu/~wuzhengx/). In this paper, we introuce two variants of CPM, 
* CPM<sub>IN</sub>: Input-base CPM uses auxiliary token to represent the intervention, and is trained in a supervised way of predicting counterfactual output. This model is built on an input-level intervention.
* CPM<sub>HI</sub>: Hidden-state CPM uses Interchange Intervention Training (IIT) to localize concept information within its representations, and swaps hidden-states to represent the intervention. It is trained in a supervised way of predicting counterfactual output. This model is built on a hidden-state intervention.

This codebase contains implementations and experiments for **CPM<sub>HI</sub>**. If you experience any issues or have suggestions, please contact me either thourgh the issues page or at wuzhengx@cs.stanford.edu. 

## Citation
If you use this repository, please consider to cite our relevant papers:
```stex
  @article{geiger-etal-2021-iit,
        title={Causal Proxy Models For Concept-Based Model Explanations}, 
        author={Wu, Zhengxuan and D'Oosterlinck, Karel and Geiger, Atticus and Zur, Amir and Potts, Christopher},
        year={2022},
        eprint={},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
  }

  @article{geiger-etal-2021-iit,
        title={Inducing Causal Structure for Interpretable Neural Networks}, 
        author={Geiger, Atticus and Wu, Zhengxuan and Lu, Hanson and Rozner, Josh and Kreiss, Elisa and Icard, Thomas and Goodman, Noah D. and Potts, Christopher},
        year={2021},
        eprint={2112.00826},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
  }
```

## Requirements
- Python 3.6 or 3.7 are supported.
- Pytorch Version: 1.11.0
- Transfermers Version: 4.21.1
- Datasets Version: Version: 2.3.2

## Training **CPM<sub>HI</sub>**

```bash
python Proxy_training.py \
--model_name_or_path ./saved_models/bert-base-uncased.opentable.CEBaB.sa.5-class.exclusive.seed_42/ \
--task_name CEBaB \
--dataset_name CEBaB/CEBaB \
--do_train \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--learning_rate 8e-05 \
--output_dir ./proxy_training_results/your_first_try/ \
--cache_dir ./train_cache/ \
--seed 42 \
--report_to none \
--wandb_metadata wuzhengx:Causal-Proxy-Model \
--logging_steps 1 \
--alpha 1.0 \
--beta 1.0 \
--gemma 3.0 \
--overwrite_output_dir \
--intervention_h_dim 192 \
--counterfactual_type true \
--k 19684 \
--interchange_hidden_layer 10 \
--save_steps 10 \
--early_stopping_patience 20
```

