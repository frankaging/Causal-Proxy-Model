![Python 3.7](https://img.shields.io/badge/python-3.7-blueviolet.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-MIT-05b502.svg?style=plastic)

<h1 align="center">
  <b>Causal Proxy Models For Concept-Based Model Explanations</b>
</h1>

<p align="center">
  <b><a href="https://nlp.stanford.edu/~wuzhengx/">Zhengxuan Wu</a>*, <a href="https://www.kareldoosterlinck.com/">Karel D'Oosterlinck</a>*, <a href="https://atticusg.github.io/">Atticus Geiger</a>*, <a href="https://www.linkedin.com/in/amir-zur-a924ba187/">Amir Zur</a>, <a href="https://web.stanford.edu/~cgpotts/">Christopher Potts</a></b></span>
</p>

The codebase contains some implementations of our preprint [Causal Proxy Models For Concept-Based Model Explanations](https://arxiv.org/abs/2209.14279). In this paper, we introuce two variants of CPM, 
* CPM<sub>IN</sub>: Input-base CPM uses auxiliary token to represent the intervention, and is trained in a supervised way of predicting counterfactual output. This model is built on an input-level intervention.
* CPM<sub>HI</sub>: Hidden-state CPM uses Interchange Intervention Training (IIT) to localize concept information within its representations, and swaps hidden-states to represent the intervention. It is trained in a supervised way of predicting counterfactual output. This model is built on a hidden-state intervention.

This codebase contains implementations and experiments for both **CPM<sub>IN</sub>** and **CPM<sub>HI</sub>**. If you experience any issues or have suggestions, please contact me either thourgh the issues page or at wuzhengx@cs.stanford.edu or at karel.doosterlinck@ugent.be. 

## Citation
If you use this repository, please consider to cite our relevant papers:
```stex
  @article{wu-etal-2021-cpm,
        title={Causal Proxy Models For Concept-Based Model Explanations}, 
        author={Wu, Zhengxuan and D'Oosterlinck, Karel and Geiger, Atticus and Zur, Amir and Potts, Christopher},
        year={2022},
        eprint={2209.14279},
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


## Installation
First clone the directory. Then run the following command to initialize the submodules:

```bash
git submodule init; git submodule update
```

## Loading Black-box Models for CEBaB
These models are avaliable from the [CEBaB website](https://cebabing.github.io/CEBaB/). Here is one example about how to load these models!

```python
from transformers import AutoTokenizer, BertForNonlinearSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("CEBaB/bert-base-uncased.CEBaB.sa.5-class.exclusive.seed_42")

model = BertForNonlinearSequenceClassification.from_pretrained("CEBaB/bert-base-uncased.CEBaB.sa.5-class.exclusive.seed_42")
```

## Loading **CPMs** for CEBaB
We aim to make all of our **CPMs** public. Currently, they are be found on [our huggingface repo](https://huggingface.co/CPMs).

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("CPMs/cpm.hi.bert-base-uncased.layer.10.size.192")

model = AutoModelForSequenceClassification.from_pretrained("CPMs/cpm.hi.bert-base-uncased.layer.10.size.192")
```

Note that we also have different helpers to load these models into our explainer module. Please refer to notebooks under `experiments` folder.

## Training **CPM<sub>IN</sub>**

To train **CPM<sub>IN</sub>**, we follow the basic finetuning setup since the intervention is on the inputs. To train, you should first go to `CEBaB-inclusive/eval_pipeline/`; and you can run the following command to train.

```bash
python main.py \
--model_architecture bert-base-uncased \
--train_setting inclusive \
--model_output_dir model_output \
--output_dir output \
--flush_cache true \
--task_name opentable_5_way \
--batch_size 128 \
--k_array 19684
```

To train with different variants of *approximate counterfactuals*, you need to change the flag `--train_setting approximate` for metadata-sampled counterfactuals. Note that in this setting, you can ignore the field `--k_array`. You should change `--model_architecture` for different model architectures.

## Training **CPM<sub>HI</sub>**

To train **CPM<sub>HI</sub>**, we adapt interchange intervention training (IIT). To train, you can use the following command, and you can refer to our paper for configurations.

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
To train with different variants of *approximate counterfactuals*, you need to change the flag `--counterfactual_type approximate` for metadata-sampled counterfactuals. Note that in this setting, you can ignore the field `--k`. You should change `--model_name_or_path` for different model architectures. These models can be downloaded from [CEBaB website](https://cebabing.github.io/CEBaB/).

