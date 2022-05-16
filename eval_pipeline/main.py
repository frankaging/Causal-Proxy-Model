import argparse
import os

import datasets
import numpy as np
import pandas as pd

from eval_pipeline.explainers import CONEXP, LIME, CausaLM, TCAV, INLP
from eval_pipeline.models.bert import BERTForCEBaB
# TODO: get rid of these seed maps or describe somewhere how they work
from eval_pipeline.utils import (
    OPENTABLE_BINARY,
    OPENTABLE_TERNARY,
    OPENTABLE_5_WAY,
    BERT,
    SEEDS_ELDAR2ZEN,
    SEEDS_ZEN2ELDAR,
    preprocess_hf_dataset,
    save_output,
    average_over_seeds, SEEDS_ELDAR
)
from eval_pipeline.pipeline import run_pipelines


def get_caces_for_ultimate_results_table(args, cebab):
    # TODO we need to get rid of this function and compute these more elegantly
    train, dev, test = preprocess_hf_dataset(cebab, one_example_per_world=True, verbose=1, dataset_type=args.dataset_type)

    # init model and explainer, assuming all are based on the same architecture
    models = [BERTForCEBaB(f'CEBaB/{args.model_architecture}.CEBaB.sa.{args.num_classes}.exclusive.seed_{SEEDS_ELDAR2ZEN[s]}', device=args.device)
              for s in args.seeds]
    explainers = [CONEXP()] * len(models)

    # run pipeline
    df = run_pipelines(models, explainers, train, dev, dataset_type=args.dataset_type)[-1]

    relevant_directions = [('Negative', 'Positive'), ('Negative', 'unknown'), ('unknown', 'Positive')]
    relevant_rows = [idx for idx in df.index if (idx[1], idx[2]) in relevant_directions]
    relevant_cols = [col for col in df.columns if col[-1] == 'ICaCE']
    df = df.loc[relevant_rows, relevant_cols]
    for c in df.columns:
        df[c] = df[c].apply(lambda x: x[1])
    df['avg'] = df.apply(np.mean, axis=1)
    df['std'] = df.apply(np.std, axis=1)
    caces, caces_std = df['avg'].to_numpy().reshape(4, 3), df['std'].to_numpy().reshape(4, 3)

    if args.output_dir:
        filename = f'caces__{args.task_name}__bert-base-uncased.csv'
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        pd.DataFrame(caces,
                     columns=['neg to pos', 'neg to unk', 'unk to pos'],
                     index=['ambiance', 'food', 'noise', 'service']).to_csv(f'{args.output_dir}/{filename}')


def get_explainers(seed, args, model):
    causalm = CausaLM(
        factual_model_path=f'CEBaB/{args.model_architecture}.CEBaB.causalm.factual.{args.num_classes}.exclusive.seed_{seed}',
        ambiance_model_path=f'CEBaB/{args.model_architecture}.CEBaB.causalm.ambiance.{args.num_classes}.exclusive.seed_{seed}',
        food_model_path=f'CEBaB/{args.model_architecture}.CEBaB.causalm.food.{args.num_classes}.exclusive.seed_{seed}',
        noise_model_path=f'CEBaB/{args.model_architecture}.CEBaB.causalm.noise.{args.num_classes}.exclusive.seed_{seed}',
        service_model_path=f'CEBaB/{args.model_architecture}.CEBaB.causalm.service.{args.num_classes}.exclusive.seed_{seed}',
        empty_cache_after_run = True,
        device=args.device,
    )
    return [CONEXP(), LIME(), causalm, INLP(device=args.device), TCAV(model, device=args.device)]


def main():
    # TODO: add explanations of these arguments or examples
    # TODO: add dev/test argument
    # TODO: work with different model classes

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default=OPENTABLE_BINARY)
    parser.add_argument('--seeds', nargs='+', default=SEEDS_ELDAR)
    parser.add_argument('--model_architecture', type=str, default=BERT)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # data
    cebab = datasets.load_dataset('CEBaB/CEBaB', use_auth_token=True)
    if args.task_name == OPENTABLE_BINARY:
        args.dataset_type = '2-way'
        args.num_classes = '2-class'
    elif args.task_name == OPENTABLE_TERNARY:
        args.dataset_type = '3-way'
        args.num_classes = '3-class'
    elif args.task_name == OPENTABLE_5_WAY:
        args.dataset_type = '5-way'
        args.num_classes = '5-class'
    else:
        raise ValueError(f'Unsupported task \"{args.task_name}\"')

    train, dev, test = preprocess_hf_dataset(cebab, one_example_per_world=True, verbose=1, dataset_type=args.dataset_type)

    # for every seed
    pipeline_outputs = []
    for seed in args.seeds:
        # TODO: support multiple models
        model = BERTForCEBaB(f'CEBaB/{args.model_architecture}.CEBaB.sa.{args.num_classes}.exclusive.seed_{SEEDS_ELDAR2ZEN[int(seed)]}', device=args.device)

        explainers = get_explainers(seed, args, model)
        models = [model] * len(explainers) # TODO: these are shallow copies! If one explainer manipulates a model without copying, this could give bugs for other methods!

        pipeline_output = run_pipelines(models, explainers, train, dev, dataset_type=args.dataset_type, shorten_model_name=True)
        pipeline_outputs.append(pipeline_output)

    # average over the seeds
    pipeline_outputs_averaged = average_over_seeds(pipeline_outputs)

    # save output
    if args.output_dir:
        filename_suffix = f'{args.task_name}__{args.model_architecture}'
        save_output(os.path.join(args.output_dir, f'final__{args.task_name}__{args.model_architecture}'), filename_suffix, *pipeline_outputs_averaged)


if __name__ == '__main__':
    main()
