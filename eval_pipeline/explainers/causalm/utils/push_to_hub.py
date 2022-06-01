import argparse
import logging
import warnings

import huggingface_hub
from huggingface_hub import create_repo, list_models, delete_repo

from modeling import LSTMCausalmForNonlinearSequenceClassification, BertCausalmForNonlinearSequenceClassification, \
    RobertaCausalmForSequenceClassification, GPT2CausalmForNonlinearSequenceClassification, RobertaCausalmConfig
from utils import OUTPUTS_CAUSALM, OUTPUTS_CEBAB, LSTM, ROBERTA, BERT, GPT2, CEBAB, OPENTABLE_BINARY, OPENTABLE_TERNARY, OPENTABLE_5_WAY, \
    PROJECT_DIR, ROBERTA_VOCAB_SIZE

from transformers import logging

warnings.filterwarnings(action='ignore', message=r'.*')

logging.set_verbosity_error()


def push_model(args):
    if args.is_causalm_model:
        pretrained_path = OUTPUTS_CAUSALM / args.task_name / args.model_architecture / f'{args.tc}__{args.cc}' / f'seed_{args.seed}' / 'downstream'
    else:
        pretrained_path = OUTPUTS_CEBAB / args.task_name / args.model_architecture / 'None__None' / f'seed_{args.seed}'
    pretrained_path = str(pretrained_path)

    if args.model_architecture == LSTM:
        model = LSTMCausalmForNonlinearSequenceClassification.from_pretrained(
            pretrained_path,
            fasttext_embeddings_path=str(PROJECT_DIR / 'utils' / 'lstm_embeddings.bin')
        )
    elif args.model_architecture == BERT:
        model = BertCausalmForNonlinearSequenceClassification.from_pretrained(pretrained_path)
    elif args.model_architecture == ROBERTA:
        config = RobertaCausalmConfig.from_pretrained(pretrained_path)
        if args.is_causalm_model:
            config.max_position_embeddings = 512
            config.type_vocab_size = 2
        else:
            config.max_position_embeddings = 514
            config.type_vocab_size = 1
        config.vocab_size = 50265
        config.save_pretrained(pretrained_path)
        print(f'Updated config file at: {pretrained_path}')
        model = RobertaCausalmForSequenceClassification.from_pretrained(pretrained_path, config=config)
    elif args.model_architecture == GPT2:
        model = GPT2CausalmForNonlinearSequenceClassification.from_pretrained(pretrained_path)
    else:
        raise RuntimeError(f'Illegal architecture "{args.model_architecture}"')

    if args.task_name == OPENTABLE_BINARY:
        setup = '2-class'
    elif args.task_name == OPENTABLE_TERNARY:
        setup = '3-class'
    elif args.task_name == OPENTABLE_5_WAY:
        setup = '5-class'
    else:
        raise RuntimeError(f'Illegal task "{args.task_name}"')

    repo_path = create_repo(
        organization=CEBAB,
        name=f'{args.model_architecture}.{CEBAB}.causalm.{args.tc}__{args.cc}.{setup}.exclusive.seed_{args.seed}',
        exist_ok=True
    )
    print(f'Pushing to: {repo_path}')
    model.push_to_hub(repo_path_or_name=repo_path, organization=CEBAB, use_auth_token=True)
    print('Success.')


def push_config(args):
    if args.is_causalm_model:
        pretrained_path = OUTPUTS_CAUSALM / args.task_name / args.model_architecture / f'{args.tc}__{args.cc}' / f'seed_{args.seed}' / 'downstream'
    else:
        pretrained_path = OUTPUTS_CEBAB / args.task_name / args.model_architecture / 'None__None' / f'seed_{args.seed}'
    pretrained_path = str(pretrained_path)

    if args.task_name == OPENTABLE_BINARY:
        setup = '2-class'
    elif args.task_name == OPENTABLE_TERNARY:
        setup = '3-class'
    elif args.task_name == OPENTABLE_5_WAY:
        setup = '5-class'
    else:
        raise RuntimeError(f'Illegal task "{args.task_name}"')

    repo_path = f'{CEBAB}/{args.model_architecture}.{CEBAB}.causalm.{args.tc}__{args.cc}.{setup}.exclusive.seed_{args.seed}'
    print(f'Pushing config to: {repo_path}')
    huggingface_hub.upload_file(
        path_or_fileobj=f'{pretrained_path}/config.json',
        path_in_repo='config.json',
        repo_id=repo_path
    )
    print('Success.')


def main():
    print()
    print()
    print()
    print()
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_causalm_model', type=bool, default=False)
    parser.add_argument('--task_name', type=str, default=OPENTABLE_BINARY)
    parser.add_argument('--model_architecture', type=str, default=ROBERTA)
    parser.add_argument('--tc', type=str, default='ambiance')
    parser.add_argument('--cc', type=str, default='food')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--push_model', type=bool, default=False)
    parser.add_argument('--push_config', type=bool, default=False)
    args = parser.parse_args()

    if args.push_model:
        push_model(args)
    if args.push_config:
        push_config(args)


if __name__ == '__main__':
    main()
