from methods.utils.modeling_utils import BertForNonlinearSequenceClassification
from methods.causalm.bert_causalm import BertCausalmForNonlinearSequenceClassification
from methods.utils.constants import BERT

from huggingface_hub import create_repo

if __name__ == '__main__':
    seeds = list(range(42, 47))
    treatments = ['ambiance', 'food', 'service', 'noise']
    architecture = BERT
    dataset = 'CEBaB'
    setup = '2-class'
    task = 'causalm'
    train_set = 'exclusive'

    for s in seeds:
        for tc in treatments:
            # get model
            model_path = f'./outputs_causalm/opentable_binary/{architecture}/{tc}__None/seed_{s}/downstream'
            model = BertCausalmForNonlinearSequenceClassification.from_pretrained(model_path)

            # push to hub
            repo_name = f'{architecture}.{dataset}.{task}.{tc}.{setup}.{train_set}.seed_{s}'
            repo_path = create_repo(repo_name, organization='CEBaB', private=True, exist_ok=True)
            model.push_to_hub(repo_path_or_name=repo_path, organization='CEBaB', use_auth_token=True, private=True)

    for s in seeds:
        # get model
        model_path = f'./outputs_run_opentable/opentable_binary/{architecture}/None__None/seed_{s}'
        model = BertForNonlinearSequenceClassification.from_pretrained(model_path)

        # push to hub
        repo_name = f'{architecture}.{dataset}.{task}.factual.{setup}.{train_set}.seed_{s}'
        repo_path = create_repo(repo_name, organization='CEBaB', exist_ok=True)
        model.push_to_hub(repo_path_or_name=repo_path, organization='CEBaB', use_auth_token=True)

