import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(sys.path[0], '..'))

from experiments.utils.dataset_builder import get_dataset
from experiments.utils.general_utils import seed_worker
from experiments.utils.log_likelihoods import get_log_likelihood
from scripts import train_models
from models import BasicNeuralNet

def get_models(name, size):
    if name == 'basic':
        return [BasicNeuralNet()]*size
    else:
        raise ValueError(f'Unknown dataset {name}')

def get_result(task, test_sets, models):
    if task == 'loglikelihoods':
        return get_log_likelihood(test_sets, models)
    else:
        raise ValueError(f'Unknown task {name}')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Distribution Analysis")
    parser.add_argument("--task", default="loglikelihoods", type=str, 
                        help="loglikelihoods")
    parser.add_argument("--dataset", default="mnist", type=str, 
                        help="mnist|fashion_mnist")
    parser.add_argument("--model", default="basic", type=str, 
                        help="basic")
    parser.add_argument("--size", default=5, type=int,
                        help="number of classifiers")
    parser.add_argument("--noise_start", default=0.0001, type=float,
                        help="start of noise schedule")
    parser.add_argument("--noise_end", default=0.02, type=float,
                        help="end of noise schedule")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--turnoff_wandb', action="store_true")
    args = parser.parse_args()
    
    if not args.turnoff_wandb:
        import wandb
        wandb.init(project='distribution_analysis')

    train_sets, test_sets = get_dataset(args.dataset, args.size, args.noise_start, args.noise_end)

    train_gens = []
    g = torch.Generator()
    g.manual_seed(0)

    for i in range(len(train_sets)):
        train_generator = DataLoader(train_sets[i],
                                    batch_size=10,
                                    shuffle=True,
                                    num_workers=8,
                                    pin_memory=True,
                                    drop_last=True,
                                    worker_init_fn=seed_worker,
                                    generator=g)
        train_gens.append(train_generator)
    
    print("Loaded dataset..")

    torch.manual_seed(args.seed)

    models = get_models(args.model, args.size)
    models, epoch_data_set, loss_data_set = train_models(train_gens, models, not args.turnoff_wandb)

    print("Trained models...")

    result = get_result(args.task, test_sets, models)
    model_states = []
    for i in range(len(models)):
        model_states.append(models[i].state_dict())

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/mnist", exist_ok=True)
    os.makedirs("results/fashion_mnist", exist_ok=True)

    filename = ""
    if args.dataset == 'mnist':
        filename =  f"results/mnist/{args.task}_{args.seed}.pt"
    elif args.dataset == 'fashion_mnist':
        filename =  f"results/fashion_mnist/{args.task}_{args.seed}.pt"
    torch.save(model_states, filename)