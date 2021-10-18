import os
import sherpa
import argparse
import itertools 


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default='0,1,2,3', type=str)
parser.add_argument('--max_concurrent', help='Number of concurrent processes', type=int, default=8)
args = parser.parse_args()


os.makedirs("ParallelResults/Models", exist_ok=True)


# Define all the hyperparameters and their corresponding ranges
parameters = [
    sherpa.Choice("activation", ["relu", "elu"]),
    sherpa.Choice("batch_normalization", [0, 1]),
    sherpa.Continuous("dropout", [0, 1]),
    sherpa.Continuous("learning_rate_decay", [0., 0.5]),
    sherpa.Continuous("learning_rate", [0.0001, 0.1]),
    sherpa.Discrete("number_of_nodes", [32, 256]),
    sherpa.Discrete("number_of_layers", [3, 25]),
    sherpa.Choice("preprocessing", ["log", "min_max", "none", "standardize"]),
]


GPUs = [int(x) for x in args.gpus.split(',')]
processes_per_gpu = args.max_concurrent // len(GPUs)
resources = list(itertools.chain.from_iterable(itertools.repeat(x, processes_per_gpu) for x in GPUs))

scheduler = sherpa.schedulers.LocalScheduler(resources=resources)


# Search algorithm for HPs
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=1000)


sherpa.optimize(
    parameters=parameters,
    algorithm=algorithm,
    lower_is_better=True,
    output_dir="ParallelResults",
    scheduler=scheduler,
    max_concurrent=args.max_concurrent,
    command="/home/jott1/tf2_env/bin/python main.py",
)
