# Hyperparameter Optimization

## Dataset

The dataset is appreciatively borrowed from Kimoske, Metodiev & Thaler's [EnergyFlow](https://energyflow.network/docs/datasets/#quark-and-gluon-nsubs) library. We have 100,000 jets, each with 45 N-subjettiness observables and a truth value (`1 = quark`, `0 = gluon`). It can be loaded with 

```
all_events = energyflow.qg_nsubs.load(num_data=-1, cache_dir=DATA_DIRECTORY)
```

In the [benchmark model](https://github.com/hep-lbdl/LBL-HEP-ML-training/blob/043b673b7b1dd0487d9e86b58a81e605428c3448/hyperparameter-optimization/scripts/benchmark_model.py#L32) into a Pytorch `DataLoader` with a `[train, val, test]` split of `[80000, 10000, 10000]`.

## Model

A [quick and dirty network](https://github.com/hep-lbdl/LBL-HEP-ML-training/blob/043b673b7b1dd0487d9e86b58a81e605428c3448/hyperparameter-optimization/scripts/benchmark_model.py#L22) is provided in the benchmark model, which follows the overall structure used in [the original paper](https://arxiv.org/abs/1704.08249). It is an `n_layer` fully connected network with `n_channels` hidden channels. There are options for normalization and dropout. 

## Usage

If you already know and love Pytorch Lightning, this model should be ready to train out-of-the-box. If you prefer something else, pull out any functions or model definitions you find useful. At this stage, there are no hyperparameter optimization tools included, so add in your favorite!

## State-of-the-art performance

Given by "background rejection rate". That is: for a particular signal efficiency (defined as `signal_goal=0.5` in the [configuration file](https://github.com/hep-lbdl/LBL-HEP-ML-training/blob/main/hyperparameter-optimization/scripts/config.yaml)), background rejection rate is `1 / fpr`. Epsilon is conventionally used for this value, so `eps` is used in the [function to calculate this metric](https://github.com/hep-lbdl/LBL-HEP-ML-training/blob/043b673b7b1dd0487d9e86b58a81e605428c3448/hyperparameter-optimization/scripts/benchmark_model.py#L88).