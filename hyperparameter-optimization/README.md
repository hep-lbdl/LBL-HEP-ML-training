# Hyperparameter Optimization

## Dataset

The dataset is appreciatively borrowed from Kimoske, Metodiev & Thaler's [EnergyFlow](https://energyflow.network/docs/datasets/#quark-and-gluon-nsubs) library. We have 100,000 jets, each with 45 N-subjettiness observables and a truth value (`1 = quark`, `0 = gluon`). It can be loaded with 

```
all_events = energyflow.qg_nsubs.load(num_data=-1, cache_dir=DATA_DIRECTORY)
```

In the [benchmark model](https://github.com/hep-lbdl/LBL-HEP-ML-training/blob/main/hyperparameter-optimization/scripts/benchmark_model.py) provided, this dataset is converted in [setup()](https://github.com/hep-lbdl/LBL-HEP-ML-training/blob/8c4f2e477991aee650aa508e2b1ef0048dd59d3e/hyperparameter-optimization/scripts/benchmark_model.py#L31) into a Pytorch `DataLoader` with a `[train, val, test]` split of `[80000, 10000, 10000]`.

## Model

A quick and di