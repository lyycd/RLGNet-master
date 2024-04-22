# RLGNet



## Environment Setup

```bash
conda create -n RLGNet python=3.10

conda activate RLGNet

pip install -r requirement.txt
```

## Train models

We've saved the hyperparameters from the paper in `'args.yaml'`, so you can train the models directly:

```bash
python main.py -d <dataset> --train_local --train_global --train_repeat
```

The program will sequentially train the global, local, and repeat modules. If you have sufficient computational resources, we suggest the following approach to reduce training time:

1.	Train the local and global modules simultaneously in two different terminals:

```bash
python main.py -d <dataset> --train_local
```

```bash
python main.py -d <dataset> --train_global
```

2. Once the above commands have completed, train the repeat module:


```bash
python main.py -d <dataset> --train_repeat
```

## Evaluate models

To evaluate the results of the model:

```bash
python main.py -d <dataset> --test_repeat [--multi_step] [--test_global] [--test_local]
```

Adding the `--multi_step` parameter will evaluate the results of multi-step reasoning, otherwise, it will evaluate the results of single-hop reasoning. If you need to only evaluate the effects of the local and global models separately, add `--test_local` and `--test_global`, respectively.

## Model Weight

We've uploaded the model weights as well, just download the corresponding dataset weights to evaluate the model.
