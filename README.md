# CEAL: Cost-Effective Active Learning for Deep Image Classification
> Unofficial Pytorch Implementation of [CEAL](https://arxiv.org/abs/1701.03551)

- *This is not an official implementation of CEAL!*
- I've only tested this code with CIFAR10 and modified CUB200 dataset because of limitation of time and GPU resources I've got.
- Feel free to leave me an issue or make a pull request!

## Files
- `CEAL.ipynb`: CEAL on CIFAR10 dataset
- `CEAL_CUB.ipynb`: CEAL on modified CUB200 dataset
- `model.py`: defines model

## modified CUB200
- Because of limited GPU resource, I modified original CUB200 dataset with 200 class into 20 classes.
- First, sort the data by class name. (A-Z)
- And select the top 20 classes for training and testing.
- I used train/test split provided by TensorFlow.
    - You can find it on [kaggle](https://www.kaggle.com/datasets/skyil7/cub200-2011-with-traintest-split)

## Original Paper
Wang, Keze, et al. "Cost-effective active learning for deep image classification." IEEE Transactions on Circuits and Systems for Video Technology 27.12 (2016): 2591-2600.