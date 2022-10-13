# Vision Transformer

## Introduction

![ViT.png](https://s2.loli.net/2022/01/19/w3CyXNrhEeI7xOF.png)

Network for Vision Transformer. The pytorch version. 

## Quick start

1. Clone this repository

```shell
git clone https://github.com/Runist/torch_Vision_Transformer
```
2. Install torch_Vision_Transformer from source.

```shell
cd torch_Vision_Transformer
pip install -r requirements.txt
```
3. Download the **flower dataset**.
```shell
wget https://github.com/Runist/image-classifier-keras/releases/download/v0.2/dataset.zip
unzip dataset.zip
```
4. Modifying the [config.py](https://github.com/Runist/torch_Vision_Transformer/blob/master/config.py).
5. Download pretrain weights, the url in [utils.py](https://github.com/Runist/torch_Vision_Transformer/blob/master/utils.py).
6. Start train your model.

```shell
python train.py
```
7. Open tensorboard to watch loss, learning rate etc. You can also see training process and training process and validation prediction.

```shell
tensorboard --logdir ./summary/log
```
![tensorboard.png](https://s2.loli.net/2022/10/12/p7KtB1uXMkqvreN.png)
8. Get prediction of model.

```shell
python predict.py
```

## Train your dataset

You need to store your data set like this:

```shell
├── train
│   ├── daisy
│   ├── dandelion
│   ├── roses
│   ├── sunflowers
│   └── tulips
└── validation
    ├── daisy
    ├── dandelion
    ├── roses
    ├── sunflowers
    └── tulips
```

## Reference

Appreciate the work from the following repositories:

- [WZMIAOMIAO](https://github.com/bubbliiiing)/[vision_transformer](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)


## License

Code and datasets are released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
