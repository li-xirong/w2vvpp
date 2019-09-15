# w2vvpp
W2VV++: A fully deep learning solution for ad-hoc video search

## Requirements
* Ubuntu 16.04
* cuda 10
* python 2.7.12
* PyTorch 1.1.0
* tensorboard 1.14.0
* numpy 1.16.4

We used virtualenv to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.

```
virtualenv --system-site-packages ~/w2vvpp
source ~/w2vvpp/bin/activate
pip install -r requirements.txt
deactivate
```

## Get started

### Data

The sentence encoding network for W2VV++, namely ```MultiScaleTxtEncoder```, needs a pretrained word2vec (w2v) model. In this work, we use a w2v trained on English tags associated with 30 million Flickr images.  Run the following script to download the Flickr w2v model and extract the folder at $HOME/VisualSearch/. The zipped model is around 3.1 gigabytes, so the download may take a while.

```bash
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH; cd $ROOTPATH

# download and extract pre-trained word2vec
wget http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz
tar zxf word2vec.tar.gz
```

The following three datasets are used for training, validation and testing: tgif-msrvtt10k, tv2016train and iacc.3. For more information about these datasets, please see https://github.com/li-xirong/avs.


**Video feature data**
+ 4096-dim resnext101-resnet152: [tgif-msrvtt10k](http://lixirong.net/data/mm2019/tgif-msrvtt10k-mean_resnext101-resnet152.tar.gz)(1.6G), [tv2016train](http://lixirong.net/data/mm2019/tv2016train-mean_resnext101-resnet152.tar.gz)(2.9M), [iacc.3](http://lixirong.net/data/mm2019/iacc.3-mean_resnext101-resnet152.tar.gz)(4.7G)

```bash
# get visual features per dataset
wget http://lixirong.net/data/mm2019/tgif-msrvtt10k-mean_resnext101-resnet152.tar.gz
wget http://lixirong.net/data/mm2019/tv2016train-mean_resnext101-resnet152.tar.gz
wget http://lixirong.net/data/mm2019/iacc.3-mean_resnext101-resnet152.tar.gz
```

**Sentence data**
+ Sentences: [tgif-msrvtt10k](http://lixirong.net/data/mm2019/tgif-msrvtt10k-sent.tar.gz), [tv2016train](http://lixirong.net/data/mm2019/tv2016train-sent.tar.gz)
+ TRECVID 2016 / 2017 / 2018 AVS topics and ground truth: [iacc.3](http://lixirong.net/data/mm2019/iacc.3-avs-topics.tar.gz)

```bash
# get sentences
wget http://lixirong.net/data/mm2019/tgif-msrvtt10k-sent.tar.gz
wget http://lixirong.net/data/mm2019/tv2016train-sent.tar.gz
wget http://lixirong.net/data/mm2019/iacc.3-avs-topics.tar.gz
```




**Pre-trained models**
+ [W2VV++(subspace)](http://lixirong.net/data/mm2019/w2vvpp_resnext101_resnet152_subspace.pth.tar)(224 MB)

Model | TV16 | TV17 | TV18 
|--- | ---| ---| ---|
|W2VV++(subspace) | 0.150 | 0.197 | 0.109 |

Note that due to better implemenations including improved coding and the use of latest pytorch, the performance is better than those reported in our ACMMM'19 paper.

### Scripts for training, testing and evaluation

Before executing the following scripts, please check if the environment (data, software, etc) is ready by running [test_env.py](test_env.py):
```bash
python test_env.py
```

#### Do everything from sratch

```bash
source ~/w2vvpp/bin/activate
# build vocabulary on the training set
./do_build_vocab.sh

# train w2vvpp on tgif-msrvtt10k based on "w2vvpp_resnext101_subspace" config
trainCollection=tgif-msrvtt10k
valCollection=tv2016train
val_set=setA
model_config=w2vvpp_resnext101_subspace

./do_train.sh $trainCollection $valCollection $val_set $model_config

# test w2vvpp on iacc.3
./do_test.sh iacc.3 $trainCollection $valCollection $val_set $model_config

sim_name=$trainCollection/$valCollection/$val_set/$model_config

cd tv-avs-eval
./do_eval.sh iacc.3 $sim_name tv16
./do_eval.sh iacc.3 $sim_name tv17
./do_eval.sh iacc.3 $sim_name tv18
```

#### Test and evaluate a pre-trained model

Assume the model has been placed at the following path:
```bash
$rootpath/tgif-msrvtt10k/w2vvpp_train/tv2016train/setA/w2vvpp_resnext101_subspace/runs_0/w2vvpp_resnext101_resnet152_subspace.pth.tar
```

```bash

# apply the trained w2vvpp model on iacc.3 for answering tv16 / tv17 / tv18 queries
trainCollection=tgif-msrvtt10k
valCollection=tv2016train
val_set=setA
model_config=w2vvpp_resnext101_subspace
sim_name=$trainCollection/$valCollection/$val_set/$model_config

./do_test.sh iacc.3 $trainCollection $valCollection $val_set $model_config

# evaluate the performance
cd tv-avs-eval
./do_eval.sh iacc.3 $sim_name tv16  # tv16 infAP: 0.150
./do_eval.sh iacc.3 $sim_name tv17  # tv17 infAP: 0.197
./do_eval.sh iacc.3 $sim_name tv18  # tv18 infAP: 0.109
```

## Tutorials

1. [Use a pre-trained w2vv++ model to encode a given sentence](tutorial.ipynb)


## Citation

```
@inproceedings{mm19-w2vvpp,
title = {{W2VV}++: Fully Deep Learning for Ad-hoc Video Search},
author = {Xirong Li and Chaoxi Xu and Gang Yang and Zhineng Chen and Jianfeng Dong},
year = {2019},
booktitle = {ACMMM},
}
```
