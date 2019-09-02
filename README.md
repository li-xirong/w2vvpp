# w2vvpp
W2VV++: A fully deep learning solution for ad-hoc video search

## Requirements
* Ubuntu 16.04
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

### Scripts for training, testing and evaluation
```
source ~/w2vvpp/bin/activate
# build vocabulary on the training set
./do_build_vocab.sh

# train w2vvpp on tgif-msrvtt10k based on "w2vvpp_resnext101_subspace" config
# you can change config name in do_train.sh
./do_train.sh

# test w2vvpp on iacc.3
./do_test.sh

./do_eval.sh

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
