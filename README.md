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
Required | Size | Description
--- | --- | ---
word2vec | 3.1G | English word2vec trained on Flickr tags

Run the following script to download and extract a pre-trained word2vec. The extracted data is placed in $HOME/VisualSearch/.

```
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH; cd $ROOTPATH

# download and extract pre-trained word2vec
wget http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz
tar zxf word2vec.tar.gz
```

Dataset | No. of video clips or gifs | No. of sentences | Downloads |
--- | --- | --- | ---
**training**: tgif-msrvtt10k | 110,855 | 324,534 | [resnext-resnet](http://lixirong.net/data/mm19/tgif-msrvtt10k-resnext-resnet.tar.gz) (XX.X GB), [sentences](http://lixirong.net/data/mm19/tgif-msrvtt10k-sent.tar.gz) (XX.X MB) |
**validation**: tv2016train | 200 | 400 | [resnext-resnet](http://http://lixirong.net/data/mm19/tv2016train-resnext-resnet.tar.gz) XX.X MB, [sentences](http://lixirong.net/data/mm19/tv2016train-sent.tar.gz) (XX.XX MB)  |
**test**: IACC.3 | 335,944 | 90 | [resnext-resnet](http://http://lixirong.net/data/mm19/iacc.3-resnext-resnet.tar.gz) (XX.X GB), [sentences](http://lixirong.net/data/mm19/iacc.3-sent.tar.gz) (XX.X MB) |

For more information about the three datasets, please refer to https://github.com/li-xirong/avs.

```bash
# get visual features
wget http://lixirong.net/data/mm19/tgif-msrvtt10k-resnext-resnet.tar.gz
wget http://http://lixirong.net/data/mm19/tv2016train-resnext-resnet.tar.gz
wget http://http://lixirong.net/data/mm19/iacc.3-resnext-resnet.tar.gz

# get sentences
wget http://lixirong.net/data/mm19/tgif-msrvtt10k-sent.tar.gz
wget http://lixirong.net/data/mm19/tv2016train-sent.tar.gz
wget http://lixirong.net/data/mm19/iacc.3-sent.tar.gz
```

### Scripts
```
source ~/w2vvpp/bin/activate
# build vocabulary on the training set
./do_build_vocab.sh

# train w2vvpp on tgif-msrvtt10k based on "w2vvpp_resnext101_subspace" config
# you can change config name in do_train.sh
./do_train.sh

# test w2vvpp on iacc.3
./do_test.sh

```




## Citation

```
@inproceedings{mm19-w2vvpp,
title = {{W2VV}++: Fully Deep Learning for Ad-hoc Video Search},
author = {Xirong Li and Chaoxi Xu and Gang Yang and Zhineng Chen and Jianfeng Dong},
year = {2019},
booktitle = {ACMMM},
}
```
