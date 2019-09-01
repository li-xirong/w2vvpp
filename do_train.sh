
rootpath=$HOME/VisualSearch
trainCollection=tgif-msrvtt10k
valCollection=tv2016train
val_set=setA

config=w2vvpp_resnext101-resnet152_subspace
config=w2vvpp_resnext101_subspace

python trainer.py $trainCollection $valCollection \
    --rootpath $rootpath --config $config --val_set $val_set
