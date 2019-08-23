
rootpath=$HOME/VisualSearch2
trainCollection=tgif-msrvtt10k
valCollection=tv2016train
val_set=setA
logger_postfix=runs_0

config=w2vvpp_resnext101-resnet152_subspace
config=w2vvpp_resnext101_subspace

python trainer.py $trainCollection $valCollection \
    --rootpath $rootpath --config $config --val_set $val_set \
    --logger_postfix $logger_postfix
