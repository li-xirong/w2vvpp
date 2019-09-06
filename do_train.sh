
rootpath=$HOME/VisualSearch
trainCollection=tgif-msrvtt10k
valCollection=tv2016train
val_set=setA

config=$1

python trainer.py $trainCollection $valCollection \
    --rootpath $rootpath --config $config --val_set $val_set
