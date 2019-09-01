
rootpath=$HOME/VisualSearch
trainCollection=tgif-msrvtt10k
valCollection=tv2016train

testCollection=iacc.3

config=w2vvpp_resnext101-resnet152_subspace
config=w2vvpp_resnext101_subspace

logger_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$config/runs_0

overwrite=1

python predictor.py $testCollection \
    --rootpath $rootpath --text_sets tv16.avs.txt tv17.avs.txt tv18.avs.txt \
    --logger_path $logger_path --overwrite $overwrite
