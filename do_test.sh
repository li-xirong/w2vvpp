
rootpath=$HOME/VisualSearch
trainCollection=tgif-msrvtt10k
valCollection=tv2016train

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 testCollection config"
    exit
fi

testCollection=$1
config=$2

model_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$config/runs_0/w2vvpp_resnext101_resnet152_subspace.pth.tar
sim_name=$trainCollection/$valCollection/$config

overwrite=0

python predictor.py $testCollection $model_path $sim_name \
    --query_sets tv16.avs.txt tv17.avs.txt tv18.avs.txt \
    --rootpath $rootpath  --overwrite $overwrite
