
rootpath=$HOME/VisualSearch
trainCollection=tgif-msrvtt10k
valCollection=tv2016train
val_set=setA

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 testCollection trainCollection valCollection val_set config"
    exit
fi

testCollection=$1
trainCollection=$2
valCollection=$3
val_set=$4
config=$5

model_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$val_set/$config/runs_0/w2vvpp_resnext101_resnet152_subspace.pth.tar

if [ ! -f "$model_path" ]; then
    echo "model not found: $model_path"
    exit
fi

sim_name=$trainCollection/$valCollection/$val_set/$config

overwrite=0

python predictor.py $testCollection $model_path $sim_name \
    --query_sets tv16.avs.txt tv17.avs.txt tv18.avs.txt \
    --rootpath $rootpath  --overwrite $overwrite
