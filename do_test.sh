
rootpath=$HOME/VisualSearch
overwrite=0

#trainCollection=tgif-msrvtt10k
#valCollection=tv2016train
#val_set=setA
#config=w2vvpp_resnext101-resnet152_subspace
#model_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$val_set/$config/runs_0/model_best.pth.tar
#sim_name=$trainCollection/$valCollection/$val_set/$config

#./do_test.sh iacc.3 ~/VisualSearch/tgif-msrvtt10k/w2vvpp_train/tv2016train/setA/w2vvpp_resnext101-resnet152_subspace/runs_0/model_best.pth.tar tgif-msrvtt10k/tv2016train/setA/w2vvpp_resnext101-resnet152_subspace/runs_0

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 testCollection model_path sim_name query_sets"
    #exit
fi

testCollection=$1
testCollection=iacc.3
testCollection=native_language_queries
model_path=$2
model_path=$rootpath/w2vvpp/w2vvpp_resnext101_resnet152_subspace_bow_v191212.pth.tar
sim_name=$3
sim_name=w2vvpp_resnext101_resnet152_subspace_bow_v191212
query_sets=$4 # tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18 and tv19.avs.txt for TRECVID19
query_sets=tv16.avs.txt,tv17.avs.txt,tv18.avs.txt
query_sets=native_language_queries.txt

if [ ! -f "$model_path" ]; then
    echo "model not found: $model_path"
    exit
fi


python predictor.py $testCollection $model_path $sim_name \
    --query_sets $query_sets --save_embs \
    --rootpath $rootpath  --overwrite $overwrite
