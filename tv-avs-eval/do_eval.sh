rootpath=$HOME/VisualSearch
trainCollection=tgif-msrvtt10k
valCollection=tv2016train

overwrite=0

test_collection=$1
edition=$2
model_config=$3

input=$rootpath/$test_collection/w2vvpp_test/$edition.avs.txt/$trainCollection/w2vvpp_train/$valCollection/$model_config/runs_0/id.sent.score.txt


bash do_txt2xml.sh $input $edition $overwrite

python trec_eval.py ${input}.xml --rootpath $rootpath --collection $test_collection --edition $edition --overwrite $overwrite

