rootpath=$HOME/VisualSearch
sim_name=tgif-msrvtt10k/tv2016train/setA/w2vvpp_resnext101_subspace
overwrite=0

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 testCollection topic_set $sim_name"
    exit
fi

test_collection=$1
topic_set=$2
sim_name=$3

score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt

bash do_txt2xml.sh $score_file $topic_set $overwrite
python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite

