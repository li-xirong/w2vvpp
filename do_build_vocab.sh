
train_collection='tgif-msrvtt10k'
overwrite=0

for encoding in soft_bow_nsw bow bow_nsw gru
do
    python build_vocab.py $train_collection --encoding $encoding --overwrite $overwrite
done


