
train_collection='tgif-msrvtt10k'

for encoding in bow bow_nosw gru
do
    python build_vocab.py $train_collection --encoding $encoding
done


