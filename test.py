
import pickle


vocab_path = 'VisualSearch/tgif-msrvtt10k/TextData/vocab/bow_5.pkl'
vocab = pickle.load(open(vocab_path, 'rb'))
for i in range(10):
    print i, vocab[i], vocab.find(vocab[i])