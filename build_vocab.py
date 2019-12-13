# --------------------------------------------------------
# Pytorch W2VV++
# Written by Xirong Li & Chaoxi Xu
# --------------------------------------------------------

from __future__ import print_function
import pickle
from collections import Counter
import os
import sys

import numpy as np
from common import *
from bigfile import BigFile
from evaluation import compute_sim
from util import checkToSkip, makedirsforfile
from textlib import TextTool, Vocabulary


def read_from_txt_file(cap_file):
    captions = []
    with open(cap_file, 'rb') as fr:
        for line in fr:
            cap_id, caption = line.split(' ', 1)
            captions.append(caption.strip())
    return captions

def build_vocab(options, cap_file, encoding, threshold, lang):
    nosw = '_nsw' in encoding
    logger.info("Build a simple vocabulary wrapper from %s", cap_file)
    captions = read_from_txt_file(cap_file)
    counter = Counter()
    
    for i, caption in enumerate(captions):
        tokens = TextTool.tokenize(caption, language=lang, remove_stopword=nosw)
        counter.update(tokens)
  
    # Discard if the occurrence of the word is less than min_word_cnt.
    word_counts = [(word,cnt) for word, cnt in counter.items() if cnt >= threshold]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary(encoding)
    if 'gru' in encoding:
        vocab.add('<pad>')
        vocab.add('<start>')
        vocab.add('<end>')
        vocab.add('<unk>')

    if 'soft' in encoding:
        vocab.word2sim = {}
        w2v_data_path = os.path.join(options.rootpath, 'word2vec', 'flickr', 'vec500flickr30m')
        w2v_reader = BigFile(w2v_data_path)
        words = [w[0] for w in word_counts]
        renames, vectors = w2v_reader.read(words)
        rename2index = dict( zip( renames, range(len(renames)) ) )
        vectors = np.array(vectors)
        sim_matrix = compute_sim(vectors, vectors)
        inds = np.argsort(sim_matrix, axis=1)

    # Add words to the vocabulary.
    for word, c in word_counts:
        vocab.add(word)
        if 'soft' in encoding:
            if word in rename2index:
                idx = rename2index[word]
                ind = inds[idx][::-1]
                word_sim = [(renames[i] , sim_matrix[idx][i]) for i in ind[:21]]
            else:
                word_sim = []
            vocab.word2sim[word] = word_sim
    return vocab, word_counts


def process(options, collection):
    overwrite = options.overwrite
    rootpath = options.rootpath
    threshold = options.threshold
    encoding = options.encoding
    language = options.language

    vocab_file = os.path.join(rootpath, collection, 'TextData', 'vocab', '%s_%d.pkl'% (encoding, threshold))
    count_file = os.path.join(os.path.dirname(vocab_file), '%s_%d.txt'% (encoding, threshold))

    if checkToSkip(vocab_file, overwrite):
        return 0
    
    cap_file = os.path.join(rootpath, collection, 'TextData', '%s.caption.txt'%collection)
    vocab, word_counts = build_vocab(options, cap_file, encoding, threshold=threshold, lang=language)
    
    makedirsforfile(vocab_file)
    with open(vocab_file, 'wb') as fw:
        pickle.dump(vocab, fw, pickle.HIGHEST_PROTOCOL)
    logger.info("Saved vocabulary of %d words to %s", len(vocab), vocab_file)

    with open(count_file, 'w') as fw:
        fw.write('\n'.join(['%s %d' % x for x in word_counts]))
    
    logger.info("Saved word-counts to %s", count_file)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
        
    from optparse import OptionParser
 
    parser = OptionParser(usage="""usage: %prog [options] collection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--language", default=DEFAULT_LANG, type="string", help="language (default: %s)" % DEFAULT_LANG)
    parser.add_option("--encoding", default="bow", type="choice", choices=TEXT_ENCODINGS, help="text encoding strategy. Valid choices are %s. (default: %s)" % (TEXT_ENCODINGS, DEFAULT_TEXT_ENCODING))
    parser.add_option("--threshold", default=5, type="int", help="minimum word occurrence (default: %d)" % MIN_WORD_COUNT)
    
                      
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    assert(options.language in ['en', 'zh']), 'language %s not supported' % options.language
    
    return process(options, args[0])

if __name__=="__main__":
    sys.exit(main())
