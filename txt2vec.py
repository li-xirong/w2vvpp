import numpy as np
import pickle
from bigfile import BigFile
from config import logger
from textlib import TextTool


class Txt2Vec(object):
    '''
    norm: 0 no norm, 1 l_1 norm, 2 l_2 norm
    '''
    def __init__(self, data_path, norm=0, clean=True, lang='en'):
        logger.info(self.__class__.__name__+ ' initializing ...')
        self.data_path = data_path
        self.norm = norm
        self.lang = lang
        self.clean = clean
        assert (norm in [0, 1, 2]), 'invalid norm %s' % norm

        
    def _preprocess(self, query):
        words = TextTool.tokenize(query, clean=self.clean, language=self.lang) 
        return words
    
    def _do_norm(self, vec):
        assert (1 == self.norm or 2 == self.norm)
        norm = np.linalg.norm(vec, self.norm)
        return vec / (norm + 1e-10) # avoid divide by ZERO

    def _encoding(self, words):
        raise Exception("encoding not implemented yet!")
    
    def encoding(self, query):
        words = self._preprocess(query)
        vec = self._encoding(words)
        if self.norm > 0:
            return self.do_norm(vec)
        return vec
    
       
class BowVec(Txt2Vec):

    def __init__(self, data_path, norm=0, clean=True, lang='en'):
        super(BowVec, self).__init__(data_path, norm, clean, lang)
        self.vocab = pickle.load(open(data_path, 'rb'))
        self.ndims = len(self.vocab)
        logger.info('vob size: %d, vec dim: %d' % (len(self.vocab), self.ndims))
     
    def _encoding(self, words):   
        vec = np.zeros(self.ndims, )
        
        for word in words:
            idx = self.vocab.find(word)
            if idx>=0:
                vec[idx] += 1      
        return vec 


class W2Vec(Txt2Vec):
    def __init__(self, data_path, norm=0, clean=True, lang='en'):
        super(W2Vec, self).__init__(data_path, norm, clean, lang)
        self.w2v = BigFile(data_path)
        vocab_size, self.ndims = self.w2v.shape()
        logger.info('vob size: %d, vec dim: %d' % (vocab_size, self.ndims))

    def _encoding(self, words):
        renamed, vectors = self.w2v.read(words)

        if len(vectors) > 0:
            vec = np.array(vectors).mean(axis=0)
        else:
            vec = np.zeros(self.ndims, )
        return vec
        

class BowVecNSW(BowVec):
    def __init__(self, data_path, norm=0, clean=True, lang='en'):
        super(BowVecNSW, self).__init__(data_path, norm, clean, lang)
        if '_nsw' not in data_path:
            logger.error('WARNING: loaded a vocabulary that contains stopwords')

    def _preprocess(self, query):
        words = TextTool.tokenize(query, clean=self.clean, language=self.lang, remove_stopword=True) 
        return words        


class W2VecNSW(W2Vec):

    def _preprocess(self, query):
        words = TextTool.tokenize(query, clean=self.clean, language=self.lang, remove_stopword=True) 
        return words        


NAME_TO_T2V = {'bow': BowVec, 'bow_nsw':  BowVecNSW, 'w2v': W2Vec, 'w2v_nsw': W2VecNSW}


def get_txt2vec(name):
    assert name in NAME_TO_T2V
    return NAME_TO_T2V[name]


if __name__ == '__main__':
    t2v = BowVec('VisualSearch/tgif-msrvtt10k/TextData/vocab/bow_5.pkl')
    t2v = BowVecNSW('VisualSearch/tgif-msrvtt10k/TextData/vocab/bow_nsw_5.pkl')
    t2v = BowVecNSW('VisualSearch/tgif-msrvtt10k/TextData/vocab/bow_5.pkl')
    t2v = W2Vec('VisualSearch/word2vec/flickr/vec500flickr30m')
    t2v = W2VecNSW('VisualSearch/word2vec/flickr/vec500flickr30m')
    
    vec = t2v.encoding('a dog runs on grass')
    print vec.shape
