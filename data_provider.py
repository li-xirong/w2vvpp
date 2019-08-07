import torch
import torch.utils.data as data
import pickle
from bigfile import BigFile
from textlib import TextTool, Vocabulary
from txt2vec import get_lang, BowVec, BowVecNSW, W2Vec, W2VecNSW

def collate_img(data):
    feats, idxs, img_ids = zip(*data)
    feats = torch.stack(feats, 0)
    return feats, idxs, img_ids

def collate_text(data):

    data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_w2vs, cap_bows, idxs, cap_ids = zip(*data)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    target = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        target[i, :end] = cap[:end]

    cap_w2vs = torch.stack(cap_w2vs, 0) if cap_w2vs[0] is not None else None
    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    targets = (target, cap_w2vs, cap_bows)

    return targets, lengths, idxs, cap_ids


class ImageDataset(data.Dataset):

    def __init__(self, params):
        img_feat_dir = params['img_feat_dir']
        self.img_feat_file = BigFile(img_feat_dir)
        self.img_ids = self.img_feat_file.names
        self.length = len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_tensor = torch.Tensor(self.img_feat_file.read_one(img_id))

        return img_tensor, index, img_id

    def __len__(self):
        return self.length

    
class TextDataset(data.Dataset):

    def __init__(self, params):
        cap_file = params['cap']
        w2v_feat_dir = params['w2v']
        bow_vocab_path = params['bow']
        rnn_vocab_path = params['rnn']
        self.lang = get_lang(rnn_vocab_path)
            
        self.t2v_w2v = W2VecNSW(w2v_feat_dir) if w2v_feat_dir else None
        self.t2v_bow = BowVecNSW(bow_vocab_path) if bow_vocab_path else None
        self.rnn_vocab = pickle.load(open(rnn_vocab_path, 'rb'))
              
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)

        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        caption = self.captions[cap_id]
        rnn_vocab = self.rnn_vocab

        if self.t2v_w2v is not None:
            cap_w2v = self.t2v_w2v.encoding(caption)
            cap_w2v = torch.Tensor(cap_w2v)
        else:
            cap_w2v = None
            
        if self.t2v_bow is not None:
            cap_bow = self.t2v_bow.encoding(caption)
            cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        tokens =  TextTool.tokenize(caption, language=self.lang, remove_stopword=False)
        tokens =  ['<start>'] + tokens + ['<end>']
        caption = [rnn_vocab(token) for token in tokens]
        cap_tensor = torch.Tensor(caption)

        return cap_tensor, cap_w2v, cap_bow, index, cap_id

    def __len__(self):
        return self.length


def img_provider(params):
    data_loader = torch.utils.data.DataLoader(dataset=ImageDataset(params),
                                              batch_size=params['batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=params['num_workers'],
                                              collate_fn=collate_img)
    return data_loader


def txt_provider(params):   
    data_loader = torch.utils.data.DataLoader(dataset=TextDataset(params),
                                              batch_size=params['batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=params['num_workers'],
                                              collate_fn=collate_text)
    return data_loader


if __name__ == '__main__':
    import os
    data_path = 'VisualSearch'
    collection = 'tgif-msrvtt10k'
    vid_feat = 'mean_resnext101_resnet152'
    vid_feat_dir = os.path.join(data_path, collection, 'FeatureData', vid_feat)
 
    img_loader = img_provider({'img_feat_dir': vid_feat_dir, 'batch_size':100, 'num_workers':2})
    
    for i, (feat_vecs, idxs, img_ids) in enumerate(img_loader):
        print i, feat_vecs.shape, len(idxs)
        break
    
    
    cap = os.path.join(data_path, collection, 'TextData', '%s.caption.txt' % collection)
    bow = os.path.join(data_path, collection, 'TextData', 'vocab', 'bow_nsw_5.pkl')
    w2v = os.path.join(data_path, 'word2vec/flickr/vec500flickr30m')
    rnn = os.path.join(data_path, collection, 'TextData', 'vocab', 'gru_5.pkl')
    
    txt_loader = txt_provider({'cap':cap, 'bow':bow, 'w2v':w2v, 'rnn':rnn, 'batch_size':100, 'num_workers':2})
    
    for i, (captions, lengths, idxs, cap_ids) in enumerate(txt_loader):
        print i, captions[0].shape, len(lengths), len(cap_ids)
        break
        