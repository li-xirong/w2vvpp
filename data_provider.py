import torch
import torch.utils.data as data
import pickle
from bigfile import BigFile
from textlib import TextTool, Vocabulary
from txt2vec import get_lang, BowVec, BowVecNSW, W2Vec, W2VecNSW, IndexVec

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

def collate_pair(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    vis_feats, cap_idxs, cap_w2vs, cap_bows, idxs, vis_ids, cap_ids = zip(*data)

    vis_feats = torch.stack(vis_feats, 0)

    lengths = [len(cap) for cap in cap_idxs]
    target = torch.zeros(len(cap_idxs), max(lengths)).long()
    for i, cap in enumerate(cap_idxs):
        end = lengths[i]
        target[i, :end] = cap[:end]

    cap_w2vs = torch.stack(cap_w2vs, 0) if cap_w2vs[0] is not None else None
    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    targets = (target, cap_w2vs, cap_bows)

    return vis_feats, targets, lengths, idxs, vis_ids, cap_ids


class VisionDataset(data.Dataset):

    def __init__(self, params):
        self.vis_feat_file = BigFile(params['vis_feat']) if isinstance(params['vis_feat'], str) else params['vis_feat']
        self.vis_ids = self.vis_feat_file.names
        self.length = len(self.vis_ids)

    def __getitem__(self, index):
        vis_id = self.vis_ids[index]
        vis_tensor = self.get_tensor_by_id(vis_id)

        return vis_tensor, index, vis_id

    def get_tensor_by_id(self, vis_id):
        vis_tensor = torch.Tensor(self.vis_feat_file.read_one(vis_id))

        return vis_tensor

    def __len__(self):
        return self.length

    
class TextDataset(data.Dataset):

    def __init__(self, params):
        cap_file = params['cap']

        self.t2v_idx = IndexVec(params['rnn']) if isinstance(params['rnn'], str) else params['rnn']
        self.t2v_w2v = W2VecNSW(params['w2v']) if isinstance(params['w2v'], str) else params['w2v']
        self.t2v_bow = BowVecNSW(params['bow']) if isinstance(params['bow'], str) else params['bow']
              
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
        cap_idx, cap_w2v, cap_bow = self.get_tensor_by_id(cap_id)

        return cap_idx, cap_w2v, cap_bow, index, cap_id

    def get_tensor_by_id(self, cap_id):
        caption = self.captions[cap_id]

        if self.t2v_idx is not None:
            cap_idx = self.t2v_idx.encoding(caption)
            cap_idx = torch.Tensor(cap_idx)
        else:
            cap_idx = None

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

        return cap_idx, cap_w2v, cap_bow

    def __len__(self):
        return self.length


class PairDataset(data.Dataset):

    def __init__(self, params):
        self.visData = VisionDataset(params)
        self.txtData = TextDataset(params)

        self.cap_ids = self.txtData.cap_ids
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        vis_id = self.get_visId_by_capId(cap_id)

        cap_idx, cap_w2v, cap_bow = self.txtData.get_tensor_by_id(cap_id)
        vis_tensor = self.visData.get_tensor_by_id(vis_id)

        return vis_tensor, cap_idx, cap_w2v, cap_bow, index, vis_id, cap_id

    def get_visId_by_capId(self, cap_id):
        vis_id = cap_id.split('#', 1)[0]

        return vis_id

    def __len__(self):
        return self.length


def img_provider(params):
    data_loader = torch.utils.data.DataLoader(dataset=VisionDataset(params),
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


def pair_provider(params):
    data_loader = torch.utils.data.DataLoader(dataset=PairDataset(params),
                                              batch_size=params['batch_size'],
                                              shuffle=params['shuffle'],
                                              pin_memory=True,
                                              num_workers=params['num_workers'],
                                              collate_fn=collate_pair)
    return data_loader


if __name__ == '__main__':
    import os
    data_path = 'VisualSearch'
    collection = 'tgif-msrvtt10k'
    vid_feat = 'mean_resnext101_resnet152'
    vid_feat_dir = os.path.join(data_path, collection, 'FeatureData', vid_feat)
 
    img_loader = img_provider({'vis_feat': vid_feat_dir, 'batch_size':100, 'num_workers':2})
    
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
        

    pair_loader = pair_provider({'vis_feat': vid_feat_dir, 'cap':cap, 'bow':bow, 'w2v':w2v, 'rnn':rnn, 'batch_size':100, 'num_workers':2, 'shuffle':True})
    for i, (vis_feats, captions, lengths, idxs, vis_ids, cap_ids) in enumerate(pair_loader):
        print i, vis_feats.shape, captions[0].shape, len(lengths), len(cap_ids)
        break
