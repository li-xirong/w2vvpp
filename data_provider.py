import torch
import torch.utils.data as data
import pickle
from textlib import TextTool, Vocabulary

def collate_img(data):

    images, idxs, img_ids = zip(*data)

    images = torch.stack(images, 0)

    return images, idxs, img_ids

def collate_text(data):

    data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_w2vs, cap_bows, idxs, cap_ids = zip(*data)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    target = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        target[i, :end] = cap[:end]

    # multi-scale
    # word2vec
    cap_w2vs = torch.stack(cap_w2vs, 0) if cap_w2vs[0] is not None else None
    # bag of words
    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    targets = (target, cap_w2vs, cap_bows)

    return targets, lengths, idxs, cap_ids


class ImageDataset(data.Dataset):

    def __init__(self, img_feat_file):
        self.img_feat_file = img_feat_file
        self.img_ids = img_feat_file.names
        self.length = len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_tensor = torch.Tensor(self.img_feat.read_one(img_id))

        return img_tensor, index, img_id

    def __len__(self):
        return self.length

    
class TextDataset(data.Dataset):

    def __init__(self, cap_file, w2v_feat_dir, bow_vocab_path, rnn_vocab_path, lang='en'):
        self.captions = {}

        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)

        self.t2v_w2v = Text2VecW2V(w2v_feat_dir)
        self.t2v_bow = Text2VecBoW(bow_vocab_path)
        self.rnn_vocab = pickle.load(open(rnn_vocab_path, 'rb'))
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        caption = self.captions[cap_id]
        vocab = self.vocab

        # word2vec
        if self.w2v2vec is not None:
            cap_w2v = self.w2v2vec.mapping(caption)
            if cap_w2v is None:
                cap_w2v = torch.randn(self.w2v2vec.ndims)
            else:
                cap_w2v = torch.Tensor(cap_w2v)
        else:
            cap_w2v = None
        # bag-of-words
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.randn(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        tokens = clean_str_filterstop(caption)
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        cap_tensor = torch.Tensor(caption)

        return cap_tensor, cap_w2v, cap_bow, index, cap_id

    def __len__(self):
        return self.length


def img_provider(img_feat, batch_size=100, num_workers=2):

    dset = ImageData(img_feat)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_img)
    return data_loader

def text_provider(cap_file, vocab, w2v2vec, bow2vec, batch_size=100, num_workers=2):

    dset = TextData(cap_file, w2v2vec, bow2vec, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_text)
    return data_loader
