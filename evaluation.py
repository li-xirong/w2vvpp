from __future__ import print_function
import os
import pickle

import numpy
import time
import numpy as np
import torch
from collections import OrderedDict

from generic_utils import Progbar


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm

        
def encode_data(model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """
    model.switch_to_train()

    vis_embs = None
    txt_embs = None
    vis_ids = ['']*len(data_loader.dataset)
    txt_ids = ['']*len(data_loader.dataset)

    pabr = Progbar(len(data_loader.dataset))
    for i, (vis_inputs, txt_inputs, idxs, batch_vis_ids, batch_txt_ids) in enumerate(data_loader):

        vis_emb = model.vis_net(vis_inputs)
        txt_emb = model.txt_net(txt_inputs)

        if vis_embs is None:
            vis_embs = np.zeros((len(data_loader.dataset), vis_emb.size(1)))
            txt_embs = np.zeros((len(data_loader.dataset), txt_emb.size(1)))

        vis_embs[idxs] = vis_emb.data.cpu().numpy().copy()
        txt_embs[idxs] = txt_emb.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            txt_ids[idx] = batch_txt_ids[j]
            vis_ids[idx] = batch_vis_ids[j]

        pbar.add(vis_emb.size(0))

    return vis_embs, txt_embs, vis_ids, txt_ids


def eval_v2t(vis_embs, txt_embs, npts=None, measure='cosine'):
    """
    Vision->Text (Vision Annotation)
    vis_embs: (N, N) matrix of videos/images
    txt_embs: (N, N) matrix of captions
    """
    images = l2norm(images)
    captions = l2norm(captions)

    if npts is None:
        npts = images.shape[0]

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)

    if measure == 'cosine':
        d = numpy.dot(images, captions.T)
        inds = numpy.argsort(d, axis=1)

        for index in range(npts):
            ind = inds[index][::-1]
            rank = numpy.where(ind == index)[0][0]

            ranks[index] = rank

            ranks[index] = rank
            top1[index] = ind[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mir = (1.0/(ranks+1)).mean()
    if return_ranks:
        return (r1, r5, r10, medr, meanr, mir), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr, mir)
