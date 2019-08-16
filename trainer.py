# --------------------------------------------------------
# Pytorch W2VV++
# Written by Xirong Li & Chaoxi Xu
# --------------------------------------------------------

from __future__ import print_function

import os
import sys
import time
import json
import shutil
import pickle
import logging
import argparse

import torch
import numpy as np
import tensorboard_logger as tb_logger

import data_provider as data
from txt2vec import get_txt2vec
from model import get_model, get_we
import util
from common import *
from bigfile import BigFile
from generic_utils import Progbar
from evaluation import encode_data, eval_v2t


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('trainCollection', type=str,
                        help='train collection')
    parser.add_argument('valCollection', type=str,
                        help='validation collection')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],
                        help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument('--val_set', type=str,
                        help='validation collection set (setA, setB). (default: setA)')
    parser.add_argument('--num_epochs', default=80, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--logger_postfix', default='runs_0',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--config_name', type=str, default='mean_pyresnext-101_rbps13k',
                        help='model configuration file. (default: mean_pyresnext-101_rbps13k')

    args = parser.parse_args()
    return args

def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent = 2))

    rootpath = opt.rootpath
    trainCollection = opt.trainCollection
    valCollection = opt.valCollection

    config = load_config('configs/%s.py' % opt.config_name)

    logger_path = os.path.join(rootpath, trainCollection, 'w2vvpp_train', valCollection, opt.config_name, opt.logger_postfix)
    print(logger_path)

    if util.checkToSkip(os.path.join(logger_path, 'model_best.pth.tar'), opt.overwrite):
        sys.exit(0)
    util.makedirs(logger_path)

    tb_logger.configure(logger_path, flush_secs=5)

    collections = {'train': trainCollection, 'val': valCollection}

    cap_files = {x: os.path.join(rootpath, collections[x], 'TextData', '%s.caption.txt'%collections[x]) for x in collections}

    vis_feat_files = {x: BigFile(os.path.join(rootpath, collections[x], 'FeatureData', config.vid_feat)) for x in collections}

    config.txt_fc_layers = map(int, config.txt_fc_layers.split('-'))
    config.vis_fc_layers = map(int, config.vis_fc_layers.split('-'))
    if config.vis_fc_layers[0] == 0:
        config.vis_fc_layers[0] = vis_feat_files['train'].ndims
    else:
        assert config.vis_fc_layers[0] == vis_feat_files['train'].ndims, \
                'visual dimension not consistent(%s != %s)' % (config.vis_fc_layers[0], vis_feat_files['train'].ndims)

    w2vec = None
    bowvec = None
    idxvec = None
    if '@' in config.text_encoding:
        bow_encoding, w2v_encoding, rnn_encoding = config.text_encoding.split('@')
        rnn_encoding, config.pooling = rnn_encoding.split('_', 1)

        # Load Vocabulary Wrapper
        bow_vocab_file = os.path.join(rootpath, trainCollection, 'TextData', 'vocab', '%s_%d.pkl'%(bow_encoding, config.threshold))
        bowvec = get_txt2vec(bow_encoding)(bow_vocab_file, norm=config.bow_norm)

        w2v_data_path = os.path.join(rootpath, "word2vec", 'flickr', 'vec500flickr30m')
        w2vec = get_txt2vec(w2v_encoding)(w2v_data_path)

        rnn_vocab_file = os.path.join(rootpath, trainCollection, 'TextData', 'vocab', '%s_%d.pkl'%(rnn_encoding, config.threshold))
        idxvec = get_txt2vec('idxvec')(rnn_vocab_file)
        config.vocab_size = len(idxvec.vocab)
        if config.we_dim == 500:
            config.we = get_we(idxvec.vocab, w2v_data_path)

        if config.pooling == 'mean_last':
            config.txt_fc_layers[0] = config.rnn_size*2 + w2vec.ndims + bowvec.ndims
        else:
            config.txt_fc_layers[0] = config.rnn_size + w2vec.ndims + bowvec.ndims

    elif config.text_encoding in ['bow', 'bow_nsw']:
        bow_vocab_file = os.path.join(rootpath, trainCollection, 'TextData', 'vocab', '%s_%d.pkl'%(config.text_encoding, config.threshold))
        bowvec = get_txt2vec(config.text_encoding)(bow_vocab_file, norm=config.bow_norm)
        config.txt_fc_layers[0] = bowvec.ndims
    else:
        raise NotImplementedError('%s not implemented'%config.text_encoding)


    # Construct the model
    model = get_model('w2vvpp')(config)
    print(model.vis_net, model.txt_net)

    # Load data loaders
    data_loaders = {x: data.pair_provider({'vis_feat':vis_feat_files[x], 'cap':cap_files[x],
                                           'bow':bowvec, 'w2v':w2vec, 'rnn':idxvec,
                                           'batch_size':opt.batch_size, 'num_workers':opt.workers,'shuffle':(x=='train')})
                    for x in collections}

    # Train the Model
    best_perf = 0
    best_mir = 0
    best_miri = 0
    no_impr_counter = 0
    lr_counter = 0
    fout_val_perf_hist = open(os.path.join(logger_path, 'val_perf_hist.txt'), 'w')
    for epoch in range(opt.num_epochs):

        print('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, get_learning_rate(model.optimizer)[0]))
        print('-'*10)
        # train for one epoch
        train(opt, data_loaders['train'], model, epoch)

        rsum = 0
        # evaluate on validation set
        mir, miri = validate(opt, data_loaders['val'], model, measure=config.measure)
        perf = miri

        print(' * Current perf: {}'.format(rsum))
        print(' * Best perf: {}'.format(best_perf))
        print('')
        fout_val_perf_hist.write('epoch_%d:\nImage2Text: %f\nText2Image: %f\n' % (epoch, mir, miri))
        fout_val_perf_hist.flush()

        # remember best R@ sum and save checkpoint
        is_best = perf > best_perf
        if is_best:
            best_perf = perf
            best_mir = mir
            best_miri = miri
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_perf': best_perf,
            'opt': opt,
            'bow': bowvec,
            'w2v': w2vec,
            'rnn': idxvec,
        }, is_best, filename='checkpoint_epoch_%s.pth.tar'%epoch, prefix=logger_path + '/')

        lr_counter += 1
        decay_learning_rate(opt, model.optimizer, config.lr_decay_rate)

        if not is_best:
            # Early stop occurs if the validation performance
            # does not improve in ten consecutive epochs
            no_impr_counter += 1
            if no_impr_counter > 10:
                print('Early stopping happended.\n')
                break
            # When the validation performance decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            if lr_counter > 2:
                decay_learning_rate(opt, model.optimizer, 0.5)
                lr_counter = 0
        else:
            no_impr_counter = 0

    print(opt.logger_name)
    fout_val_perf_hist.close()
    print('best performance on validation:')
    print("Image to text: {}".format(best_mir))
    print("Text to image: {}".format(best_miri))
    print("")
    with open(os.path.join(opt.logger_name, 'val_perf.txt'), 'w') as fout:
        fout.write('best performance on validation:')
        fout.write('\nImage to text: {}'.format(best_mir))
        fout.write('\nText to image: {}'.format(best_miri))


def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    train_logger = util.LogCollector()

    # switch to train mode
    model.switch_to_train()

    progbar = Progbar(len(train_loader.dataset))
    end = time.time()
    for i, train_data in enumerate(train_loader):

        data_time.update(time.time() - end)
        model.logger = train_logger

        vis_inputs, txt_inputs, _, _, _  = train_data

        loss = model.train(vis_inputs, txt_inputs)

        progbar.add(vis_inputs.size(0), values=[('loss', loss)])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=i)
        tb_logger.log_value('step', i, step=i)
        tb_logger.log_value('batch_time', batch_time.val, step=i)
        tb_logger.log_value('data_time', data_time.val, step=i)


def validate(opt, val_loader, model, measure='cosine'):
    # compute the encoding for all the validation images and captions
    vis_embs, txt_embs, _, _ = encode_data(model, val_loader)

    # caption retrieval
    (r1, r5, r10, medr, meanr, mir) = eval_v2t(vis_embs, txt_embs, measure=measure)
    print(" * Image to text:")
    print(" * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)]))
    print(" * medr, meanr, mir: {}".format([round(medr, 3), round(meanr, 3), round(mir, 3)]))
    print(" * "+'-'*10)

    # image retrieval
    (r1i, r5i, r10i, medri, meanri, miri) = eval_v2t(txt_embs, vis_embs, measure=measure)
    print(" * Text to image:")
    print(" * r_1_5_10: {}".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)]))
    print(" * medr, meanr, mir: {}".format([round(medri, 3), round(meanri, 3), round(miri, 3)]))
    print(" * "+'-'*10)

    # currscore = miri

    # # record metrics in tensorboard
    # tb_logger.log_value('r1', r1, step=model.Eiters)
    # tb_logger.log_value('r5', r5, step=model.Eiters)
    # tb_logger.log_value('r10', r10, step=model.Eiters)
    # tb_logger.log_value('medr', medr, step=model.Eiters)
    # tb_logger.log_value('meanr', meanr, step=model.Eiters)
    # tb_logger.log_value('mir', mir, step=model.Eiters)
    # tb_logger.log_value('r1i', r1i, step=model.Eiters)
    # tb_logger.log_value('r5i', r5i, step=model.Eiters)
    # tb_logger.log_value('r10i', r10i, step=model.Eiters)
    # tb_logger.log_value('medri', medri, step=model.Eiters)
    # tb_logger.log_value('meanri', meanri, step=model.Eiters)
    # tb_logger.log_value('miri', miri, step=model.Eiters)
    # tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return mir, miri


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    resfile = os.path.join(prefix, filename)
    torch.save(state, resfile)
    if is_best:
        shutil.copyfile(resfile, os.path.join(prefix, 'model_best.pth.tar'))


def adjust_learning_rate(opt, learning_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def decay_learning_rate(opt, optimizer, decay):
    """decay learning rate to the last LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay

def get_learning_rate(optimizer):
    """Return learning rate"""
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


if __name__ == '__main__':
    main()
