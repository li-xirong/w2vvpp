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
from vocab import Vocabulary  # NOQA
from text2vec import get_text_encoder
from model import get_model, get_we_parameter
from w2vv import W2VV
import evaluation
from evaluation import eval_i2t, AverageMeter, LogCollector, encode_data
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import checkToSkip
from utils.generic_utils import Progbar

INFO = __file__

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
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--finetune', action='store_true', 
                        help='Fine-tune the model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model_config', type=str, default='mean_pyresnext-101_rbps13k',
                        help='model configuration file. (default: mean_pyresnext-101_rbps13k')

    args = parser.parse_args()
    return args

def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']

def print_config(config):
    attributes = [attr for attr in dir(config) 
                      if not attr.startswith('__')]
    print(', '.join('%s: %s' % item for item in config.items() ))#(attr, config[attr]) for attr in attributes))

def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent = 2))

    rootpath = opt.rootpath
    trainCollection = opt.trainCollection
    valCollection = opt.valCollection

    model_config = opt.model_config
    if trainCollection == 'msrvtt10ktrain':
        config = load_config('msrvtt10k_configs/%s.py' % model_config)
    else:
        config = load_config('w2vv_configs/%s.py' % model_config)

    opt.logger_name = os.path.join(rootpath, trainCollection, 'w2vv++_train', valCollection, INFO,
                    model_config, opt.logger_name)
    print(opt.logger_name)

    if checkToSkip(os.path.join(opt.logger_name, 'model_best.pth.tar'), opt.overwrite) or \
         checkToSkip(os.path.join(opt.logger_name, 'val_perf.txt'), opt.overwrite):
        sys.exit(0)
        pass
    try:
        os.makedirs(opt.logger_name)
    except:
        pass

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    img_feature = config.img_feature
    bow_vocab = config.bow_vocab+'.pkl'
    rnn_vocab = config.rnn_vocab+'.pkl'

    # collections: trian, val
    collections = {'train': trainCollection, 'val': valCollection}
    # caption file path
    if trainCollection == 'msrvtt10ktrain':
        cap_file = {'train': '%s.caption.txt'%trainCollection,
                'val': '%s.caption.txt'%valCollection}
    else:
        cap_file = {'train': '%s.caption.txt'%trainCollection,
                'val': '%s/%s.caption.txt'%(opt.val_set, valCollection)}
    caption_files = { x: os.path.join(rootpath, collections[x], 'TextData', cap_file[x])
                        for x in collections }

    # Load Image features
    video2subvideo = None
    if 'sub' in img_feature:
        img_feat_path = { 'train': os.path.join(rootpath, collections['train'], 'FeatureData', img_feature),
              'val':   os.path.join(rootpath, collections['val'], 'FeatureData', img_feature.rsplit('-', 1)[0])
                          }
        video2subvideopath = os.path.join(img_feat_path['train'], 'video2subvideo.txt')
        with open(video2subvideopath) as reader:
            video2subvideo = eval(reader.read())
    else:
        img_feat_path = {x: os.path.join(rootpath, collections[x], 'FeatureData', img_feature)
                            for x in collections }
    img_feats = {x: BigFile(img_feat_path[x]) for x in img_feat_path}

    config.img_dim = img_feats['train'].ndims
    config.text_fc_layers = map(int, config.text_fc_layers.split('-'))
    config.img_fc_layers = map(int, config.img_fc_layers.split('-'))
    if config.img_fc_layers[0] == 0:
        config.img_fc_layers[0] = config.img_dim
    else:
        assert config.img_fc_layers[0] == config.img_dim, '%s != %s' % (config.img_fc_layers[0], config.img_dim)

    w2v2vec = None
    bow2vec = None

    w2v_data_path = os.path.join(rootpath, "word2vec", 'flickr', 'vec500flickr30m')

    if '@' in config.text_encode_style:
        rnn_style, bow_style, w2v_style = config.text_encode_style.split('@')

        # Load Vocabulary Wrapper
        rnn_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 
                                        rnn_style, rnn_vocab)
        rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
        config.vocab_size = len(rnn_vocab)

        bow_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary',
                                        bow_style, bow_vocab)
        bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))

        w2v2vec = get_text_encoder(w2v_style)(w2v_data_path)
        bow2vec = get_text_encoder(bow_style)(bow_vocab, L1_norm=(config.bow_norm=='l1'),
                                                         L2_norm=(config.bow_norm=='l2'))
        if config.rnn_type == 'mean_last':
            config.text_fc_layers[0] = config.rnn_size*2 + w2v2vec.ndims + bow2vec.ndims
        else:
            config.text_fc_layers[0] = config.rnn_size + w2v2vec.ndims + bow2vec.ndims
    elif config.text_encode_style == 'gru':
        rnn_style = config.text_encode_style
        # Load Vocabulary Wrapper
        rnn_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 
                                        rnn_style, rnn_vocab)
        rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
        config.vocab_size = len(rnn_vocab)
        if config.rnn_type == 'mean_last':
            config.text_fc_layers[0] = config.rnn_size*2
        else:
            config.text_fc_layers[0] = config.rnn_size
    elif config.text_encode_style in ['bow', 'bow_filterstop']:
        bow_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary',
                                        bow_style, bow_vocab)
        bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
        bow2vec = get_text_encoder(bow_style)(bow_vocab, L1_norm=(config.bow_norm=='l1'),
                                                         L2_norm=(config.bow_norm=='l2'))
        config.text_fc_layers[0] = bow2vec.ndims
    else:
        raise NotImplementedError('%s not implemented'%config.text_encode_style)


    config.we_parameter = get_we_parameter(rnn_vocab, w2v_data_path)

    # Load data loaders
    data_loaders = data.get_precomp_loaders(
            caption_files, img_feats, rnn_vocab, w2v2vec, bow2vec, opt.batch_size, opt.workers, None)
#                {'train':video2subvideo, 'val': None} )

    # Construct the model
    model = get_model(config.model)(config)  #VSE(config)
    #model = W2VV(config)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, data_loaders['val'], model, measure=config.measure)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    best_mir = 0
    best_miri = 0
    no_impr_counter = 0
    lr_counter = 0
    fout_val_perf_hist = open(os.path.join(opt.logger_name, 'val_perf_hist.txt'), 'w')
    assert config.lr_decay_method in ['origin', 'own']
    for epoch in range(opt.num_epochs):

        print('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, get_learning_rate(model.optimizer)[0]))
        print('-'*10)
        # train for one epoch
        train(opt, data_loaders['train'], model, epoch)

        rsum = 0
        # evaluate on validation set
        if opt.trainCollection == 'msrvtt10ktrain':
            mir, miri = validate_mc(opt, data_loaders['val'], model, measure=config.measure)
            rsum = mir + miri
        else:
            mir, miri = validate(opt, data_loaders['val'], model, measure=config.measure)
            rsum = miri

        print(' * Current perf: {}'.format(rsum))
        print(' * Best perf: {}'.format(best_rsum))
        print('')
        fout_val_perf_hist.write('epoch_%d:\nImage2Text: %f\nText2Image: %f\n' % (epoch, mir, miri))
        fout_val_perf_hist.flush()

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        if is_best:
            best_rsum = rsum
            best_mir = mir
            best_miri = miri
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_epoch_%s.pth.tar'%epoch, prefix=opt.logger_name + '/')

        if config.lr_decay_method == 'own':
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
        elif config.lr_decay_method == 'origin':
            adjust_learning_rate(opt, config.lr, model.optimizer, epoch+1)

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
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    progbar = Progbar(len(train_loader.dataset))
    end = time.time()
    for i, train_data in enumerate(train_loader):
        #if opt.reset_train:
        #    # Always reset to train mode, this is not the default behavior
        #    model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        b_size, loss = model.train_emb(epoch, *train_data)

        progbar.add(b_size, values=[('loss', loss)])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        #if model.Eiters % opt.log_step == 0:
        #    logging.info(
        #        'Epoch: [{0}][{1}/{2}]\t'
        #        '{e_log}\t'
        #        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #        .format(
        #            epoch, i, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        #if model.Eiters % opt.val_step == 0:
        #    validate(opt, val_loader, model)

def validate_mc(opt, val_loader, model, measure='cosine'):
    # compute the encoding for all the validation images and captions
    video_embs, cap_embs, video_ids, caption_ids = encode_data(
        model, val_loader, opt.log_step, logging.info)

    # we load data as video-sentence pairs
    # but we only need to forward each video once for evaluation
    # so we get the video set and mask out same videos with feature_mask
    feature_mask = []
    evaluate_videos = set()
    for video_id in video_ids:
        feature_mask.append(video_id not in evaluate_videos)
        evaluate_videos.add(video_id)
    video_embs = video_embs[feature_mask]
    video_ids = [x for idx, x in enumerate(video_ids) if feature_mask[idx] is True]

    c2i_all_errors = evaluation.cal_error(video_embs, cap_embs, measure)
    # caption retrieval
    r1, r5, r10, medr, meanr = evaluation.i2t(c2i_all_errors, n_caption=20)
    result = " * Image to text:"
    result += " * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)])
    result += " * medr, meanr: {}".format([round(medr, 3), round(meanr, 3)])
    result += " * "+'-'*10
    sum_r = r1 + r5 + r10

    # image retrieval
    r1i, r5i, r10i, medri, meanri = evaluation.t2i(c2i_all_errors, n_caption=20)
    result += " * Text to image:"
    result += " * r_1_5_10: {}".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)])
    result += " * medr, meanr, mir: {}".format([round(medri, 3), round(meanri, 3)])
    result += " * "+'-'*10
    sum_ri = r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('sum_r', sum_r, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('sum_ri', sum_ri, step=model.Eiters)
    tb_logger.log_value('rsum', sum_r + sum_ri, step=model.Eiters)

    return sum_r, sum_ri


def validate(opt, val_loader, model, measure='cosine'):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, _, _ = encode_data(
        model, val_loader, opt.log_step, logging.info)

    # caption retrieval
    (r1, r5, r10, medr, meanr, mir) = eval_i2t(img_embs, cap_embs, measure=measure)
    print(" * Image to text:")
    print(" * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)]))
    print(" * medr, meanr, mir: {}".format([round(medr, 3), round(meanr, 3), round(mir, 3)]))
    print(" * "+'-'*10)

    # image retrieval
    (r1i, r5i, r10i, medri, meanri, miri) = eval_i2t(cap_embs, img_embs, measure=measure)
    print(" * Text to image:")
    print(" * r_1_5_10: {}".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)]))
    print(" * medr, meanr, mir: {}".format([round(medri, 3), round(meanri, 3), round(miri, 3)]))
    print(" * "+'-'*10)
    #logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
    #             (r1i, r5i, r10i, medri, meanri, miri))
    # sum of recalls to be used for early stopping
    #currscore = r1 + r5 + r10 + r1i + r5i + r10i
    # currscore = mir + miri
    # currscore = mir
    currscore = miri

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('mir', mir, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('miri', miri, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    # return currscore
    return mir, miri


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
