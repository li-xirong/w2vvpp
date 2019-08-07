from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn

from torch.nn.utils.clip_grad import clip_grad_norm  

from loss import ContrastiveLoss
from bigfile import BigFile

def get_we(vocab, w2v_dir):
    w2v = BigFile(w2v_dir)
    ndims = w2v.ndims
    nr_words = len(vocab)
    words = [vocab[i] for i in range(nr_words)]
    we = np.random.uniform(low=-1.0, high=1.0, size=(nr_words, ndims))

    renamed, vecs = w2v.read(words)
    for i, word in enumerate(renamed):
        idx = vocab.find(word)
        we[idx] = vecs[i]

    return torch.Tensor(we)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)


class IdentityNet(nn.Module):
    def __init__(self, opt):
        super().__init__()       

    def forward(self, input_x):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized        if self.img_fc:
        return input_x


class TransformNet(nn.Module):
    def __init__(self, fc_layers, opt):
        super().__init__()
        
        self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
        if opt.batch_norm:
            self.bn1 = nn.BatchNorm1d(fc_layers[1])
        else:
            self.bn1 = None
                   
        if opt.activation == 'tanh':
            self.activation = nn.Tanh()
        elif opt.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None

        if opt.dropout > 1e-3:
            self.dropout = nn.Dropout(p=opt.dropout)
        else:
            self.dropout = None

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        xavier_init_fc(self.fc1)
    

    def forward(self, input_x):
        """Extract image feature vectors."""
      
        features = self.fc1(input_x)
                  
        if self.bn1 is not None:
            features = self.bn1(features)
        
        if self.activation is not None:
            features = self.activation(features)
  
        if self.dropout is not None:
            features = self.dropout(features)
              
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super().load_state_dict(new_state)

class VisTransformNet (TransformNet):
    def __init__(self, opt):
        super().__init__(opt.img_fc_layers, opt)
    
class TxtTransformNet (TransformNet):
    def __init__(self, opt):
        super().__init__(opt.txt_fc_layers, opt)
    
    
class TxtEncoder(nn.Module):
    def __init__(self, opt):
        super(TxtEncoder, self).__init__()       

    def forward(self, txt_input):
        return txt_input
        
        
class GruTxtEncoder(TxtEncoder):
    def __init_rnn(self, opt):
        self.rnn = nn.GRU(opt.we_dim, opt.rnn_size, opt.rnn_num_layer, batch_first=True)
    
    def __init__(self, opt):
        super().__init__()
        self.pooling = opt.pooling  
        self.rnn_size = opt.rnn_size
        self.we = nn.Embedding(opt.vocab_size, opt.we_dim)
        if opt.we_sim == 500:
            self.we.weight = nn.Parameter(opt.we) # initialize with a pre-trained 500-dim w2v

        self.__init_rnn(opt)  
 

    def forward(self, txt_input):
        """Handles variable size captions
        """
        x, lengths = txt_input
        batch_size = x.size(0)
        x = self.we(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
    
        if self.pooling == 'mean':
            out = torch.zeros(batch_size, self.rnn_size).cuda()
            for i, ln in enumerate(lengths):
                out[i] = torch.mean(padded[0][i][:ln], dim=0)
        elif self.pooling == 'last':
            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.cuda()
            out = torch.gather(padded[0], 1, I).squeeze(1)
        elif self.rnn_type == 'mean_last':
            out1 = torch.zeros(batch_size, self.rnn_size).cuda()
            for i, ln in enumerate(lengths):
                out1[i] = torch.mean(padded[0][i][:ln], dim=0)

            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.cuda()
            out2 = torch.gather(padded[0], 1, I).squeeze(1)
            out = torch.cat((out1, out2), dim=1)
        return out


class MultiScaleTxtEncoder (GruTxtEncoder):
    def __init__(self, opt):
        super().__init__(opt)

    def forward(self, txt_input):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x, lengths, cap_w2vs, cap_bows = txt_input
        rnn_out = super().forward(self, (x, lengths))
        out = torch.cat((rnn_out, cap_w2vs, cap_bows), dim=1)
        return out
  
class TxtNet (nn.Module):
    def __init_encoder(self, opt):
        self.encoder = TxtEncoder(opt)
        
    def __init_transformer(self, opt):
        self.transformer = TxtTransformNet(opt)
        
    def __init__(self, opt):
        self.__init_encoder(opt)
        self.__init_transformer(opt)
    
    def forward(self, txt_input):
        features = self.encoder(txt_input)
        features = self.transformer(features)              
        return features
   
class MultiScaleTxtNet (TxtNet):
    def __init_encoder(self, opt):
        self.encoder = MultiScaleTxtEncoder(opt)
    
class CrossModalNetwork(object):

    def __init_vis_net(self, opt):
        self.vis_net = VisNet(opt)

    def __init_txt_net(self, opt):
        self.txt_net = TxtNet(opt)
    
    def __init__(self, opt):
        self.__init_vis_net(opt)
        self.__init_txt_net(opt)
        
        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            self.vis_net.cuda()
            self.txt_net.cuda()
            cudnn.benchmark = True

        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation,
                                         cost_style=opt.cost_style,
                                         direction=opt.direction)
    
        params = list(self.vis_net.parameters())
        params += list(self.txt_net.parameters())
        self.params = params

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.lr)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.lr)

        self.iters = 0

    def state_dict(self):
        state_dict = [self.vis_net.state_dict(), self.txt_net.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vis_net.load_state_dict(state_dict[0])
        self.txt_net.load_state_dict(state_dict[1])

    def switch_to_train(self):
        self.vis_net.train()
        self.txt_net.train()

    def switch_to_eval(self):
        self.vis_net.eval()
        self.txt_net.eval()

    def compute_loss(self, vis_embs, txt_embs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(txt_embs, vis_embs)
        # pytorch 0.3.1
        #self.logger.update('Loss', loss.data[0], vis_emb.size(0)) 
        # pytorch 0.4.0
        self.logger.update('Loss', loss.item(), vis_embs.size(0))
        return loss

    def train(self, vis_input, txt_input):
        """One training step given images and captions.
        """
        self.iters += 1
        self.logger.update('it', self.iters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vis_embs = self.vis_net(vis_input)
        txt_embs = self.txt_net(txt_input)
        #['captions'], txt_input['lengths'])
        
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.compute_loss(vis_embs, txt_embs)

        # pytorch 0.3.1
        #loss_value = loss.data[0]
        
        # pytorch 0.4.0
        loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        return loss_value

'''
class W2VV (CrossModalNetwork):
    def __init_vis_net(self, opt):
        self.vis_net = IdentityNet(opt)

    def __init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet(opt)
'''

class W2VVPP (CrossModalNetwork):
    def __init_vis_net(self, opt):
        self.vis_net = VisTransformerNet(opt)

    def __init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet(opt)


NAME_TO_MODELS = {'w2vvpp': W2VVPP}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]

if __name__ == '__main__':
    model = get_model('w2vvpp')
    
