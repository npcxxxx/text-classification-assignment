import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from util import *
from config import Config

class CNN(nn.Module):
    def __init__(self,
                 class_num=200,
                 embed_dim=300,
                 kernel_num=256,
                 kernel_size_list=(3, 4, 5),
                 dropout=0.3):
        super(CNN, self).__init__()

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(embed_dim, kernel_num, kernel_size)
            for kernel_size in kernel_size_list
        ])

        self.linear = nn.Linear(kernel_num * len(kernel_size_list), class_num)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x的形状： (batch, word_nums)
        # x在embedding后的形状： (batch, word_nums, embed_dim)
        # conv1d参数要求: (batch, in_channels, in_length)，因此需要转换维度
        x = x.transpose(1, 2)

        # x卷积后的形状: (batch, kernel_num, out_length)
        # out_length = word_nums - kernel_size + 1
        x = [F.relu(conv1d(x)) for conv1d in self.conv1d_list]

        # x池化后的形状: (batch, kernel_num, 1)
        # squeeze把最后这个维度消掉
        x = [F.max_pool1d(i, i.shape[2]).squeeze(2) for i in x]

        # x在拼接后的形状: (batch, kernel_num * len(kernel_size_list))
        x = torch.cat(x, dim=1)
        x = self.dropout(x)

        # x输出的形状: (batch, class_num)
        x = self.linear(x)
        return x

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        w = weight[i].unsqueeze(1)
        _s = torch.mm(seq[i], w)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return attn_vectors

class CRAN(nn.Module):
    def __init__(self, num_tokens, embed_size):
        super(CRAN, self).__init__()
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.lookup = nn.Embedding(num_tokens, embed_size)

        self.dropout = nn.Dropout(Config.dropout)
        self.batch_size = Config.batch_size
        self.word_gru_hidden = Config.word_gru_hidden

        self.word_gru = nn.GRU(embed_size, self.word_gru_hidden, bidirectional=True)
        self.weight_W_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 2 * self.word_gru_hidden))
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.bias_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 1))

        self.fc = nn.Linear(num_tokens * 2 * self.word_gru_hidden, Config.output_size)

        self.softmax = nn.Softmax()

        self.cnn = CNN(class_num=2 * Config.word_gru_hidden, embed_dim=embed_size, kernel_num=Config.kernel_num, kernel_size_list=Config.kernel_size_list, dropout=Config.dropout)

    def forward(self, embed, state_word):
        # embedding
        embedded = self.lookup(embed)

        # CNN层
        weight_proj_word = self.cnn(embedded)

        # RNN层
        output_word, state_word = self.word_gru(embedded, state_word)
        output_word = self.dropout(output_word)

        # ATTENTION层
        word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, weight_proj_word)
        word_attn_norm = self.softmax(word_attn.transpose(1, 0))
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1, 0))

        # 输出层
        final_feature = torch.cat([word_attn_vectors[:, i, :] for i in range(word_attn_vectors.shape[1])], dim=1)
        final_out = self.fc(final_feature)
        return self.softmax(final_out)

    def init_hidden(self):
        return Variable(torch.zeros(2, self.num_tokens, self.word_gru_hidden))

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # 随迭代次数增加，缩小学习率
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # 验证集评估准确率
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

        return train_losses, val_accuracies


if __name__ == '__main__':
    import os
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    x = torch.LongTensor(64, 500).random_(0, 10)
    config = Config()
    model = CRAN(500, 300)
    state_word = model.init_hidden()
    output = model(x, state_word)
    print('output: ', output.size())









