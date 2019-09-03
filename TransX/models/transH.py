#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/3 15:30
# @author: zhangcw
# @content: transH model

import torch
import torch.nn as nn


class transH(nn.Module):
    def __init__(self, config):
        super(transH, self).__init__()
        self.config = config
        self.batch_h = None
        self.batch_t = None
        self.batch_r = None

        self.ent_embeddings = nn.Embedding(self.config.ent_num, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.rel_num, self.config.hidden_size)
        self.norm_vector = nn.Embedding(self.config.rel_num, self.config.hidden_size)
        self.criterion = nn.MarginRankingLoss(self.config.margin, reduction='sum')
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)

    def get_positive_score(self, score):
        # score of positive triples
        return score[0:self.config.batch_size]

    def get_negative_score(self, score):
        # score of negative triples
        negative_score = score[self.config.batch_size:self.config.batch_seq_size]
        negative_score = negative_score.view(-1, self.config.batch_size)
        negative_score = torch.mean(negative_score, 0)
        return negative_score

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, self.config.p_norm, -1)

    def _transfer(self, e, norm):
        norm = torch.nn.functional.normalize(norm, p=2, dim=-1)
        return e - torch.sum(e * norm, -1, True) * norm

    def loss(self, p_score, n_score):
        if self.config.use_gpu:
            y = torch.Tensor([-1]).cuda()
        else:
            y = torch.Tensor([-1])
        return self.criterion(p_score, n_score, y)

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.rel_embeddings(self.batch_r)
        r_norm = self.norm_vector(self.batch_r)
        h = self._transfer(h, r_norm)
        t = self._transfer(t, r_norm)
        score = self._calc(h, t, r)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        '''
        print("batches:")
        print("batch_h:",self.batch_h)
        print("batch_t:",self.batch_t)
        print("batch_r:",self.batch_r)
        print("Embedding Vectors:")
        print("embedding_h:",h)
        print("embedding_t:",t)
        print("embedding_r:",r)
        print("score")
        print(score)
        print("p_score:\n",p_score)
        print("n_score:\n",n_score)
        print(self.loss(p_score, n_score))
        '''
        return self.loss(p_score, n_score)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()
