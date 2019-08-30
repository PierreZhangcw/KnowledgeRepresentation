#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import logging
import numba
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'
                    )

def to_var(x,use_gpu):
    if use_gpu:
        return torch.from_numpy(x).cuda()
    else:
        return torch.from_numpy(x)

class Config(object):
    def __init__(self):
        # data_path
        self.in_path = "data/FB15K/"
        # number of positive samples in one batch
        self.batch_size = 4
        # dimension of embedding
        self.hidden_size = 50  # self.dim = 100
        # negative samples / positive samples in one batch
        self.negative_ent = 1
        self.negative_rel = 0
        # entity and relation embedding size
        self.ent_size = self.hidden_size
        self.rel_size = self.hidden_size
        # margin hyper-parameter
        self.margin = 1.0
        # optimisation method
        self.opt_method = "SGD"
        self.optimizer = None
        # learning rate decay
        self.lr_decay = 0
        # weight decay
        self.weight_decay = 0
        self.lmbda = 0.0
        # learninig rate
        self.alpha = 0.005 #0.01

        # self.early_stopping_patience = 10
        # num of batchs
        self.nbatches = 100
        self.p_norm = 1
        self.use_gpu = False
        # number of epochs
        self.train_times = 100

    def init(self):
        self.load_data()
        self.batch_size = int(self.trainTotal / self.nbatches)
        self.batch_seq_size = self.batch_size * (
                1 + self.negative_ent + self.negative_rel
        )
        self.loss_record = []
        #self.nbatches = int(self.trainTotal/self.batch_size)

    def data_parse(self,label):
        if label == 'train':
            path = self.in_path + "train2id.txt"
        elif label == 'test':
            path = self.in_path + "test2id.txt"
        else:
            path = self.in_path + "valid2id.txt"
        with open(path,'r') as f:
            lines = f.readlines()
        total = int(lines[0].strip())
        data = np.array([list(map(lambda x:int(x),line.strip().split())) for line in lines[1:]],dtype=np.int64)
        return total,data

    def load_data(self):
        # load_train
        self.trainTotal,self.train_data = self.data_parse('train')
        # load_test
        self.testTotal,self.test_data = self.data_parse('test')
        # load_valid
        self.validTotal,self.valid_data = self.data_parse('valid')
        # number of entities
        with open(self.in_path+"entity2id.txt",'r') as f:
            line = f.readline()
            self.ent_num = int(line.strip())
        # number of relations
        with open(self.in_path + "relation2id.txt", 'r') as f:
            line = f.readline()
            self.rel_num = int(line.strip())


    def set_train_model(self, model):
        logging.info("Initializing training model...")
        self.model = model
        self.trainModel = self.model(config=self)
        if self.use_gpu:
            self.trainModel.cuda()
        if self.optimizer != None:
            pass
        else:
            self.optimizer = optim.SGD(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        logging.info("Finish initializing")

    def sampling(self,batch_index):
        # negative sample
        batch_h = []
        batch_h_ = []
        batch_r = []
        batch_t = []
        batch_t_ = []
        for index in batch_index:
            h,t,r = self.train_data[index]
            batch_h.append(h)
            batch_r.append(r)
            batch_t.append(t)
            ent = np.random.randint(self.ent_num)
            choice = np.random.randint(1,3)
            if choice==1:
                h_ = ent
                t_ = t
                if ent==h:
                    h_+=1
                if h_>=self.ent_num:
                    h_-=2
            else:
                h_ = h
                t_ = ent
                if ent==t:
                    t_+=1
                if t_>=self.ent_num:
                    t_-=2
            batch_h_.append(h_)
            batch_t_.append(t_)
        self.batch_h = np.array(batch_h+batch_h_,dtype=np.int64)
        self.batch_r = np.array(batch_r+batch_r,dtype=np.int64)
        self.batch_t = np.array(batch_t+batch_t_,dtype=np.int64)

    def train_one_step(self):
        # train for one batch
        self.trainModel.batch_h = to_var(self.batch_h, self.use_gpu)
        self.trainModel.batch_t = to_var(self.batch_t, self.use_gpu)
        self.trainModel.batch_r = to_var(self.batch_r, self.use_gpu)
        # self.trainModel.batch_y = to_var(self.batch_y, self.use_gpu)
        self.optimizer.zero_grad()
        loss = self.trainModel()
        loss.backward()
        self.optimizer.step()
        #print("loss_item:",loss.item())
        return loss.item()

    @numba.jit()
    def train(self):
        logging.info("Start training...")
        for epoch in tqdm(range(self.train_times)):
        #for epoch in range(self.train_times):
            #logging.info("###shuffling train data......###")
            train_data_index = list(range(self.trainTotal))
            np.random.shuffle(train_data_index)
            res = 0.0
            #logging.info("###Start epoch %d###"%(epoch,))
            for batch in range(self.nbatches):
                #logging.info("sampling......")
                start = batch*self.batch_size
                end = start + self.batch_size
                batch_index = train_data_index[start:end]
                self.sampling(batch_index)
                #logging.info("training one step")
                loss = self.train_one_step()
                res += loss
            self.loss_record.append(res)
            # if epoch%10==0:
                # logging.info("Epoch %d | loss: %f" % (epoch, res))
    
    @numba.jit()
    def test(self):
        # self.model.batch_ent = torch.from_numpy(np.int64(range(self.ent_num)))
        logging.info("Total triples for testing:%d"%(self.testTotal,))
        self.trainModel.batch_t = to_var(np.int64(range(self.ent_num)),self.use_gpu)
        #self.trainModel.batch_t = self.batch_ent
        hit10_tail = 0
        for i in tqdm(range(self.testTotal)):
            h,t,r = self.test_data[i]
            self.trainModel.batch_h = to_var(np.array([np.int64(h)]),self.use_gpu)
            self.trainModel.batch_r = to_var(np.array([np.int64(r)]),self.use_gpu)
            results = self.trainModel.predict()
            top10 = results.argsort()[:10]
            if t in top10:
                hit10_tail += 1
        hit10_head = 0
        self.trainModel.batch_h = to_var(np.int64(range(self.ent_num)),self.use_gpu)
        for i in tqdm(range(self.testTotal)):
            h,t,r = self.test_data[i]
            self.trainModel.batch_t = to_var(np.array([np.int64(t)]),self.use_gpu)
            self.trainModel.batch_r = to_var(np.array([np.int64(r)]),self.use_gpu)
            results = self.trainModel.predict()
            top10 = results.argsort()[:10]
            if h in top10:
                hit10_head += 1

        logging.info("num of head@HIT10:%d"%(hit10_head,))
        logging.info("num of tail@HIT10:%d"%(hit10_tail,))
        #logging.info("@HIT10：%d"%(hit10/self.testTotal,))

    def write_loss(self):
        import json
        with open("loss/loss.json",'w') as f:
            json.dump(self.loss_record,f,indent=1)







