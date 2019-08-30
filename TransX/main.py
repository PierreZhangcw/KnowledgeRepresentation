#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
from models.transE import transE
from config.config import Config

model = Config()
model.set_train_model(transE)
model.train()
model.test()
