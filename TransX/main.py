#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
# @date: 2019/8/2 13:44
# @author: zhangcw
# @content: test the code of transX model

from models.transE import transE
from config.config import Config

model = Config()
model.set_train_model(transE)
model.train()
model.test()
