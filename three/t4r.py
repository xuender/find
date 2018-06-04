#! usr/bin/python
# coding=utf-8
# 训练结果应用,预测是否大于80
from keras.models import load_model
import numpy
model = load_model('h5/2-9.h5')

for i in range(1,100):
    p = model.predict([i])
    print('预测%d=%f'%(i, p[0]))
