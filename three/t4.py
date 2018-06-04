#! usr/bin/python
# coding=utf-8
# 判断是否都是3的倍数
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import numpy


def data():
    # 测试数据
    dataset = numpy.loadtxt("t3s.csv", delimiter=",")
    X = dataset[:, 0:1]
    Y = dataset[:, 1]


RET = []


def run(oneSize, twoSize):
    # 初始化随机数
    numpy.random.seed(7)
    # create model
    model = Sequential()
    model.add(Dense(oneSize, input_dim=1,
                    kernel_initializer='uniform', activation='relu'))
    model.add(Dense(twoSize, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # 日志图表
    tb = TensorBoard(log_dir='./logs/'+str(oneSize)+'/'+str(twoSize), histogram_freq=0,
                     write_graph=True, write_images=True)  # 在当前目录新建logs文件夹，记录 evens.out
    # 训练
    model.fit(X, Y, epochs=100, batch_size=10, verbose=0, callbacks=[tb])
    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # 测试
    testset = numpy.loadtxt("t3l.csv", delimiter=",")
    score = model.evaluate(testset[:, 0:1], testset[:, 1], verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # 保存模型
    model.save('h5/%d-%d.h5' % (oneSize, twoSize))
    del model
    RET.append([oneSize, twoSize, scores[1]*100])


data()
for i in range(1, 10):
    for f in range(1, 10):
        run(i, f)
print RET
