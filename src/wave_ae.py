from utils.lsf_utils import *
import tensorflow as tf
import soundfile as sf
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

class AE_Model():
    def __init__(self,dropout=0.1,shape=[200,100,10],input_size=266,activation=tf.nn.relu,strides=1,kernel_size = [12],filters =[4],pool_size=[4],pooltype='max',fromhidden=False ):

        print('hello I am fully connected utoencoder model')
        self.input_size = input_size
        self.input = tf.placeholder(shape=[None,self.input_size],dtype=tf.float32,name='input')
        self.hiddeninput = tf.placeholder(shape=[None,shape[-1]],dtype=tf.float32,name='hidden_input')

        hidden1 = tf.layers.dense(self.input,units=shape[0],activation=activation)
        hidden1 = tf.layers.dropout(hidden1,rate=dropout)
        hidden2 = tf.layers.dense(hidden1,units=shape[1],activation=activation)
        hidden2 = tf.layers.dropout(hidden2,rate=dropout)
        hidden3 = tf.layers.dense(hidden2,units=shape[2],activation=tf.nn.tanh)
        self.encoded = hidden3
        if fromhidden:
            hidden3 = self.hiddeninput
        hidden3 = tf.layers.dropout(hidden3, rate=dropout)
        decode1 = tf.layers.dense(hidden3,units=shape[-2],activation=activation)
        decode1 = tf.layers.dropout(decode1,rate=dropout)
        decode2 = tf.layers.dense(decode1,units=shape[-3],activation=activation)
        decode2 = tf.layers.dropout(decode2,rate=dropout)
        decode3 = tf.layers.dense(decode2,units=self.input_size,activation=None)

        self.decode = decode3

        self.loss = tf.losses.mean_squared_error(labels=self.input, predictions=self.decode)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)


def main(outdir = '../out/dense_ae_200_100_10/'):

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    trainegg = trainegg.tolist()

    size = 16000 * 1.0 / 60
    trainegg, testegg, _, _ = loadeggsong('../out/',size)
    model = AE_Model(cnn=False)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_size = 512
        for i in range(100000):
            batch = random.sample(trainegg, batch_size)
            sess.run(model.optimizer, feed_dict={model.input: batch})
            if (i + 1) % 1000 == 0:
                trainloss = sess.run(model.loss, feed_dict={model.input: batch})
                testloss = sess.run(model.loss, feed_dict={model.input: testegg})
                print("%d,trainloss %.6f, testloss %.6f" % (i + 1, trainloss, testloss))
                pred = sess.run(model.decode, feed_dict={model.input: testegg})
                encoded = sess.run(model.encoded, feed_dict={model.input: data})
                f = open(outdir + 'hidden_%d.pkl' % (i + 1), 'wb')
                pickle.dump(encoded, f)
                sf.write(outdir + 'ae_test_%d.flac' % (i + 1), pred.flatten(), 16000)
                plt.plot(range(len(test.flatten())), test.flatten(), label='origin', linewidth=0.01)
                plt.plot(range(len(pred.flatten())), pred.flatten(), label='pred', linewidth=0.01)
                saver = tf.train.Saver()
                saver.save(sess, outdir + 'model_%d.cpkt' % (i + 1))
                plt.legend()
                plt.savefig(outdir + 'ae_test_%d.png' % (i + 1))
                plt.close()

if __name__  == "__main__":
    outdir = '../out/dense_ae_200_100_10/'
    main(outdir)