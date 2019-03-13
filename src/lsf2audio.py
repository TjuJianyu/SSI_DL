from utils.lsf_utils import *
import tensorflow as tf
import soundfile as sf
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

class lsf_Model():
    def __init__(self,dropout=0.1,shape=[50,100],input_size=12,output_size=266,activation=tf.nn.relu ):

        print('hello I am fully connected lsf 2 audio model')
        self.input_size = input_size
        self.output_size = output_size
        self.input = tf.placeholder(shape=[None,self.input_size],dtype=tf.float32,name='input')
        self.y = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32,name='output')

        decode1 = tf.layers.dense(self.input,units=shape[0],activation=activation)
        #decode1 = tf.layers.dropout(decode1,rate=dropout)
        decode2 = tf.layers.dense(decode1,units=shape[1],activation=activation)
        #decode2 = tf.layers.dropout(decode2,rate=dropout)
        decode3 = tf.layers.dense(decode2,units=self.output_size,activation=None)

        self.decode = decode3

        self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.decode)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

def lsf_main(outdir='../out/lsf2audio_50_100'):
    f = open('../out/test_lsf/lsf_hamming_16kHZ.pkl','rb')
    lsf = pickle.load(f)[0]
    train_lsf = np.concatenate([lsf[:25000],lsf[30000:]])
    test_lsf = np.array(lsf[25000:30000])
    size = 16000 * 1.0 / 60
    _,_, train, test = loadeggsong('../out/',size)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    model = lsf_Model(cnn=False)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_size=512

        for i in range(100000):
            batchid = np.random.randint(0,len(train),batch_size)
            batch_x = train_lsf[batchid]
            batch_y = train[batchid]

            sess.run(model.optimizer,feed_dict={model.input:batch_x,model.y:batch_y})
            if (i+1) % 1000 == 0:
                trainloss = sess.run(model.loss,feed_dict={model.input:batch_x,model.y:batch_y})
                testloss = sess.run(model.loss,feed_dict={model.input:test_lsf,model.y:test})
                print("%d,trainloss %.6f, testloss %.6f" % (i+1,trainloss,testloss))
                pred = sess.run(model.decode,feed_dict={model.input:test_lsf})

                sf.write(outdir+'lsf_reconstruct_test_%d.flac' % (i+1),pred.flatten(),16000)
                #pred = sess.run(model.decode,feed_dict={model.input:[batch[0]]})[0]
                plt.plot(range(len(test.flatten())),test.flatten(),label='origin',linewidth=0.1)
                plt.plot(range(len(pred.flatten())), pred.flatten(), label='pred', linewidth=0.1)
                saver = tf.train.Saver()
                saver.save(sess,outdir+'model_%d.cpkt' % (i+1))
                plt.legend()
                plt.savefig(outdir+'ae_test_%d.png' % (i+1))
                plt.close()

if __name__ == "__main__":
    outdir = '../out/lsf2audio_50_100'
    lsf_main(outdir)


