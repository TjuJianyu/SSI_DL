from utils.lsf_utils import *
import tensorflow as tf
import soundfile as sf
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

class egg_Model():
    def __init__(self,cnn=True,dropout=0.1,shape=[200,100,50],output_size=266,input_size=266,activation=tf.nn.relu,
                 strides=1,kernel_size = [12,12,6],filters =[4,8,16],pool_size=[4,4,2],pooltype='max',fromhidden=False,optimize=True ):

        print('hello I can be 1d CNN / fully connected model from egg wave / sine wave / any wave to audio.')
        print("I can become better by parameters tuning, as Jianyu almost did not tune parameters.")
        self.input_size = input_size
        self.output_size = output_size
        self.input = tf.placeholder(shape=[None,self.input_size],dtype=tf.float32,name='input')
        self.output = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32,name='output')
        if not cnn:
            hidden1 = tf.layers.dense(self.input,units=shape[0],activation=activation)
            hidden1 = tf.layers.dropout(hidden1,rate=dropout)
            hidden2 = tf.layers.dense(hidden1,units=shape[1],activation=activation)
            hidden2 = tf.layers.dropout(hidden2,rate=dropout)
            hidden3 = tf.layers.dense(hidden2,units=shape[2],activation=tf.nn.tanh)
            self.encoded = hidden3
            hidden3 = tf.layers.dropout(hidden3, rate=dropout)
            decode1 = tf.layers.dense(hidden3,units=shape[-2],activation=activation)
            decode1 = tf.layers.dropout(decode1,rate=dropout)
            decode2 = tf.layers.dense(decode1,units=shape[-3],activation=activation)
            decode2 = tf.layers.dropout(decode2,rate=dropout)
            decode3 = tf.layers.dense(decode2,units=self.input_size,activation=None)

            self.decode = decode3

        else:
            self.input = tf.placeholder(shape=[None, self.input_size,1], dtype=tf.float32, name='input')
            self.output = tf.placeholder(shape=[None, self.output_size,1], dtype=tf.float32, name='output')
            hidden1 = tf.layers.conv1d(self.input, filters=filters[0], kernel_size=kernel_size[0], strides=strides,
                             padding='same', activation=activation, use_bias=True)
            hidden1 = tf.layers.max_pooling1d(hidden1,pool_size=pool_size[0],strides=strides*2,padding='same')
            hidden1 = tf.layers.dropout(hidden1, rate=dropout)

            hidden2 = tf.layers.conv1d(hidden1, filters=filters[1], kernel_size=kernel_size[1], strides=strides,
                                       padding='same', activation=activation, use_bias=True)
            hidden2 = tf.layers.max_pooling1d(hidden2, pool_size=pool_size[1], strides=strides*2, padding='same')
            hidden2 = tf.layers.dropout(hidden2, rate=dropout)

            hidden3 = tf.layers.conv1d(hidden2, filters=filters[2], kernel_size=kernel_size[2], strides=strides,
                                       padding='same', activation=activation, use_bias=True)
            hidden3 = tf.layers.max_pooling1d(hidden3, pool_size=pool_size[2], strides=strides*2, padding='same')
            hidden3 = tf.layers.dropout(hidden3, rate=dropout)

            decode1 = tf.layers.conv1d(hidden3, filters=filters[2], kernel_size=kernel_size[2], strides=strides,
                                       padding='same', activation=activation, use_bias=True)
            shape = decode1.shape[1]

            decode1 = tf.reshape(decode1,[-1,1,shape,decode1.shape[-1]])
            decode1 = tf.image.resize_bilinear(decode1, (1,60))
            decode1 = tf.reshape(decode1, [-1, 60, decode1.shape[-1]])
            decode1 = tf.layers.dropout(decode1, rate=dropout)

            decode2 = tf.layers.conv1d(decode1, filters=filters[1], kernel_size=kernel_size[1], strides=strides,
                                       padding='same', activation=activation, use_bias=True)

            decode2 = tf.reshape(decode2, [-1, 1, decode2.shape[1], decode2.shape[-1]])
            decode2 = tf.image.resize_bilinear(decode2, (1,120))
            decode2 = tf.reshape(decode2,[-1,120,decode2.shape[-1]])
            decode2 = tf.layers.dropout(decode2, rate=dropout)

            decode3 = tf.layers.conv1d(decode2, filters=filters[0], kernel_size=kernel_size[0], strides=strides,
                                       padding='same', activation=activation, use_bias=True)

            decode3 = tf.reshape(decode3,[-1,1,decode3.shape[1],decode3.shape[-1]])
            decode3 = tf.image.resize_bilinear(decode3, (1,266))
            decode3 = tf.reshape(decode3,[-1,266,decode3.shape[-1]])
            decode3 = tf.layers.dropout(decode3, rate=dropout)

            self.decode = tf.layers.conv1d(decode3,filters=1,kernel_size=3, strides=1,padding='same',activation=None)

            #print(self.decode.shape)

        self.loss = tf.losses.mean_squared_error(labels=self.output, predictions=self.decode)
        if optimize:
            self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

def main():
    size = int(16000 * 1.0/60)
    outdir = '../../out/cnn_ae_songfeat/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    _, _, trainy, testy = loadeggsong('../../out/')

    data_dir = '../../out/'

    f = open(data_dir+'trainsongfeat_fit.pkl','rb')
    trainegg = pickle.load(f)
    f.close()

    f = open(data_dir+'testsongfeat_fit.pkl','rb')
    testegg = pickle.load(f)
    f.close()


    train = []
    for val in trainegg:
        train.append(test_func(range(size),val[0],val[1],val[2],val[3]))
    test = []
    for val in testegg:
        test.append(test_func(range(size),val[0], val[1], val[2], val[3]))
    train = np.array(train)
    test = np.array(test)

    cnn =True
    if cnn:
        train = train[:,:,np.newaxis]
        test = test[:, :, np.newaxis]
        trainy = trainy[:, :, np.newaxis]
        testy = testy[:, :, np.newaxis]


    model = egg_Model(cnn=True,dropout=0)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_size=512

        for i in range(1000000):

            batchid = np.random.randint(0, len(train), batch_size)
            #print(batchid)
            batch_x = train[batchid]
            batch_y = trainy[batchid]
            #print(batch_y.shape)
            #print(batch_x.shape)
            sess.run(model.optimizer,feed_dict={model.input:batch_x,model.output:batch_y})
            if (i+1) % 1000 == 0:
                trainloss = sess.run(model.loss,feed_dict={model.input:batch_x,model.output:batch_y})
                testloss = sess.run(model.loss,feed_dict={model.input:test,model.output:testy})
                print("%d,trainloss %.6f, testloss %.6f" % (i+1,trainloss,testloss))
                pred = sess.run(model.decode,feed_dict={model.input:test})


                print(pred.max(),pred.min())
                #f = open(outdir+'ae_test_%d.pkl' % (i+1),'wb')
                #pickle.dump(pred,f)
                #f.close()
                sf.write(outdir+'ae_test_%d.wav' % (i+1),pred.flatten(),16000,subtype='FLOAT')

                plt.plot(range(len(test.flatten())),test.flatten(),label='origin',linewidth=0.01)
                plt.plot(range(len(pred.flatten())), pred.flatten(), label='pred', linewidth=0.01)
                saver = tf.train.Saver()
                saver.save(sess,outdir+'model_%d.cpkt' % (i+1))
                plt.legend()
                plt.savefig(outdir+'ae_test_%d.png' % (i+1))
                plt.close()
                #plt.show()


if __name__ == "__main__":
