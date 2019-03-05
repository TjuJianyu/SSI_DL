import tensorflow as tf
from lsf_utils import *
import soundfile as sf
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
class Model():
    def __init__(self,cnn=True,dropout=0.1,shape=[200,100,2],input_size=266,activation=tf.nn.relu,strides=1,kernel_size = [12],filters =[4],pool_size=[4],pooltype='max',fromhidden=False ):

        print('hello I am model')
        self.input_size = input_size
        self.input = tf.placeholder(shape=[None,self.input_size],dtype=tf.float32,name='input')
        self.hiddeninput = tf.placeholder(shape=[None,shape[-1]],dtype=tf.float32,name='hidden_input')
        if not cnn:
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

        else:

            hidden1 = tf.layers.conv1d(self.input,filters=filters[0],kernel_size=kernel_size[0],strides = strides, padding='valid',activation=activation,use_bias=True)


            if pooltype=='average':
                hidden1 = tf.layers.average-pooling1d(hidden1,pool_size=pool_size[0],strides=strides,padding='valid')
            else:
                hidden1 = tf.layers.max_pooling1d(hidden1,pool_size=pool_size[0],strides=strides,padding='valid')



        self.loss = tf.losses.mean_squared_error(labels=self.input, predictions=self.decode)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

class lsf_Model():
    def __init__(self,cnn=True,dropout=0.1,shape=[50,100],input_size=12,output_size=266,activation=tf.nn.relu,
                 strides=1,kernel_size = [12],filters =[4],pool_size=[4],pooltype='max',fromhidden=False ):

        print('hello I am model')
        self.input_size = input_size
        self.output_size = output_size
        self.input = tf.placeholder(shape=[None,self.input_size],dtype=tf.float32,name='input')
        self.y = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32,name='output')
        if not cnn:


            decode1 = tf.layers.dense(self.input,units=shape[0],activation=activation)
            #decode1 = tf.layers.dropout(decode1,rate=dropout)
            decode2 = tf.layers.dense(decode1,units=shape[1],activation=activation)
            #decode2 = tf.layers.dropout(decode2,rate=dropout)
            decode3 = tf.layers.dense(decode2,units=self.output_size,activation=None)

            self.decode = decode3

        else:
            pass


        self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.decode)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
class egg_Model():
    def __init__(self,cnn=True,dropout=0.1,shape=[200,100,50],output_size=266,input_size=266,activation=tf.nn.relu,
                 strides=1,kernel_size = [12,12,6],filters =[4,8,16],pool_size=[4,4,2],pooltype='max',fromhidden=False ):

        print('hello I am model')
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
            decode1 = tf.image.resize_nearest_neighbor(decode1, (1,60))
            decode1 = tf.reshape(decode1, [-1, 60, decode1.shape[-1]])
            decode1 = tf.layers.dropout(decode1, rate=dropout)

            decode2 = tf.layers.conv1d(decode1, filters=filters[1], kernel_size=kernel_size[1], strides=strides,
                                       padding='same', activation=activation, use_bias=True)

            decode2 = tf.reshape(decode2, [-1, 1, decode2.shape[1], decode2.shape[-1]])
            decode2 = tf.image.resize_nearest_neighbor(decode2, (1,120))
            decode2 = tf.reshape(decode2,[-1,120,decode2.shape[-1]])
            decode2 = tf.layers.dropout(decode2, rate=dropout)

            decode3 = tf.layers.conv1d(decode2, filters=filters[0], kernel_size=kernel_size[0], strides=strides,
                                       padding='same', activation=activation, use_bias=True)

            decode3 = tf.reshape(decode3,[-1,1,decode3.shape[1],decode3.shape[-1]])
            decode3 = tf.image.resize_nearest_neighbor(decode3, (1,266))
            decode3 = tf.reshape(decode3,[-1,266,decode3.shape[-1]])
            decode3 = tf.layers.dropout(decode3, rate=dropout)

            self.decode = tf.layers.conv1d(decode3,filters=1,kernel_size=3, strides=1,padding='same',activation=None)

            #print(self.decode.shape)

        self.loss = tf.losses.mean_squared_error(labels=self.output, predictions=self.decode)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
def lsf_main():
    f = open('../../out/test_lsf/lsf_hamming_16kHZ.pkl','rb')
    lsf = pickle.load(f)[0]



    size = 16000 * 1.0/60
    outdir = '../../out/dense_ae_200_100_10/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    wave_data = []
    for i in range(5):
        f = open('../../out/song%d.wav' % (i+1),'rb')
        data,samplerate = sf.read(f,dtype='int16')
        print(data)
        wave_data.extend(data.tolist()[:-int(size*10)])
    wave_data = np.array(wave_data,dtype='float32')
    #print(wave_data)
    #print(wave_data.max())

    #wave_data /= wave_data.max()
    #wave_data = u_law(wave_data,255)

    mean = wave_data.mean()
    std = wave_data.std()
    wave_data -= mean
    wave_data /= std

    #print(wave_data)
    #print(data.max(),data.min())
    #wave_data = u_law(wave_data,255)
    #print(wave_data.max(),wave_data.min())
    #print(wave_data)



    data = []
    #print(size)
    #print(len(wave_data)/68146)
    #print(len(wave_data)/size)

    for i in range(int(len(wave_data) / size)):
        instance = wave_data[int(i*size):int((i)*size) + int(size)]
        data.append(instance)

    data = np.array(data,dtype='float')

    #data = data.tolist()
    #(len(data))
    train_lsf = np.concatenate([lsf[:25000],lsf[30000:]])
    test_lsf = np.array(lsf[25000:30000])
    train =np.concatenate([data[:25000],data[30000:]])
    test = np.array(data[25000:30000])
    train_xy = zip(train_lsf,train)
    #sf.write('../../out/ae_test_ulow.flac' , test.flatten(), 16000)
    model = lsf_Model(cnn=False)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_size=512

        for i in range(100000):
            batchid = np.random.randint(0,len(train),batch_size)
            #print(batchid)
            batch_x = train_lsf[batchid]
            batch_y = train[batchid]

            #print(train)
            #batch = tf.train.batch([train],batch_size=batch_size)
            #print(batch)
            sess.run(model.optimizer,feed_dict={model.input:batch_x,model.y:batch_y})
            if (i+1) % 1000 == 0:
                trainloss = sess.run(model.loss,feed_dict={model.input:batch_x,model.y:batch_y})
                testloss = sess.run(model.loss,feed_dict={model.input:test_lsf,model.y:test})
                print("%d,trainloss %.6f, testloss %.6f" % (i+1,trainloss,testloss))
                pred = sess.run(model.decode,feed_dict={model.input:test_lsf})

                #f = open(outdir+'hidden_%d.pkl' % (i+1),'wb')
                #pickle.dump(encoded,f)

                sf.write(outdir+'lsf_reconstruct_test_%d.flac' % (i+1),pred.flatten(),16000)
                #pred = sess.run(model.decode,feed_dict={model.input:[batch[0]]})[0]
                plt.plot(range(len(test.flatten())),test.flatten(),label='origin',linewidth=0.1)
                plt.plot(range(len(pred.flatten())), pred.flatten(), label='pred', linewidth=0.1)
                #saver = tf.train.Saver()
                #saver.save(sess,outdir+'model_%d.cpkt' % (i+1))
                plt.legend()
                plt.show()
                #plt.savefig(outdir+'ae_test_%d.png' % (i+1))
                plt.close()
                #plt.show()


    #print(len(wave_dataset))
    #print(len(wave_dataset)*1.0/68146)

    #wave_dataset.reshape(())
def main():
    size = 16000 * 1.0/60
    outdir = '../../out/dense_ae_200_100_2/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    wave_data = []
    for i in range(5):
        f = open('../../out/song%d.wav' % (i+1),'rb')
        data,samplerate = sf.read(f,dtype='int16')
        print(data)
        wave_data.extend(data.tolist()[:-int(size*10)])
    wave_data = np.array(wave_data,dtype='float32')
    #print(wave_data)
    #print(wave_data.max())

    #wave_data /= wave_data.max()
    #wave_data = u_law(wave_data,255)

    mean = wave_data.mean()
    std = wave_data.std()
    wave_data -= mean
    wave_data /= std

    #print(wave_data)
    #print(data.max(),data.min())
    #wave_data = u_law(wave_data,255)
    #print(wave_data.max(),wave_data.min())
    #print(wave_data)



    data = []
    #print(size)
    #print(len(wave_data)/68146)
    #print(len(wave_data)/size)

    for i in range(int(len(wave_data) / size)):
        instance = wave_data[int(i*size):int((i)*size) + int(size)]
        data.append(instance)

    data = np.array(data,dtype='float')

    #data = data.tolist()
    #(len(data))
    train =np.concatenate([data[:25000],data[30000:]]).tolist()
    test = np.array(data[25000:30000])

    sf.write('../../out/ae_test_ulow.flac' , test.flatten(), 16000)
    model = Model(cnn=False)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_size=512

        for i in range(100000):
            batch = random.sample(train,batch_size)
            #print(train)
            #batch = tf.train.batch([train],batch_size=batch_size)
            #print(batch)
            sess.run(model.optimizer,feed_dict={model.input:batch})
            if (i+1) % 1000 == 0:
                trainloss = sess.run(model.loss,feed_dict={model.input:batch})
                testloss = sess.run(model.loss,feed_dict={model.input:test})
                print("%d,trainloss %.6f, testloss %.6f" % (i+1,trainloss,testloss))
                pred = sess.run(model.decode,feed_dict={model.input:test})

                encoded = sess.run(model.encoded,feed_dict={model.input:data})
                f = open(outdir+'hidden_%d.pkl' % (i+1),'wb')
                pickle.dump(encoded,f)

                sf.write(outdir+'ae_test_%d.flac' % (i+1),pred.flatten(),16000)
                #pred = sess.run(model.decode,feed_dict={model.input:[batch[0]]})[0]
                plt.plot(range(len(test.flatten())),test.flatten(),label='origin',linewidth=0.01)
                plt.plot(range(len(pred.flatten())), pred.flatten(), label='pred', linewidth=0.01)
                saver = tf.train.Saver()
                saver.save(sess,outdir+'model_%d.cpkt' % (i+1))
                plt.legend()
                plt.savefig(outdir+'ae_test_%d.png' % (i+1))
                plt.close()
                #plt.show()


    #print(len(wave_dataset))
    #print(len(wave_dataset)*1.0/68146)

    #wave_dataset.reshape(())

def egg_main():
    size = 16000 * 1.0/60
    outdir = '../../out/cnn_ae_egg30/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    wave_data = []
    egg_data = []
    for i in range(5):
        f = open('../../out/song%d.wav' % (i+1),'rb')
        data,samplerate = sf.read(f,dtype='int16')
        wave_data.extend(data.tolist()[:-int(size*10)])
        f = open('../../out/egg%d.wav' % (i + 1), 'rb')
        data, samplerate = sf.read(f, dtype='int16')
        egg_data.extend(data.tolist()[:-int(size * 10)])

    wave_data = np.array(wave_data,dtype='float32')
    mean = wave_data.mean()
    std = wave_data.std()
    wave_data -= mean
    wave_data /= std

    egg_data = np.array(egg_data, dtype='float32')
    mean = egg_data.mean()
    std = egg_data.std()
    egg_data -= mean
    egg_data /= std
    egg_data = np.concatenate([[0]*30,egg_data[:-30]])


    data = []
    ydata = []
    for i in range(int(len(wave_data) / size)):
        instance = wave_data[int(i*size):int((i)*size) + int(size)]
        ydata.append(instance)
        egginstance = egg_data[int(i * size):int((i) * size) + int(size)]
        data.append(egginstance)

    data = np.array(data,dtype='float')
    ydata = np.array(ydata, dtype='float')

    #data = data.tolist()
    #(len(data))
    cnn =True

    train =np.concatenate([data[:25000],data[30000:]])
    test = np.array(data[25000:30000])
    trainy = np.concatenate([ydata[:25000], ydata[30000:]])
    testy = np.array(ydata[25000:30000])
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

        for i in range(100000):

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



                sf.write(outdir+'ae_test_%d.flac' % (i+1),pred.flatten(),16000)

                plt.plot(range(len(test.flatten())),test.flatten(),label='origin',linewidth=0.01)
                plt.plot(range(len(pred.flatten())), pred.flatten(), label='pred', linewidth=0.01)
                saver = tf.train.Saver()
                saver.save(sess,outdir+'model_%d.cpkt' % (i+1))
                plt.legend()
                plt.savefig(outdir+'ae_test_%d.png' % (i+1))
                plt.close()
                #plt.show()


    #print(len(wave_dataset))
    #print(len(wave_dataset)*1.0/68146)

    #wave_dataset.reshape(())



def reproduce(path = '../../out/52002/predict_is16sd0.000.pkl',modelpath='../../out/dense_ae_200_100_2/model_50000.cpkt',recpath='../../out/reconstruct.flac'):
    size = 16000 * 1.0 / 60
    wave_data = []
    for i in range(5):
        f = open('../../out/song%d.wav' % (i + 1), 'rb')
        data, samplerate = sf.read(f, dtype='int16')
        print(data)
        wave_data.extend(data.tolist()[:-int(size * 10)])
    wave_data = np.array(wave_data, dtype='float32')
    mean = wave_data.mean()
    std = wave_data.std()
    wave_data -= mean
    wave_data /= std
    data = []

    for i in range(int(len(wave_data) / size)):
        instance = wave_data[int(i * size):int((i) * size) + int(size)]
        data.append(instance)

    data = np.array(data, dtype='float')
    test = np.array(data[25000:30000])

    f=open(path,'rb')
    hidden = pickle.load(f)
    print(len(hidden))
    print(hidden.shape)

    model = Model(cnn=False,fromhidden=True)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,modelpath)
        realhidden = sess.run(model.encoded,feed_dict={model.input:test})
        predict = sess.run(model.decode,feed_dict={model.hiddeninput:hidden})
        print(predict.shape)
        sf.write(recpath,predict.flatten(),16000)
        plt.plot()
    for i in range(1):
        #plt.plot(range(5000), test[:, 0].flatten())
        plt.plot(range(5000),realhidden[:,i].flatten())
        plt.plot(range(5000),hidden[:,i].flatten())

        plt.show()



if __name__ == "__main__":
    egg_main()
    #reproduce()
    #lsf_main()