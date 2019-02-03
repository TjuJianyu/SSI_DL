import wave
import numpy as np
from spectrum import linear_prediction as lpd
import time
import random
#from tensorflow.keras.regularizers import l1,l2,l1_l2i
import tensorflow as tf
import cv2
from audiolazy import lpc
#from matplotlib import pyplot as plt
import scipy.signal as spsig
import os
from tqdm import tqdm
import pickle 
from scipy import signal 
import math
import argparse
EXP_NAME = str(random.randint(0,65536))
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def nowtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 

def wavereader(f_path):
    f = wave.open(f_path,"rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data,dtype=np.short)
    return wave_data, nchannels, sampwidth, framerate, nframes
def musicdata_wavereader():
    songs = ['184341','190633','192504','193153','193452']
    song_name ="MICRO_RecFile_1_20140523_%s_Micro_EGG_Sound_Capture_monoOutput1.wav"
    wave_dataset = []
    totalnframes = 0
    
    zerocount = 0
    for song in songs:
        wave_data, nchannels, sampwidth, framerate, nframes\
        = wavereader('../data/Songs_Audio/%s' % (song_name % song))
        print(nframes)
        print(nframes*1.0/44100 * 60)
        print(int(nframes/44100) * 60)
        
        i = nframes-1
        count = 0
        while wave_data[i] == 0:
            count += 1
            i-=1
        zerocount  += count
        print("#zero end",count)
        #eliminate the last 10 * 1/60 s
        wave_data = wave_data[:-int(10 * framerate * 1/60)]
        nframes -= int(10 * framerate * 1/60)
        wave_dataset += wave_data.tolist()
        totalnframes += nframes
        print("loaded wave %s: nchannels %d, sampwidth %d, framerate %d, nframes %d" %\
              (song,nchannels, sampwidth, framerate, nframes))
    #print(zerocount)
    wave_dataset = np.array(wave_dataset)
   
    wave_dataset = 2*(wave_dataset -(-32767))/ ( 32767 - (-32767))-1

    #print(wave_dataset[:10])
    print("total # frames %d" % totalnframes)
    
    return wave_dataset, nchannels, sampwidth, framerate, totalnframes

def load_dataset(IS16=False,reshape=False):
    print('loading...')
    if IS16:
        f = open("../out/test_lsf/lsf.pkl","rb")
        lsf = pickle.load(f)
        lsf = lsf.tolist()
        lips = []
        tongue=[]
        tong_fdir = "../out/resize_tongue/%s.tif"
        lips_fdir = "../out/resize_lips/%s.tif"
        for i in tqdm(range(1,68147)):
            
            ran = str(i)
            for i in range(6-len(ran)):
                ran = '0'+ran
            lframe_0 = cv2.imread(lips_fdir%ran,0)
            lframe_0.resize(32,32)
            lframe_0 = np.array(lframe_0)
            lips.append(lframe_0.reshape((lframe_0.shape[0],lframe_0.shape[1],1)))

            tframe_0 = cv2.imread(tong_fdir%ran,0)
            tframe_0.resize(32,32)
            tframe_0 = np.array(tframe_0)
            tongue.append(tframe_0.reshape((tframe_0.shape[0],tframe_0.shape[1],1)))
            
            
            
        train_lips  =lips[:25000]
        train_lips.extend(lips[30000:])
        train_tongue =tongue[:25000]
        train_tongue.extend(tongue[30000:])
        train_lsf = lsf[:25000]
        train_lsf.extend(lsf[30000:])
        
        train_lips = np.array(train_lips)
        train_tongue = np.array(train_tongue)
        train_lsf = np.array(train_lsf)
        test_lips =np.array( lips[25000:30000])
        test_tongue = np.array(tongue[25000:30000])
        test_lsf = np.array(lsf[25000:30000])
        
        #return train_tongue,train_lips,train_lsf,test_tongue,test_lips,test_lsf
    
    elif os.path.exists("../out/train_tongue_01.pkl") and \
    os.path.exists("../out/train_lips_01.pkl") and \
    os.path.exists("../out/train_lsf_01.pkl"):
        f1 = open("../out/train_tongue_01.pkl","rb")
        f2 = open("../out/train_lips_01.pkl","rb")
        f3 = open("../out/train_lsf_01.pkl","rb")
        f4 = open("../out/test_tongue_01.pkl","rb")
        f5 = open("../out/test_lips_01.pkl","rb")
        f6 = open("../out/test_lsf_01.pkl","rb")
        train_tongue = pickle.load(f1)
        train_lips = pickle.load(f2)
        train_lsf = pickle.load(f3)
        test_tongue = pickle.load(f4)
        test_lips = pickle.load(f5)
        test_lsf = pickle.load(f6)
    
    else:
        f = open("../out/test_lsf/lsf.pkl","rb")
        lsf = pickle.load(f)
        lips = []
        tongue=[]
        tong_fdir = "../out/resize_tongue/%s.tif"
        lips_fdir = "../out/resize_lips/%s.tif"
        for i in tqdm(range(1,68147)):
            
            ran = str(i)
            for i in range(6-len(ran)):
                ran = '0'+ran
            lframe_0 = cv2.imread(lips_fdir%ran,0)
            lframe_0.resize(32,32)
            lframe_0 = np.array(lframe_0)
            lips.append(lframe_0.reshape((lframe_0.shape[0],lframe_0.shape[1],1)))

            tframe_0 = cv2.imread(tong_fdir%ran,0)
            tframe_0.resize(32,32)
            tframe_0 = np.array(tframe_0)
            tongue.append(tframe_0.reshape((tframe_0.shape[0],tframe_0.shape[1],1)))
        testrate=0.1
        length = np.array([10670146,7050413,13645126,5311110,13410518])
        length = (length / 44100*60)
        test = length*testrate
        left = length.cumsum() - test
        right = length.cumsum()
        left = left.astype(int)
        right = right.astype(int)
        test_lips,train_lips,test_tongue,train_tongue,test_lsf,train_lsf = [],[],[],[],[],[]
        for i in range(len(left)):
            test_lips.extend(lips[left[i]:right[i]])
            test_tongue.extend(tongue[left[i]:right[i]])
            test_lsf.extend(lsf[left[i]:right[i]])
 
        for i in range(len(left)):
            if i == 0:
                start = 0
            else:
                start = right[i-1]
            train_lips.extend(lips[start:left[i]])
            train_tongue.extend(tongue[start:left[i]])
            train_lsf.extend(lsf[start:left[i]])
        train_lips = np.array(train_lips,dtype=int)
        train_tongue = np.array(train_tongue,dtype=int)
        train_lsf = np.array(train_lsf)
        test_lips = np.array(test_lips,dtype=int)
        test_tongue = np.array(test_tongue,dtype=int)
        test_lsf = np.array(test_lsf)
        print(test_lsf)
        f1 = open("../out/train_tongue_01.pkl","wb")
        f2 = open("../out/train_lips_01.pkl","wb")
        f3 = open("../out/train_lsf_01.pkl","wb")
        f4 = open("../out/test_tongue_01.pkl","wb")
        f5 = open("../out/test_lips_01.pkl","wb")
        f6 = open("../out/test_lsf_01.pkl","wb")
        pickle.dump(train_tongue,f1)
        pickle.dump(train_lips,f2)
        pickle.dump(train_lsf,f3)
        pickle.dump(test_tongue,f4)
        pickle.dump(test_lips,f5)
        pickle.dump(test_lsf,f6)   
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()
        
    return train_tongue,train_lips,train_lsf,test_tongue,test_lips,test_lsf


def simple_audio2lsf(audio_data,order):
    lpcfilter = lpc.nautocor(audio_data,order)
    reproduce = np.array(list(lpcfilter(frame)))
    lpcfilter = list(lpcfilter)[0]
    lsf = lpd.poly2lsf([ lpcfilter[i] for i in range(order+1) ])
    return lsf  

def audio2lsf(audio_data, fps, order,hamming=False,\
              usefilter=True, downsample_rate=1, frame_length = 1/60, useGPU=False,bins=-1):
    #order = int(order / downsample_rate)
    #print(audio_data[:20])
    downsample_audio_data = spsig.decimate(audio_data, 4,n=8,ftype='iir')
    #print(len(audio_data),len(audio_data)*1.0/len(downsample_audio_data))
    #print(len(downsample_audio_data))
    #print(downsample_audio_data[:20])
    #downsample_audio_data = audio_data
    fps = fps / downsample_rate
    #print(downsample_audio_data[:10])
    if usefilter:
        downsample_audio_data = spsig.lfilter([1, -0.95],1,downsample_audio_data)
    #print(downsample_audio_data[:20])
    #print(spsig.lfilter([1,-0.95],1,[1,2,3,4,5,6,7,8,9]))
    exact_frame_length = fps * frame_length
    signal_length = len(downsample_audio_data)
    
    lsf = []
    error = 0
    reproduce = []
    
    downsample_audio_data = np.hstack([np.array([0]*int(exact_frame_length/2)) \
                                       , downsample_audio_data , np.array([0]*int(exact_frame_length/2))])

    for i in tqdm(range(int(signal_length / exact_frame_length))):
        start_index = int(i * exact_frame_length) - int(exact_frame_length/2)  + int(exact_frame_length/2) 
        stop_index = int((i+1) * exact_frame_length) + int(exact_frame_length/2) + int(exact_frame_length/2) 
        #print(start_index,stop_index)
        frame = downsample_audio_data[start_index:stop_index]*spsig.hamming(stop_index-start_index)
        #print(frame[360:379])
        #print(len(frame))

        #plt.plot(range(len(frame)),frame)
        #plt.show()
        lpcfilter = lpc.nautocor(frame,order)

        reproduce_iter = list(lpcfilter(frame))
        reproduce.extend(reproduce_iter)

        err = ((frame - np.array(reproduce_iter))**2).mean()
        error+=err
        #print(error)
        lpcfilter = list(lpcfilter)[0]
        #print(lpcfilter)
        lsfPframe = lpd.poly2lsf([ lpcfilter[i] for i in range(order+1) ])
        #print(lsfPframe)
        #1/0
        #print(lpcfilter,lsfPframe)
        if bins > 0:
            lsfPframe
        #print(lsfPframe)
        lsf.append(lsfPframe)
    error = error / i 
    print("error MSE:%.6f" % (error))
        

    
    return np.array(lsf), np.array(reproduce), error 

def mean_option_score(SD):
    mos = 3.56 - 0.8*SD + 0.04*(SD**2)
    dmos = mos - (3.56 - 0.8*1 + 0.04*(1**2))
    return mos, dmos

def spectral_distortion(lsf_true,lsf_pred,N,n0,n1):
    SD = []
    IS16SD = []
    print(len(lsf_true))
    print(len(lsf_pred))
    for frameid in range(len(lsf_true)):
        lpc_true = lpd.lsf2poly(lsf_true[frameid])
        lpc_pred = lpd.lsf2poly(lsf_pred[frameid])

        _, freqResponse_true = signal.freqz(b=1,a=lpc_true,worN=N)
        _, freqResponse_pred = signal.freqz(b=1,a=lpc_pred,worN=N)
        
        freq_th = freqResponse_true[n0-1:n1]
        freq_pred = freqResponse_pred[n0-1:n1]

        absoluteRadio = (freq_th.real**2+freq_th.imag**2)**0.5 / (freq_pred.real**2+freq_pred.imag**2)**0.5
        
        logValue = np.log10(absoluteRadio**2)
        bigsum = ((10*logValue)**2).sum()
        sd = math.sqrt(1.0/(n1-n0)) * bigsum
        IS16sd = math.sqrt(1.0/(n1-n0) * bigsum)
        SD.append(sd)
        IS16SD.append(IS16sd)
        
    return SD,IS16SD, sum(SD) *1.0 / len(SD),sum(IS16SD)*1.0/len(IS16SD)


def cnn_model_keras(conv3d=False,classification=0,multi_task=True,lw1=1e-8):
    
    if conv3d:
        lips_inputs = tf.keras.Input(shape=(8,32,32,1))
        tongue_inputs = tf.keras.Input(shape=(8,32,32,1))
        
        layer = tf.keras.layers.Conv3D(filters=16,kernel_size=(5,5,4),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(lips_inputs)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        layer = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,2),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        layer = tf.keras.layers.Conv3D(filters=64,kernel_size=(3,3,2),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        layer = tf.keras.layers.Conv3D(filters=128,kernel_size=(3,3,2),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        lips = tf.keras.layers.Flatten()(layer)
        
        layer = tf.keras.layers.Conv3D(filters=16,kernel_size=(5,5,4),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(tongue_inputs)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        layer = tf.keras.layers.Conv3D(filters=32,kernel_size=(3,3,2),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        layer = tf.keras.layers.Conv3D(filters=64,kernel_size=(3,3,2),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        layer = tf.keras.layers.Conv3D(filters=128,kernel_size=(3,3,2),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        tongue = tf.keras.layers.Flatten()(layer)
        
        
        concat_lip_tog = tf.keras.layers.concatenate([lips,tongue])
        concat_lip_tog = tf.keras.layers.Dense(1000, activation = tf.nn.relu,\
                                               kernel_regularizer=tf.keras.regularizers.l1(1e-4),\
                                               name='fc')(concat_lip_tog)

    else:
        lips_inputs = tf.keras.Input(shape=(32,32,1),name='lips')
        tongue_inputs = tf.keras.Input(shape=(32,32,1),name='tongue')
        layer = tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(lips_inputs)
        layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        layer = tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        layer = tf.keras.layers.Conv2D(filters=128,kernel_size=(5,5),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        lips = tf.keras.layers.Flatten()(layer)


        layer = tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(tongue_inputs)
        layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)

        layer = tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        layer = tf.keras.layers.Conv2D(filters=128,kernel_size=(5,5),kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                                       padding='same',activation=tf.nn.relu)(layer)
        layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization(axis=1)(layer)
        
        
        tongue = tf.keras.layers.Flatten()(layer)

        concat_lip_tog = tf.keras.layers.concatenate([lips,tongue])
        concat_lip_tog = tf.keras.layers.Dense(1000, activation = tf.nn.relu,\
                                               kernel_regularizer=tf.keras.regularizers.l1(1e-4),\
                                               name='fc')(concat_lip_tog)
    
    if classification > 1:
        if multi_task:
            pred = []
            for i in range(12):
                sub_pred = concat_lip_tog
                #sub_pred = tf.keras.layers.Dense(1000, activation = tf.nn.relu,  kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                #                                 name='neck_each%i' % i)(concat_lip_tog)
                #sub_pred = tf.keras.layers.Dense(1000, activation = tf.nn.relu,  kernel_regularizer=tf.keras.regularizers.l1(lw1),\
                #                                 name='neck2_each%i' % i)(concat_lip_tog)
                
                sub_pred = tf.keras.layers.Dense(classification, \
                                                 activation=tf.nn.softmax,\
                                                 name='pred_lsf%d' % i)(sub_pred)
                pred.append(sub_pred)
            model = tf.keras.Model(inputs =[lips_inputs,tongue_inputs], \
                                   outputs = pred)
        else:
            sub_pred = tf.keras.layers.Dense(1000, activation = tf.nn.relu, \
                                                 name='neck')(concat_lip_tog)
            pred = tf.keras.layers.Dense(classification, \
                                         activation=tf.nn.softmax)(concat_lip_tog)
            
            model = tf.keras.Model(inputs = [lips_inputs,tongue_inputs],outputs = pred)
            
    else:
        if multi_task:
            pred = []
            for i in range(12):
                sub_pred = concat_lip_tog
                #sub_pred = tf.keras.layers.Dense(1000, activation = tf.nn.relu, \
                #                                 name='neck_%d' % i)(concat_lip_tog)
                sub_pred = tf.keras.layers.Dense(1, \
                                                 activation=None,\
                                                 name='pred_lsf%d' % i)(sub_pred)
                pred.append(sub_pred)
            model = tf.keras.Model(inputs =[lips_inputs,tongue_inputs], \
                                   outputs = pred)        
            
        else:
            sub_pred = tf.keras.layers.Dense(1000, activation = tf.nn.relu, \
                                                 name='neck_each%i' % i)(concat_lip_tog)
            pred = tf.keras.layers.Dense(1,activation=None)(concat_lip_tog)
            model = tf.keras.Model(inputs = [lips_inputs,tongue_inputs],outputs = pred)
        
    return model



def target_preprocessing(train_lsf,test_lsf,classification,order=12):
    train_ys = []
    test_ys = []
    if classification > 0:
        for i in range(order):
            tr_i_max = train_lsf[:,i].max()
            tr_i_min = train_lsf[:,i].min()
            cl_train_lsf = (classification*((train_lsf[:,i] - tr_i_min) / (tr_i_max - tr_i_min))).astype(int)
            cl_train_lsf[cl_train_lsf>=classification]= classification-1
            cl_train_lsf = tf.keras.utils.to_categorical(cl_train_lsf,classification)
            cl_test_lsf = (classification*((test_lsf[:,i] - tr_i_min) / (tr_i_max - tr_i_min))).astype(int)
            cl_test_lsf[cl_test_lsf>=classification]= classification-1
            cl_test_lsf[cl_test_lsf<0]= 0
            cl_test_lsf = tf.keras.utils.to_categorical(cl_test_lsf,classification)
            train_ys.append(cl_train_lsf)
            test_ys.append(cl_test_lsf)
    else:
        for i in range(order):
            train_ys.append(train_lsf[:,i][:,np.newaxis])
            test_ys.append(test_lsf[:,i][:,np.newaxis])
    return train_ys,test_ys

def model_compile(model,optimizer,classification,multi_task=True):
    if classification > 0:
        if multi_task:
            loss = {'pred_lsf%d' % i: "categorical_crossentropy" for i in range(12)}
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=["categorical_crossentropy"])
        else:
            model.compile(optimizer=optimizer,loss='categorical_crossentropy',\
                          metrics=["categorical_crossentropy"])
    else:
        if multi_task:
            loss = {'pred_lsf%d' % i: "mse" for i in range(12)}
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=["mse"])
        else:
            model.compile(optimizer=optimizer,loss='mse',metrics=['mse'])

def measure(ytrue,ypred,train_lsf,classification):  
    predict = ypred
    test_lsf = ytrue
    if classification > 0:
        predict_lsf = []
        for i in range(12):
            predict_iter = predict[i].argmax(axis=1)
            tr_i_max = train_lsf[:,i].max()
            tr_i_min = train_lsf[:,i].min()
            predict_n =  ((predict_iter +0.5) / classification)*(tr_i_max - tr_i_min) + tr_i_min 
            predict_lsf.append(predict_n)
        predict_lsf = np.array(predict_lsf)
        predict_lsf = predict_lsf.transpose()
        _,_,sd,is16sd = spectral_distortion(predict_lsf,test_lsf,512,6,200)
        print("sd %.3f is16sd %.3f" % (sd,is16sd))    
    else:            
        predict = [val.flatten() for val in predict]
        predict = np.array(predict)
        predict_lsf = predict.transpose()
        predict_lsf[predict_lsf>3.1415926]=3.1415926
        predict_lsf[predict_lsf<0]=0  
        _,_,sd,is16sd = spectral_distortion(predict_lsf,test_lsf,512,6,200)
        print("sd %.3f is16sd %.3f" % (sd,is16sd))
    return sd, is16sd,predict_lsf
def data_preprocessing(train_tongue,train_lips,train_lsf,test_tongue,test_lips,test_lsf,steps=8):
    data = []
    for x in [train_tongue,train_lips,test_tongue,test_lips]:
        subdata = []
        for i in tqdm(range(len(x))):
            if (i-int(steps/2) >= 0) and (i+steps-int(steps/2) <= len(x)):
                subdata.append(x[i-int(steps/2):i+steps-int(steps/2)])
        data.append(np.array(subdata))
    for x in [train_lsf,test_lsf]:
        subdata = []
        for i in range(len(x)):
            if (i-int(steps/2) >= 0) and (i+steps-int(steps/2) <= len(x)):
                subdata.append(x[i])
        data.append(np.array(subdata))
        
    return data[0],data[1],data[4],data[2],data[3],data[5]

def keras_train(conv3d=False,classification=0,order=12,multi_task=True,IS16=False):

    try:
        os.mkdir("../out/%s" % EXP_NAME)
    except:
        pass
    #load data
    train_tongue,train_lips,train_lsf,test_tongue,test_lips,test_lsf = load_dataset(IS16)   
    
    #data preprocessing
    if conv3d:
        train_tongue,train_lips,train_lsf,test_tongue,test_lips,test_lsf = \
        data_preprocessing(train_tongue,train_lips,train_lsf,test_tongue,test_lips,test_lsf)

    #target preprocessing
    train_ys, test_ys = target_preprocessing(train_lsf,test_lsf,classification)
    
    #load model 
    model = cnn_model_keras(conv3d = conv3d,classification= classification,multi_task = multi_task)
    print(model.summary())
    
    #load optimizer
    optimizer=tf.keras.optimizers.Adam(lr = 0.0001)
    
    #compile
    model_compile(model,optimizer,classification,multi_task)
    
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,mode='min')
   
    if multi_task:
        for i in range(20):
            model.fit([train_tongue,train_lips],train_ys,\
                      validation_data=[[test_tongue,test_lips],test_ys],\
                      verbose=2,\
                      batch_size=512,initial_epoch=i,epochs=i+1,shuffle=True)
            predict = model.predict([test_tongue,test_lips],batch_size=512)
            sd, is16sd,predict_lsf = measure(test_lsf,predict,train_lsf,classification)

            
    else:
        predict = []
        for i in range(12):
            model.reset_states()
            model.fit([train_tongue,train_lips],train_ys[i],validation_data=[[test_tongue,test_lips],test_ys[i]],batch_size=512 ,epochs=5,shuffle=True)
            predict_iter = model.predict([test_tongue,test_lips],batch_size=512)
            predict.append(predict_iter)
        sd, is16sd,predict_lsf = measure(test_lsf,predict,train_lsf,classification)
        
    f = open("../out/%s/predict_epoch%d_is16sd%.3f.pkl" % (EXP_NAME,i,is16sd),'wb')
    pickle.dump(predict_lsf,f)
    f.close()           





        
def parse_args():
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--conv3d',type=bool,
                        default=False,help='use 3d cnn model')
    parser.add_argument('--classification',type=int,
                        default=100,help='count of classification target, 0 for regression')
    parser.add_argument('--multi_task',type=bool,
                        default=True,help='12 lsf target as multi task or not')
    parser.add_argument('--is16',type=bool,
                        default=True,help='use test dataset as is16')
    parser.add_argument('--name',type=str,
                        default='',help='name of experiment')
    
   
   
   
    return parser.parse_args()
    
    
    


if __name__ == "__main__":
    
    args = parse_args()
    
    keras_train(conv3d=args.conv3d,
                classification=args.classification,
                multi_task=1 if args.multi_task else 0,
                IS16=args.is16)
    
    
