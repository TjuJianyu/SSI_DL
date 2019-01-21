import wave
import numpy as np
#from specturm import linear_prediction as lpd
import tensorflow as tf
import cv2
from audiolazy import lpc


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
    for song in songs:
        wave_data, nchannels, sampwidth, framerate, nframes\
        = wavereader('../data/Songs_Audio/%s' % (song_name % song))
        wave_dataset += wave_data.tolist()
        print("loaded wave %s: nchannels %d, sampwidth %d, framerate %d, nframes %d" %\
              (song,nchannels, sampwidth, framerate, nframes)) 
    return wave_dataset, nchannels, sampwidth, framerate, nframes

def imagereader():
    cv2.imread()
    
def simple_cnn_model():
    lips = tf.placeholder(dtype='int32',shape=[None,lip_l,lip_w,1])
    togues = tf.placeholder(dtype='int32',shape=[None,tog_l,tog_w,1])
    
    conv_lip = tf.layers.conv2d(inputs=lips,filters=16,kernel_size=(5,5),PADDING="same",activation=tf.nn.relu)
    conv_lip = tf.layers.max_pooling2d(inputs = conv_lip, pool_size=(2,2),strides=(2,2),padding='same')
    conv_lip = tf.layers.batch_normalization(inputs=conv_lip,axis=1)
    
    
    
    
    


if __name__ == "__main__":
#     wave_data, nchannels, sampwidth, framerate, nframes = \
#     wavereader("..\data\Songs_Audio\MICRO_RecFile_1_20140523_184341_Micro_EGG_Sound_Capture_monoOutput1.wav")
#     print( nchannels, sampwidth, framerate, nframes)
#     print(wave_data)
    #musicdata_wavereader()
    lpcfilter = lpc.nautocor([1, -2, 3, -4, -3, 2, -3, 2, 1], order=3)
    reproduct = lpcfilter([1, -2, 3, -4, -3, 2, -3, 2, 1])
    print(list(reproduct))