import wave
import numpy as np
from spectrum import linear_prediction as lpd
import time
import random
import tensorflow as tf
import cv2
from audiolazy import lpc
import scipy.signal as spsig
import os
from tqdm import tqdm
import pickle
import datetime
from scipy import signal
import math
import argparse
import scipy.io as sio
import audioop
def downsampleWav(src, dst, inrate=44100, outrate=16000, inchannels=1, outchannels=1):
    if not os.path.exists(src):
        print('Source not found!')
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print('Failed to open files!')
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
    if outchannels == 1 & inchannels != 1:
        converted[0] = audioop.tomono(converted[0], 2, 1, 0)
    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted[0])
    except:
        print('Failed to write wav')
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print('Failed to close wav files')
        return False

    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Run.')

    parser.add_argument('--srcpath', type=str,
                        help='source audio path')
    parser.add_argument('--dstpath', type=str,
                        help='destinate audio path')

    parser.add_argument('--inrate', type=int,
                        default=44100, help='source audio frequency')
    parser.add_argument('--outrate', type=int,
                        default=16000, help='destinate audio frequency')

if __name__ == "__main__":
    #args = parse_args()
    #downsampleWav(args.srcpath,args.dstpath,args.inrate,args.outrate)

    song_name = "../data/Songs_Audio_matched/song%d_matched_%s.wav"
    for i,val in enumerate(['190633','184341', '192504', '193153', '193452']):
        downsampleWav(song_name % (i+1,val),'../out/song%d_16000hz.wav' % (i+1) ,44100,16000)
        downsampleWav(song_name % (i+1,val),'../out/song%d_11025hz.wav' % (i+1), 44100,11025)



