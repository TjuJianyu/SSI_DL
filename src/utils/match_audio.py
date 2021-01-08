import soundfile as sf
import numpy as np 
import wave 
import audioop



def audio_matching(src,des, head, slope, out_frames,inchannels=1):

    s_read = wave.open(src, 'r')
    s_write = wave.open(des, 'w')
    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)
    
    default_fs = 44100
    inrate = default_fs + default_fs * slope / 60
    outrate = default_fs 
    converted = audioop.ratecv(data, 2, inchannels, int(inrate), outrate, None)
    s_write.setparams((inchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
    s_write.writeframes(converted[0]) 
    s_write.close() 
    
    f = open(des, 'rb')
    data,samplerate = sf.read(f,dtype='float64')

    print(len(data), int(out_frames*default_fs/60))
    
    if head >= 0:
        data = data[int(default_fs * head / 60 * (default_fs / inrate)):]
    else:
        data = np.concatenate([np.array([0]* int(default_fs * (-head) * (default_fs / inrate) / 60 )) , data],axis=0)
    print(int(default_fs * (-head) * (default_fs / inrate) / 60 ))
    print('tail',(len(data) - int(out_frames*default_fs/60)) / 44100 * inrate / (44100 / 60) )
    if len(data) >=int(out_frames*default_fs/60):
        data = data[: int(out_frames*default_fs/60) ]
    else:
        data = np.concatenate([data, np.array([0] * (int(out_frames*default_fs/60) - len(data))) ],axis=0)
    print(len(data))
    
    print('max',data.max(),'min', data.min())
    data *= (2**15 - 1)
    data = data.astype(np.short)
    f = wave.open(des, "wb")
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(default_fs)
    f.writeframes(data.tostring())
    f.close()

    
    return True 

import os
try:
    os.mkdir('../data/Songs_Audio_matched')
except:
    pass
song_path = [
    'MICRO_RecFile_1_20140523_184341_Micro_EGG_Sound_Capture_monoOutput1.wav',
    'MICRO_RecFile_1_20140523_190633_Micro_EGG_Sound_Capture_monoOutput1.wav',
    'MICRO_RecFile_1_20140523_192504_Micro_EGG_Sound_Capture_monoOutput1.wav',
    'MICRO_RecFile_1_20140523_193153_Micro_EGG_Sound_Capture_monoOutput1.wav',
    'MICRO_RecFile_1_20140523_193452_Micro_EGG_Sound_Capture_monoOutput1.wav',
             
            ]
song_new_path = [
    'song2_matched_184341.wav',
    'song1_matched_190633.wav',
    'song3_matched_192504.wav',
    'song4_matched_193153.wav',
    'song5_matched_193452.wav'
]

slope = [0.0333,0.0589,0.0298,0.0502,0.0231]
head = [3.6944,-0.654,-0.1093,3.3839,0.6633]
out_frames =[14514,9599,18559,7235,18239]
for i in range(len(song_path)):
    audio_matching('../data/Songs_Audio/' + song_path[i], '../data/Songs_Audio_matched/' + song_new_path[i], head[i],slope[i], out_frames[i]) 
    