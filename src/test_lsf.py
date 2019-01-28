import ssi_cnn
import numpy as np
import wave 
import os
import scipy.signal as spsig
import matplotlib.pyplot as plt
from spectrum import linear_prediction as lpd
from audiolazy import lpc
import pickle 


# # test the performance of filter 
# fr_wave,nchannels, sampwidth, framerate, nframes = ssi_cnn.wavereader("../data/Songs_Audio/MICRO_RecFile_1_20140523_184341_Micro_EGG_Sound_Capture_monoOutput1.wav")

# fr_wave_filted = spsig.filtfilt([1,-0.95],1,fr_wave)
# fr_wave_filted = fr_wave_filted.astype(np.short)
# try:
#     os.mkdir("../out/test_lsf")
# except:
#     pass
# f = wave.open("../out/test_lsf/filter_MICRO_RecFile_1_20140523_184341_Micro_EGG_Sound_Capture_monoOutput1.wav", "wb")
# f.setnchannels(nchannels)
# f.setsampwidth(sampwidth)
# f.setframerate(framerate)
# f.writeframes(fr_wave_filted.tostring())
# f.close()

# plt.plot(range(len(fr_wave)),fr_wave,label="original wave")
# plt.plot(range(len(fr_wave_filted)),fr_wave,label='wave after filter [1 -0.95]')
# plt.title("effect of filter on original wave")
# plt.legend()
# plt.show()





# data,nchannels,sampwidth,framerate,nframes = ssi_cnn.musicdata_wavereader()

# lsf,reproduce,error = ssi_cnn.audio2lsf(data,framerate,12)

# lsff = open("../out/test_lsf/lsf.pkl","wb")
# pickle.dump(lsf,lsff)

# plt.plot(range(len(data)),data,label="original wave")
# plt.plot(range(len(reproduce)),reproduce,label='lpc reproduced')
# plt.title("effect of lpc reproduction ")
# plt.legend()
# plt.show()

# f = wave.open("../out/test_lsf/filter_reproduced_MICRO_RecFile_1_20140523_184341_Micro_EGG_Sound_Capture_monoOutput1.wav", "wb")
# f.setnchannels(nchannels)
# f.setsampwidth(sampwidth)
# f.setframerate(framerate)
# f.writeframes(reproduce.astype(np.short).tostring())
# f.close()



# data,nchannels,sampwidth,framerate,nframes = ssi_cnn.musicdata_wavereader()

# lsf,reproduce,error = ssi_cnn.audio2lsf(data,framerate,12,hamming=True)

# lsff = open("../out/test_lsf/lsf_hamming.pkl","wb")
# pickle.dump(lsf,lsff)

# plt.plot(range(len(data)),data,label="original wave")
# plt.plot(range(len(reproduce)),reproduce,label='lpc reproduced')
# plt.title("effect of lpc reproduction")
# plt.legend()
# plt.show()  
data,nchannels,sampwidth,framerate,nframes = ssi_cnn.musicdata_wavereader()

lsf,reproduce,error = ssi_cnn.audio2lsf(data,framerate,12,hamming=True,downsample_rate=4)

lsff = open("../out/test_lsf/lsf_hamming_ds4.pkl","wb")
pickle.dump(lsf,lsff)

f = wave.open("../out/test_lsf/filter_repro_hm_ds4_MICRO_RecFile_1_20140523_184341_Micro_EGG_Sound_Capture_monoOutput1.wav", "wb")
f.setnchannels(nchannels)
f.setsampwidth(sampwidth)
f.setframerate(framerate)
f.writeframes(reproduce.tostring())
f.close()
