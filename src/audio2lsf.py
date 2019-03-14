from utils.lsf_utils import *
import soundfile as sf
import pickle


size = 16000 * 1.0/60
wave_data = []
for i in range(5):
    f = open('../out/song%d.wav' % (i+1),'rb')
    data,samplerate = sf.read(f,dtype='int16')
    wave_data.extend(data.tolist()[:-int(size*10)])
wave_data = np.array(wave_data,dtype='float32')
print(wave_data.max(),wave_data.min())
wave_data = 2 * (wave_data - (-32767)) / (32767 - (-32767)) - 1
lsf16khz = audio2lsf(wave_data,fps=16000,order=12)
f = open('../out/test_lsf/lsf_hamming_16kHZ.pkl','wb')
pickle.dump(lsf16khz,f)


size = 16000 * 1.0/60
wave_data = []
for i in range(5):
    f = open('../out/song%d_10025.wav' % (i+1),'rb')
    data,samplerate = sf.read(f,dtype='int16')
    wave_data.extend(data.tolist()[:-int(size*10)])
wave_data = np.array(wave_data,dtype='float32')
print(wave_data.max(),wave_data.min())
wave_data = 2 * (wave_data - (-32767)) / (32767 - (-32767)) - 1
lsf11khz = audio2lsf(wave_data,fps=16000,order=12)
f = open('../out/test_lsf/lsf_hamming_11.025kHZ.pkl','wb')
pickle.dump(lsf11khz,f)


