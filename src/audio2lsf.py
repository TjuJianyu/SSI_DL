

from utils.lsf_utils import *

def vocoder(lsf,activation,fs=44100,hamming=True,frame_length=1/60, overlap_percent=50):


    exact_frame_length = fs * frame_length
    signal_length = len(downsample_audio_data)
    downsample_audio_data = np.hstack([np.array([0] * int(exact_frame_length / 2)) \
                                          , downsample_audio_data, np.array([0] * int(exact_frame_length / 2))])
    reproduced = [0]*len(downsample_audio_data)

    for i in tqdm(range(int(signal_length / exact_frame_length))):
        start_index = int(i * exact_frame_length) - int(exact_frame_length / 2) + int(exact_frame_length / 2)
        stop_index = int((i + 1) * exact_frame_length) + int(exact_frame_length / 2) + int(exact_frame_length / 2)
        if hamming:
            frame = downsample_audio_data[start_index:stop_index] * spsig.hamming(stop_index - start_index)
        else:
            frame = downsample_audio_data[start_index:stop_index]
        lsf_iter = lsf[i]
        lpcval = lpd.lsf2poly(lsf_iter)
        lpcfilter = audiolazy.ZFilter(lpcval)
        reproduced_iter = lpcfilter(frame)
        for i in range(len(reproduced_iter)):
            reproduced[start_index+i] += reproduced_iter[i]

    return reproduced[:int(exact_frame_length/2):-int(exact_frame_length/2)]


def audio2lsf_simple(audio_data,fps,order):
    lpcfilter = lpc.nautocor(frame, order)
    lsfPframe = lpd.poly2lsf([lpcfilter[i] for i in range(order + 1)])
    return lsfPframe





def audio2lsf(audio_data, fps, order, hamming=True, \
              usefilter=True, downsample_rate=1, frame_length=1 / 60, useGPU=False, bins=-1):

    downsample_audio_data = audio_data
    fps = fps 

    if usefilter:
        downsample_audio_data = spsig.lfilter([1, -0.95], 1, downsample_audio_data)

    exact_frame_length = fps * frame_length
    signal_length = len(downsample_audio_data)

    lsf = []
    error = 0
    reproduce = []

    downsample_audio_data = np.hstack([np.array([0] * int(exact_frame_length / 2)) \
                                          , downsample_audio_data, np.array([0] * int(exact_frame_length / 2))])

    for i in tqdm(range(int(signal_length / exact_frame_length))):
        start_index = int(i * exact_frame_length) - int(exact_frame_length / 2) + int(exact_frame_length / 2)
        stop_index = int((i + 1) * exact_frame_length) + int(exact_frame_length / 2) + int(exact_frame_length / 2)
        frame = downsample_audio_data[start_index:stop_index] * spsig.hamming(stop_index - start_index)
        lpcfilter = lpc.nautocor(frame, order)
        reproduce_iter = list(lpcfilter(frame))
        reproduce.extend(reproduce_iter)
        err = ((frame - np.array(reproduce_iter)) ** 2).mean()
        error += err
        lpcfilter = list(lpcfilter)[0]
        # print(lpcfilter)
        lsfPframe = lpd.poly2lsf([lpcfilter[i] for i in range(order + 1)])
        # print(lsfPframe)
        # 1/0
        # print(lpcfilter,lsfPframe)
        if bins > 0:
            lsfPframe
        # print(lsfPframe)
        lsf.append(lsfPframe)
    error = error / i
    print("error MSE:%.6f" % (error))

    return np.array(lsf), np.array(reproduce), error
