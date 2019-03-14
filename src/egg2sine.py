import heapq
import numpy as np
from utils.lsf_utils import  *
from tqdm import tqdm
import pickle
from scipy import optimize
import matplotlib.pyplot as plt



def main():
    context = 0
    trainegg,testegg,trainsong,testsong = loadeggsong('../out/')

    eachsize = len(trainegg[0])

    traineggflatten = trainegg.flatten()
    testeggflatten = testegg.flatten()

    names = ['traineggfeat','testeggfeat']
    for i,data in enumerate([traineggflatten,testeggflatten]):

        records = []
        fit_records = []
        size = len(trainegg) if i ==0 else len(testegg)
        for j in tqdm(range(size)):

            wave = data[ max(0,int((j - context)*eachsize)) : min(len(data) ,int((j+1 + context) * eachsize))]
            square_data,softmax, softmin, median= to_square_wave(wave)

            _, _, _, cyc, bound = wave_cyc(square_data, wave)
            record = cyc, sum(bound)/2, (max(bound) - min(bound))/2
            records.append(record)

            params, params_covariance = optimize.curve_fit(test_func2((max(bound) - min(bound)) / 2, cyc, sum(bound) / 2),
                                                           np.array(range(len(wave))), wave,p0=[0],maxfev = 10000)

            fit_records.append([(max(bound) - min(bound)) / 2, cyc, params[0],sum(bound) / 2])


        fit_records = np.array(fit_records)
        f = open('../out/%s_fit.pkl' % names[i],'wb')
        pickle.dump(fit_records,f)

        f.close()
        records = np.array(records)
        f = open('../out/%s.pkl' % names[i],'wb')
        pickle.dump(records,f)
        f.close()

if __name__ == "__main__":
    main()