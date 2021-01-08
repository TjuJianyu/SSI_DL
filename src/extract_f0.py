from utils.lsf_utils import *
import pandas as pd
import matplotlib.pyplot as plt

def main():
    context = 0.5
    smoothstep = 20
    threshold = 15
    _,_,trainsong,testsong = loadeggsong('../out/')

    eachsize = len(trainsong[0])

    traineggflatten = trainsong.flatten()
    testeggflatten = testsong.flatten()

    names = ['trainsongfeat','testsongfeat']
    for i,data in enumerate([traineggflatten,testeggflatten]):

        records = []
        fit_records = []
        size = len(trainsong) if i ==0 else len(testsong)
        for j in tqdm(range(size)):



            wave = data[ max(0,int((j - context)*eachsize)) : min(len(data) ,int((j+1 + context) * eachsize))]
            smoothwave = pd.Series(wave).rolling(smoothstep,center=True).mean().values[int(smoothstep/2):int(-smoothstep/2)]

            square_data,softmax, softmin, median= to_square_wave(smoothwave)

            _, _, _, cyc, bound = wave_cyc(square_data,smoothwave,threshold)
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