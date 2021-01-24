# SSI_DL

This is the code for silent speech interface paper ["Creating Song from Lip and Tongue Videos with a Convolutional Vocoder"](https://ieeexplore.ieee.org/document/9319643?source=authoralert).

## Prepare Dataset

To start with: please make out/ data/ log/ dir as follow.

```
mkdir out/ data/ log/ 
```

Download data from [here](https://github.com/TjuJianyu/SSI_dataset.git) and move all .tar.gz files to ./data/

Then unzip all files by:

```
cd SSI_DL/data
tar -xzf songs_audio.tar.gz
tar -xzf resize_lips.part1.tar.gz 
tar -xzf resize_lips.part1.tar.gz 
mkdir ../out/resize_lips 
mv part_*/* ../out/resize_lips 

tar -xzf resize_tongue.part1.tar.gz 
tar -xzf resize_tongue.part1.tar.gz 
mkdir ../out/resize_tongue 
mv part_*/* ../out/resize_tongue
```



## Preprocessing 

### Audio downsample
Downsample audio and EGG to 16khz and 10.025khz

```
python utils/downsample.py
```
### Calculating LSF for original Audio

```
python utils/audio2lsf.py
```

### Extract F0

```
python extract_f0.py
```


## Learning LSF from lips and tongue 
Train a cnn model to predict LSF coefficients from lips and tongue images. 

```
python train_lsf.py
```
## Learning F0 from lips and tongue

```
python train_f0.py
```

## Learning U/V flat from lips and tongue

```
python train_uv.py
```


## CNN Vocoder
Learning audio from LSF, F0 and U/V flat. 

```
python train_cnn_vocoder.py
```


## Generate Audio 
Sythesis audios from lips and tongue images. 

```
python test_cnn_vocode.py
```

