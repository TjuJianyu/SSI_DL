# SSI_DL
silent speech interface by deep learning 
This is the code for silent speech interface paper "xxx".

## Prepare Dataset
To start with: please make out/ data/ log/ dir as follow.
```
mkdir out/ data/ log/ 
```
and put Songs_Audio/ Songs_EGG/ Songs_lips/ Songs_Tongue/ into data/

## Preprocessing 
### Image resize
Resize lips and tongue image to (50,42)
Run
```
python image_preprocessing.py
```
### Audio downsample
Downsample audio and EGG to 16khz and 10.025khz
Run
```
python utils/downsample.py
```
## Train
### Comparing different architectures (It can still gain better by furder parameter tuning. I only tune a little bit.)
Run
```
python experiments_egg_audio.py
```

### generate activation function by EGG
Run
```
python experiments_lips_tongues_lsf.py
```

### reconstruct audio by LSF and activation function
Run 
```

```
