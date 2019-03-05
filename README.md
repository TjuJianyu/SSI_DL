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
Downsample audio to 16khz and 10.025khz
Run
```
python downsample.py --srcpath ../data/MICRO_RecFile_1_20140523_184341_Micro_EGG_Sound_Capture_monoOutput1.wav --dstpath ../out/Song1.wav 
python downsample.py --srcpath ../data/MICRO_RecFile_1_20140523_190633_Micro_EGG_Sound_Capture_monoOutput1.wav --dstpath ../out/Song2.wav 
python downsample.py --srcpath ../data/MICRO_RecFile_1_20140523_192504_Micro_EGG_Sound_Capture_monoOutput1.wav --dstpath ../out/Song3.wav 
python downsample.py --srcpath ../data/MICRO_RecFile_1_20140523_193153_Micro_EGG_Sound_Capture_monoOutput1.wav --dstpath ../out/Song4.wav 
python downsample.py --srcpath ../data/MICRO_RecFile_1_20140523_193452_Micro_EGG_Sound_Capture_monoOutput1.wav --dstpath ../out/Song5.wav 
```
