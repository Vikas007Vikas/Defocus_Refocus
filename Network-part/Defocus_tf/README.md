# Defocus Magnification
An implementation of Defocus Magnification described in the paper using tensorflow.
* [ Defocus Magnification using Conditional Adversarial Networks]

## Requirement
- Python 2.7
- Tensorflow 1.4.0 
- numpy 1.14.5
- Pretrained VGG19 file : [vgg19.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) (for training!)

## Datasets
- [GOPRO dataset](https://github.com/SeungjunNah/DeepDeblur_release)

## Pre-trained model
- [Defocus_model](https://drive.google.com/open?id=1G1eHVri3Nntyi_1zXARZX4MbYHo7pDJZ)

## Train using GOPRO dataset
1) Download pretrained VGG19 file
[vgg19.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

2) Download GOPRO dataset
[GOPRO dataset](https://github.com/SeungjunNah/DeepDeblur_release)

3) Preprocessing GOPRO dataset. 
```
python GOPRO_preprocess.py --GOPRO_path ./GOPRO/data/path --output_path ./data/output/path
```

4) Train using GOPRO dataset.
```
python main.py --train_Sharp_path ./GOPRO/path/sharp --train_Blur_path ./GOPRO/path/blur
```
## Train using your own dataset
1) Download pretrained VGG19 file 
[vgg19.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

2) Preprocess your dataset. Blur image and sharp image pair should have same index when they are sorted by name respectively. 

3) Train using GOPRO dataset.
```
python main.py --train_Sharp_path ./yourData/path/sharp --train_Blur_path ./yourData/path/blur
```
