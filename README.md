# CVMSI

This is the Pytorch code for our paper "Cross-view and Multi-step Interaction for Change Captioning" (under review).



## Installation
1. Clone this repository
2. cd CVMSI
1. Make virtual environment with Python 3.10.14
2. Install requirements (`pip install -r requirements.txt`)
3. Setup COCO caption eval tools ([github](https://github.com/mtanti/coco-caption)) 
4. An NVIDA 4090 GPU or others.

## Data
1. Download data from [Baidu drive link](https://pan.baidu.com/s/1GmLNwCE-jo-qoYpteJYsqQ?pwd=s257).

2. Download clevr-change dataset from [RobustChangeCaptioning](https://github.com/Seth-Park/RobustChangeCaptioning).

3. Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
python scripts/extract_features.py --input_image_dir ./data/images --output_dir ./data/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./data/sc_images --output_dir ./data/sc_features --batch_size 128

# processing distractor images
python scripts/extract_features.py --input_image_dir ./data/nsc_images --output_dir ./data/nsc_features --batch_size 128
```

## Testing/Inference
We provide pre-trained weights, download it from [Baidu drive link](https://pan.baidu.com/s/1u8oOEwKVc-tKsOOJrV2O4A?pwd=48ii).
```
python test_trans_c.py --cfg configs/transformer-c.yaml  --snapshot 25000 --gpu 0
```
