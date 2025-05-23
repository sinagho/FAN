# Adaptive Frequency-Aware Framework for Robust Medical Image Segmentation <br> <span style="float: right"><sub><sup>MICCAI 2025 (Submitted)</sub></sup></span>

[![arXiv](https://img.shields.io/badge/arXiv-2308.13442-b31b1b.svg)](https://arxiv.org/abs/2308.13442)

Medical image segmentation requires adaptive receptive fields to accurately capture multi-scale structures, as fixed receptive fields in traditional CNNs and vision transformers (ViTs) often lead to under-segmentation or boundary inaccuracies. To address this, we propose the Frequency Adaptive Network (FAN), which leverages spatial and frequency-aware sampling mechanisms for dynamic receptive field adaptation. FAN introduces three key components: (1) Context-Aware Convolution (CAC), which dynamically adjusts sampling offsets based on local frequency components and spatial information to optimize receptive fields for both large-scale structures and fine boundaries; (2) Selective Feature Aggregation (SFA), which selectively enhances critical frequency components within skip connections to improve fine-detail preservation; and (3) Frequency-Domain Denoising Module (FDM), which models degradation patterns in the spectral domain to enhance robustness in noisy degraded images. Experimental results demonstrate that FAN outperforms state-of-the-art methods, achieving superior localization and
<p align="center">
  <b>F</b>requency <b>E</b>nhanced <b>T</b>ransformer (<b>FET</b>) model</em>
  <br/>
  <img width="600" alt="image" src="https://github.com/sinagho/Wave/blob/main/WaveFormer_code/assets/pdfresizer.com-pdf-crop%20(5)%20(1)-1.png"/>
  <br/>
  <br>
  An illustration of our proposed <b>EW-ViT</b>
  <br/>
  <img width="700" alt="image" src="https://github.com/sinagho/Wave/blob/main/WaveFormer_code/assets/Attention_Compressed_new%20(1)-1.png"/>
  <br>
  (a) Overview of the <b>E</b>nhanced <b>W</b>ave <b>A</b>ttention Block
</p>



## Citation
```bibtex

```

## News
- Oct 28, 2024: Accepted in WACV 2025 🥳

## How to use

  ### Requirements
  
  - Ubuntu 16.04 or higher
  - CUDA 11.1 or higher
  - Python v3.7 or higher
  - Pytorch v1.7 or higher
  - Hardware Spec
    - A single GPU with 12GB memory or larger capacity (_we used RTX 3090_)

  ```
einops
h5py
imgaug
matplotlib
MedPy
numpy
opencv_python
pandas
PyWavelets
scipy
SimpleITK
tensorboardX
timm
torch
torchvision
tqdm
  ```
  `pip install -r requirements.txt`

  ### Model weights
  You can download the learned weights in the following.
   Dataset   | Model | download link 
  -----------|-------|----------------
   Synapse   | FET   | [[Download]()] 
  
  ### Training and Testing (Synapse)

1) Download the Synapse dataset from [here](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi).

2) Run the following code to install the Requirements.

    `pip install -r requirements.txt`

3) Run the below code to train the EW-ViT on the Synapse dataset.
    ```bash
    python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5 --batch_size 20 --eval_interval 20 --max_epochs 700
    ```
    **--root_path**     [Train data path]

    **--test_path**     [Test data path]

    **--eval_interval** [Evaluation epoch]
 4) Run the below code to test the EW-ViT on the Synapse dataset.
    ```bash
    python test.py --volume_path ./data/Synapse/ --output_dir ./model_out
    ```
    **--volume_path**   [Root dir of the test data]
        
    **--output_dir**    [Directory of your learned weights]
    
### Training and Testing (ACDC)

1) Download the ACDC dataset.

2) Run the following code to install the Requirements.

    `pip install -r requirements.txt`

3) Run the below code to train the EW-ViT on the ACDC dataset.
    ```bash
    python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5 --batch_size 20 --eval_interval 20 --max_epochs 700
    ```
    **--root_path**     [Train data path]

    **--test_path**     [Test data path]

    **--eval_interval** [Evaluation epoch]
 4) Run the below code to test the EW-ViT on the ACDC dataset.
    ```bash
    python test.py --volume_path ./data/Synapse/ --output_dir ./model_out
    ```
    **--volume_path**   [Root dir of the test data]
        
    **--output_dir**    [Directory of your learned weights]  

## Experiments

### Synapse Dataset
<p align="center">
  <img width="600" alt="Synapse images" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/6d2fa946-75ca-4a63-895c-ea2db633ff46">
  <img style="max-width:2020px" alt="Synapse results" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/6ac06b5e-a3bc-4de3-bd8b-d3ce2d0843f2">
</p>

### ISIC 2018 Dataset
<p align="center">
  <img style="width: 65%; float:left" alt="ISIC images" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/f0937aa3-3deb-4696-b38e-8501cf097a22">
  <img style="width: 34%;" alt="ISIC results" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/6d6a6433-af6d-4fab-b992-112b7b8dcf44">
</p>

## References
- DAEFormer [https://github.com/mindflow-institue/DAEFormer]
- ImageNetModel [https://github.com/YehLi/ImageNetModel]
