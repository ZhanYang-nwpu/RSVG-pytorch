# RSVG: Exploring Data and Model for Visual Grounding on Remote Sensing Data
##### Author: Zhan Yang 
This is the offical PyTorch code for paper **"RSVG: Exploring Data and Model for Visual Grounding on Remote Sensing Data"**, [arxiv](https://arxiv.org/abs/2210.12634).

## Introduction
This is Multi-level Cross-modal Fusion Network, the PyTorch source code of the paper "RSVG: Exploring Data and Model for Visual Grounding on Remote Sensing Data". It is built on top of the [TransVG](https://github.com/djiajunustc/TransVG) in PyTorch. Our method is a transformer-based method for visual grounding for remote sensing data (RSVG). It has achieved the SOTA performance in the RSVG task on our constructed RSVGD dataset.

### Network Architecture
<p align="middle">
    <img src="fig/architecture.jpg">
</p>

## Requirements and Installation
We recommended the following dependencies.
- Python 3.6.13
- PyTorch 1.9.0
- NumPy 1.19.2
- cuda 11.1
- opencv 4.5.5
- torchvision

## Download Data
Download our constructed RSVGD dataset files. We build the first large-scale dataset for RSVG, termed RSVGD, which can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing). The download link is available below:
```
https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing
```
   
We expect the directory and file structure to be the following:
```
./                      # current (project) directory
├── models/             # Files for implementation of RSVG model
├── utils/              # Some scripts for data processing and helper functions 
├── saved_models/       # Savepath of pth/ckpt and pre-trained model
├── logs/               # Savepath of logs
├── data_loader.py      # Load data
├── main.py             # Main code for training, validation, and test
├── README.md
└── RSVGD/              # RSVGD dataset
    ├── Annotations/    # Query expressions and bounding boxes
    │   ├── 00001.xml/
    │   └── ..some xml files..
    ├── JPEGImages/     # Remote sensing images
    │   ├── 00001.jpg/
    │   └── ..some jpg files..
    ├── train.txt       # ID of training set
    ├── val.txt         # ID of validation set
    └── test.txt        # ID of test set
```

## Training and Evaluation
```
python main.py
```

Run ```main.py``` using ```--test False``` to train new models on RSVGD.
Evaluate trained models on RSVGD using ```--test True```.

## Reference
If you found this code useful, please cite the paper after it is online. Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then I will let you know when we update.
```
@misc{https://doi.org/10.48550/arxiv.2210.12634,
  title = {RSVG: Exploring Data and Models for Visual Grounding on Remote Sensing Data},
  author = {Zhan, Yang and Xiong, Zhitong and Yuan, Yuan},
  year = {2022},
  doi = {10.48550/ARXIV.2210.12634},
  url = {https://arxiv.org/abs/2210.12634},
  publisher = {arXiv}
}
```

## Acknowledgments
Our RSVGD is constructed based on the [DIOR](http://www.escience.cn/people/JunweiHan/DIOR.html) remote sensing image dataset. We thank to the authors for releasing the dataset. Part of our code is borrowed from [TransVG](https://github.com/djiajunustc/TransVG). We thank to the authors for releasing codes. I would like to thank Xiong zhitong and Yuan yuan for helping the manuscript. I also thank the School of Artificial Intelligence, OPtics, and ElectroNics (iOPEN), Northwestern Polytechnical University for supporting this work.
