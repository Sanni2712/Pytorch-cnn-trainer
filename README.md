# Pytorch-cnn-trainer
A simple python project that trains an image recognition model based on a provided image folder dataset in the program folder and a program to test saved model

## Usage

project_folder/<br>
├── dataset (folder) <br>
├── trianer.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Trians the model and saves it in the same folder<br>
├── test.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loads the model and uses it to test by predicting an image class<br>
├── model.pth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;saved model<br>
├── test.png/jpg &nbsp;&nbsp;&nbsp;&nbsp;(almost any image format is supported)<br>
...<br>

model class in testing program should be same as the model class in training program as the . 

## Requirements
__Librearies used here__<br>
PyTorch (torch) — core deep learning library.<br>

TorchVision (torchvision) — contains datasets and image transforms.<br>

Pillow (PIL) — image processing library.<br>

### Installation

`pip install torch torchvision pillow` (CPU-only) same on Windows, Linux, and macOS <br>
(`torch.cuda.is_available()` will return False for CPU-only build)

**For CUDA support**<br>
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` (CUDA 12.1) or select your preferd version, works on Windows and linux <br>
`pip install pillow`<br>

make sure - Your GPU supports CUDA 12.1<br>
You have the matching NVIDIA driver installed (for CUDA 12.1, that’s driver version ≥ 530)<br>
You’re using Python 3.8–3.12 (PyTorch support range)

No installation required for Google Collab.

### Dataset
Your image dataset should follow this structre - <br>
dataset/<br>
├── class1/<br>
│   ├── img001.jpg<br> 
│   ├── img002.jpg<br>
│   └── ...<br>
├── class2/<br>
│   ├── img101.jpg<br>
│   └── ...<br>
...<br><br>
and the program  folder like this - 
...<br>
├── dataset (folder)<br>
├── trianer.py<br>
├── test.py<br>
...<br>
then in your trianer.py - <br>
`dataset = datasets.ImageFolder(root="dataset", transform=img_transformer)`<br>

note - .jpg / .jpeg .png .bmp .gif (static GIFs only — animated ones will only load the first frame) .tiff / .tif .webp .ppm, .pgm, .pbm (Netpbm formats) .ico (icons) are all supported.
        <br>image names can be anything, they do not need any special naming sequence, each folder can have only one type of mobject that needs to be predicted and the folder name is the predicted class name.

if you have - <br>
dataset/<br>
│<br>
├── train/<br>
│   ├── class1/<br>
│   │   ├── img001.jpg<br>
│   │   ├── img002.jpg<br>
│   │   └── ...<br>
│   ├── class2/<br>
│   │   ├── img101.jpg <br>
│   │   └── ...<br>
│<br>
└── test/   ← optional <br>
    ├── class1/<br>
    │   ├── imgX.jpg<br>
    │   └── ...<br>
    ├── class2/<br>
    │   ├── imgY.jpg<br>
    │   └── ...<br>
then in your trianer.py make sure to have - <br>
`dataset = datasets.ImageFolder(root="dataset/train", transform=img_transformer)`<br>
and `dataset = datasets.ImageFolder(root="dataset/test", transform=img_transformer)`  ← optional (only if you have a Evaluation phase, data in /trian can also be used for the same)

### Loading popular datasets directly
**MNIST (handwritten digits)**<br>
`
from torchvision import datasets`<br>
`dataset = datasets.MNIST(root="data", download=True, transform=...)
`<br>
and it will:
Download the MNIST(handwritten digits) dataset if it’s not present<br>
Store it in program folder<br>
Give you easy access to the images + labels in a PyTorch-friendly format<br>

similarly -
**Fashion-MNIST (clothing images)**
fashion_mnist = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)<br>
