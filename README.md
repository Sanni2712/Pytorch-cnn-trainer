# Pytorch-cnn-trainer
A simple python project that trains an image recognition model based on a provided image folder dataset in the program folder and a program to test saved model

## Usage
```
project_folder/
...
├── dataset (folder)
├── trianer.py              Trians the model and saves it in the same folder<br>
├── test.py                 Loads the model and uses it to test by predicting an image class<br>
├── model.pth               saved model<br>
├── test.png/jpg            (almost any image format is supported)<br>
...
```
Model class in testing program should be same as the model class in training program as the . 

## Requirements
__📦📦Packages__<br>
- PyTorch (torch) — core deep learning library.<br>
- TorchVision (torchvision) — contains datasets and image transforms.<br>
- Pillow (PIL) — image processing library.<br>

### Installation
__🔲 CPU-only__
```
pip install torch torchvision pillow
```
pip commands are same on Windows, Linux, and macOS <br>
(`torch.cuda.is_available()` will return False for CPU-only build)
<br><br>
**🟩 CUDA support**
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pillow
```
Make sure -
- Your GPU supports CUDA 12.1 - RTX 20-series (Turing), RTX 30-series (Ampere), and RTX 40-series (Ada Lovelace) <br>
- Matching NVIDIA driver is installed (for CUDA 12.1, that’s driver version ≥ 530)<br>
- Python 3.8–3.12 (PyTorch support range)

No installation required for Google Collab ♾️.

### 🗃️ Custom Image Folder Dataset
Your image dataset should follow this structre - <br>
```
dataset/<br>
├── class1/<br>
│   ├── img001.jpg<br> 
│   ├── img002.jpg<br>
│   └── ...<br>
├── class2/<br>
│   ├── img101.jpg<br>
│   └── ...<br>
...
```
and the program folder like this - 
```
...
├── dataset (folder)
├── trianer.py
├── test.py
...
```
then in your `trianer.py` should have - <br>
```
dataset = datasets.ImageFolder(root="dataset", transform=img_transformer)
```
<br>

**📌 note -**
<br>Image names can be anything, they do not need any special naming sequence, each folder can have only one object that needs to be predicted and the folder name is the predicted class name.
<br>

   supported formats -
        `.jpg` / `.jpeg` `.png` `.bmp` `.gif` (only the first frame) `.tiff`/`.tif` `.webp` `.ppm` `.pgm` `.pbm` `.ico`<br>
&nbsp;        
<br>
if you have 
```
dataset/
│
├── train/
│   ├── class1/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── img101.jpg 
│   │   └── ...
│<br>
└── test/   ← optional 
    ├── class1/
    │   ├── imgX.jpg
    │   └── ...
    ├── class2/
    │   ├── imgY.jpg
    │   └── ...
```
then in your `trainer.py` , make sure to have - <br>
```
dataset = datasets.ImageFolder(root="dataset/train", transform=img_transformer)
```
and

```
dataset = datasets.ImageFolder(root="dataset/test", transform=img_transformer)
``` 
(optional, only if you have a Evaluation phase, data in /trian can also be used for the same)

### 📥Loading popular datasets 

**MNIST (handwritten digits)**<br>
```
dataset = datasets.MNIST(root="data", download=True, transform=...)
```
Download the MNIST dataset if it’s not present<br>
Store it in program folder<br>
Give you easy access to the images + labels in a PyTorch-friendly format<br>

similarly -<br>
**Fashion-MNIST (clothing images)**
```
fashion_mnist = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
```

__CIFAR-10 (10 object classes)__
```
cifar10 = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)
```

**CIFAR-100 (100 object classes)**
```
cifar100 = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=transform
)
```

**STL-10 (similar to CIFAR-10, higher resolution)**
```
stl10 = datasets.STL10(
    root="data",
    split='train',   # 'train', 'test', or 'unlabeled'
    download=True,
    transform=transform
)
```


**Datasets that use train (bool)**
These have exactly two subsets: train and test.<br>
```
train=True  # training set
train=False # test set
```
- MNIST
- FashionMNIST
- KMNIST (japanese characters)
- QMNIST (extended version of MNSIT handwritten digits)
- CIFAR10
- CIFAR100
- EMNIST (also has split for variant selection, but train/test is still bool)
- SVHN (kind of — uses split param named "train" but accepts bool-like usage)

__Datasets that use split (string)__
These have more than two possible subsets.
```
split="train"    # training set
split="test"     # test set
split="valid"    # validation set
split="unlabeled"  # unlabeled images
split="all"      # full dataset
STL10 → "train", "test", "unlabeled"

CelebA → "train", "valid", "test", "all"

Caltech101 / Caltech256 → "train", "test", "all"
```

- COCO (Captions/Detection) → split implied by folder, not a param — but concept is the same

- VOCSegmentation / VOCDetection → "train", "val", "trainval", "test"

- Places365 → "train-standard", "train-challenge", "val"

**Datasets with neither**
These just load whatever is in the directory or URL you give — no built-in split.
ImageNet (split implied by folder)
LSUN (uses class list instead of split param)

## General Terminology:
### epoch
In machine learning, especially when training neural networks, an epoch is one complete pass through the entire training dataset.

### batch size
In our code, we have -
```
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```
`batch_size=32` means that the DataLoader will give 32 images at a time to the model during training, 
your dataset has, say, 3,200 images:

One epoch = 3,200 ÷ 32 = 100 batches.

The DataLoader will make 100 steps per epoch, each step containing 32 images

### learning rate
In our code, we have:
```
optimizer = optim.Adam(model.parameters(), lr=l_r)
```
or
```
optimizer = optim.Adam(model.parameters(), lr=0.001) #default value
```
`l_r` or `lr` is the learning rate.
When your optimizer (like torch.optim.SGD or Adam) updates the model weights, it uses this formula (simplified):

`new_weight = old_weight - lr × gradient`

gradient = direction & size of change needed (from backpropagation, core algorithm that enables the network to learn from its mistakes and improve its predictions by adjusting the weights and biases of its connections)

lr = how big each step should be in that direction

**Effects of learning rate:**
- Too high → training can oscillate wildly or fail to converge
- Too low → training becomes very slow and may get stuck in a local minimum
- Balanced → converges steadily

