import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from PIL import Image 
import time
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(root = "dataset", transform=transform)     
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CardCNN(nn.Module):                                   # Make sure this matches the original training model class
    def __init__(self, num_classes, colour_channels= 3):
        super(CardCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(colour_channels, 32, kernel_size=3, padding=1),     # Input 3x128x128
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 32x64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 64x32x32
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {f"\033[32m{device}\033[0m" if f"{device}"=="cuda" else f"{device}"} \n")

model = CardCNN(num_classes=len(test_dataset.classes), colour_channels=3)      # same architecture as training change colour channels if you have to
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss()  

correct = 0
wrong = 0
total = 0
batch_no = 0
total_loss = 0

print("Evaluating accuracy of the mmodel...")
print(f"\nUsing device: {f"\033[32m{device}\033[0m" if f"{device}" == "cuda" else f"{device}"} \n")

start_time = time.time()
with torch.no_grad():                           # disables gradient calculation (faster, less memory)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)    # get class with highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        batch_no+=1
        print(f"batch: {batch_no+1}")
       

print(f"\nTotal images: {total}\nCorrect predictions: {f"\033[32m{correct}\033[0m" if ((100 * correct / total) >=98.0)  else (f"\033[31m{correct}\033[0m" if((100 * correct / total)<=60.0) else f"{correct}")}  \nAverage loss (per image): {total_loss/total}")
print(f"Accuracy of the model: { f"\033[32m{(100 * correct / total)}%\033[0m" if ((100 * correct / total) >=98.0)  else (f"\033[31m{(100 * correct / total)}%\033[0m" if((100 * correct / total)<=60.0) else f"{(100 * correct / total)}%" )}")
print(f"\n\033[32mEvaluation completed in {time.time() - start_time:.2f} seconds\033[0m")