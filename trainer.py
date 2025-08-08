import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import torch.optim as optim
from PIL import Image

img_transformer = transforms.Compose([
    transforms.Resize((128, 128)),   # Resize to uniform size
    transforms.ToTensor(),           # Convert to tensor
])

dataset = datasets.ImageFolder(root="dataset", transform=img_transformer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
d = 0
save_model_name = input("Save the model as: _______.pth: ")

l_r = 0.001
try:
    l_r = float(input("\n\033[34mLearning rate [default lr=0.001]\033[0m \nthis value controls how big a step the optimizer takes when updating model weights during training.\nLess the lr value more the training time, more the accuracy\nEnter the lr value or simply press enter to proceed with default value: "))
except:
    l_r = 0.001
try:
   epc = int(input("\n\033[34mEpochs [default: 10]\033[0m \n(An epoch is one complete pass through the entire training dataset by the model)\nEnter the number of epochs to execute or simply press enter to proceed with default value: "))
except:
   epc = 10


print("\nClasses:", dataset.classes)


class CardCNN(nn.Module):
    def __init__(self, num_classes):
        super(CardCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input 3x128x128
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 32x64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 64x32x32
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
print(f"\nUsing device: {device} \n")
model = CardCNN(num_classes=len(dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
if l_r == "":
    optimizer = optim.Adam(model.parameters(), lr=0.001)
else:
    optimizer = optim.Adam(model.parameters(), lr=l_r)

start_time = time.time()
print(f"\033[32mtraining started...\033[0m \ndetails:\n lr count: {l_r}\n epoch count: {epc}\n")

try:
    for epoch in range(epc):
        print(f"\033[33mEpoch: {epoch+1}\033[0m")
        t = time.time()
        total_loss = 0
        model.train()
        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"  loss: {loss.item()}")
            total_loss += loss.item()

    print(f"Epoch - {epoch+1}:\n Total loss: {total_loss:.4f}")
    t = (time.time() - t)
    print(f" time taken {epoch+1}: {(t)} seconds")
    d +=t
    #print(f"\nTraining completed in {time.time() - start_time:.2f} seconds \nAverage time per epoch: {(d)/epc}")
    print(f"\n\033[32mTraining completed in {time.time() - start_time:.2f} seconds \nAverage time per epoch: {(d)/epc}\033[0m")

    torch.save(model.state_dict(), f"{save_model_name}.pth")
    print(f"model saved as: {save_model_name}.pth")
except Exception as e:
    print(f"error: {e}")
