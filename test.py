import torch
import torch.nn as nn
from PIL import Image

from torchvision import datasets, transforms

img_transformer = transforms.Compose([
    transforms.Resize((128, 128)),   # Resize to uniform size
    transforms.ToTensor(),           # Convert to tensor
])

#classes = ['ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades', 'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades', 'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades', 'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades', 'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades', 'joker', 'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades', 'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades', 'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades', 'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades', 'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades', 'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades', 'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades', 'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades']  
dataset = datasets.ImageFolder(root="dataset", transform=img_transformer)

#print("Classes:", classes)
print("Classes:", dataset.classes)

class CardCNN(nn.Module):                                   # Make sure this matches the original training model class
    def __init__(self, num_classes):
        super(CardCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),     # Input 3x128x128
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
print(f"\033[33mUsing device: {device}\033[0m\n")

model = CardCNN(num_classes=len(dataset.classes))          # Make sure this matches the original training model
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))

model.eval()  # Set model to evaluation mode
image = Image.open("test1.jpg").convert("RGB")               # Replace with your image
image = img_transformer(image).unsqueeze(0)  
with torch.no_grad():
    output = model(image)
    predicted_index = output.argmax(1).item()

print(f"\033[32mPredicted class:{dataset.classes[predicted_index]}\033[0m")

image = Image.open("test.png").convert("RGB")               # Replace with your image
image = img_transformer(image).unsqueeze(0)  
with torch.no_grad():
    output = model(image)
    predicted_index = output.argmax(1).item()
print(f"\033[32mPredicted class:{dataset.classes[predicted_index]}\033[0m") 