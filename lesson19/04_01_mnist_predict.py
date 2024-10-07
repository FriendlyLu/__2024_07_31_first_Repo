# 載入套件
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image

# 模型載入
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pt').to(device)

# 使用小畫家，繪製 0~9，實際測試看看
# 讀取影像並轉為單色
# image to a Torch tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([28, 28]),
    transforms.PILToTensor()
])

for i in range(10):
    uploaded_file = f'./myDigits/{i}.png'
    image = Image.open(uploaded_file)
    X1 = transform(image)

    # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
    X1 = torch.FloatTensor(255.0-X1).to(device)

    # 預測
    print(f'actual/prediction: {i} {model(X1).argmax(dim=1).item()}')