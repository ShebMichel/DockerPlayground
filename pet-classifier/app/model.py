import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained ResNet model
model = models.resnet18(pretrained=True)
model.eval()  # Set to evaluation mode

# ImageNet classes
import json
# with open("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as f:
#     classes = [line.strip() for line in f]

import requests

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = requests.get(url).text.splitlines()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, idx = torch.max(outputs, 1)
    return classes[idx.item()]

# Example
# list_of_cats = ["is_my_cat_healthy.jpg","silver-tabby-cat-sitting-on-green-background-free-photo.jpg",]
# for catname in list_of_cats:
#     print(predict_image("../assets/test_examples/"+str(catname)))  # Should output something like 'tabby cat'

# some cats are from:
# https://aristopet.com.au/tips-and-advice/is-my-cat-healthy/
