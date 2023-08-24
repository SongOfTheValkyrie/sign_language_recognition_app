import torch 
import torch.nn as nn
from collections import defaultdict
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = '/gris/gris-f/homestud/jplapper/SignLanguageApp/sign_language_recognition_app/asl_recognition_model.pth'
state_dict = torch.load(model_path)
mobilenet_small = mobilenet_v3_small(pretrained=True)

for param in mobilenet_small.parameters():
    param.requires_grad = False

# Adjust the final layer according to the number of classes in your dataset
num_ftrs = mobilenet_small.classifier[3].in_features
NUM_CLASSES = 24
mobilenet_small.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mobilenet_small.load_state_dict(state_dict)
mobilenet_small.eval()
dataset_path = '/gris/gris-f/homestud/jplapper/SignLanguageApp/SigNN Character Database'

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
batch_size = 64
full_dataset = datasets.ImageFolder(dataset_path, transform=data_transforms['train'])

# Split the dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Now apply the appropriate transformations to each dataset split
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])  # Assumes sequential model
        
    def forward(self, x):
        return self.features(x)

feature_extractor = FeatureExtractor(mobilenet_small).to(device)

mean_features = defaultdict(list)

# Compute the features for each sample in your training set
for inputs, labels in train_loader:
    inputs = inputs.to(device)
    features = feature_extractor(inputs).detach()
    
    for feature, label in zip(features, labels):
        mean_features[label.item()].append(feature)

# Compute the mean feature vector for each class
for label, features in mean_features.items():
    mean_features[label] = torch.stack(features).mean(0)


all_features = []

all_features_list = []
# Collect features for all samples in your training set
for inputs, _ in train_loader:
    inputs = inputs.to(device)
    features = feature_extractor(inputs).detach()
    all_features_list.append(features.view(features.size(0), -1))  # Flatten each batch of features

all_features = torch.cat(all_features_list, 0)

# Now compute the covariance and precision matrices
covariance_matrix = torch.Tensor(np.cov(all_features.cpu().data.numpy(), rowvar=False))
precision_matrix = torch.pinverse(covariance_matrix).to(device)


def mahalanobis_distance(vector, mean, precision):
    vector = vector.squeeze()
    mean = mean.squeeze()
    diff = vector - mean

    return torch.dot(diff, torch.mv(precision, diff))


def compute_mcs(input):
    feature = feature_extractor(input).squeeze()
    distances = []
    for label, mean in mean_features.items():
        distance = mahalanobis_distance(feature, mean.to(device), precision_matrix)
        distances.append(distance)
    
    return min(distances).item()

val_distances=[]
for inputs, _ in val_loader:
    inputs = inputs.to(device)
    for input in inputs:
        distance = compute_mcs(input.unsqueeze(0))  # compute_mcs expects a batch input
        val_distances.append(distance)

threshold =  np.percentile(val_distances, 95)

def is_ood(input):
    distance = compute_mcs(input)
    return distance > threshold

from PIL import Image
import torchvision.transforms as transforms

# Define the same transformations you used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # For MobileNetV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet stats
])

# Load the image
image_path = "/gris/gris-f/homestud/jplapper/SignLanguageApp/sign_language_recognition_app/ood-image.jpg"
image = Image.open(image_path)

# Apply transformations
image = transform(image)

# Add a batch dimension (since the model expects batches)
image = image.unsqueeze(0).to(device)

if is_ood(image):
    print("The input might be out-of-distribution!")
else:
    prediction = mobilenet_small(image)
    # Handle the prediction
