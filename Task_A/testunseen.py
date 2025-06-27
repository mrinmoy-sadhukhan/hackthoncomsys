# file: test_coatnet_gender.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import timm
import numpy as np

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test Directory
TEST_DIR = 'C:\\Users\\mrinmoy\\Desktop\\Comys_Hackathon5\\Comys_Hackathon5\\Task_A\\train'

# Transforms (same as validation)
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset & DataLoader
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Model Definition
class GenderClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("coatnet_0_rw_224", pretrained=False, num_classes=2)

    def forward(self, x):
        return self.model(x)

# Load Model
model = GenderClassifier().to(DEVICE)
model.load_state_dict(torch.load("best_coatnet_gender_model.pth",map_location=torch.device('cpu')))
model.eval()

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print("\nDetailed Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, zero_division=0))

##val dataset
#Test Accuracy: 0.9360
#Test Precision: 0.9460
#Test Recall: 0.9360
#Test F1-Score: 0.9385

#Detailed Classification Report:

#              precision    recall  f1-score   support

#      female       0.77      0.95      0.85        79
#        male       0.99      0.93      0.96       343

#    accuracy                           0.94       422
#   macro avg       0.88      0.94      0.90       422
#weighted avg       0.95      0.94      0.94       422
##train dataset
#Test Accuracy: 0.9590
#Test Precision: 0.9663
#Test Recall: 0.9590
#Test F1-Score: 0.9607

#Detailed Classification Report:

#              precision    recall  f1-score   support

#      female       0.80      0.99      0.88       303
#        male       1.00      0.95      0.98      1623

#    accuracy                           0.96      1926
#   macro avg       0.90      0.97      0.93      1926
#weighted avg       0.97      0.96      0.96      1926
