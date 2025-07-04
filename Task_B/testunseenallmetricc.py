# ========== Imports ========== #
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# ========== Configuration ========== #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_ROOT = 'C:\\Users\\mrinmoy\\Desktop\\Comys_Hackathon5\\Comys_Hackathon5\\Task_B\\train'  # ✅ UPDATE to your dataset path
IMG_SIZE = 224
EMBED_DIM = 256
N_WAY = 5
N_SHOT = 5
N_QUERY = 1
TEST_EPISODES = 250
MODEL_PATH = 'best_protonet.pth'  # ✅ Path to your saved model

# ========== Image Transform ========== #
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== Dataset Class ========== #
class FewShotDataset:
    def __init__(self, root_dir, transform, n_shot=5, n_query=1):
        self.root_dir = root_dir
        self.transform = transform
        self.n_shot = n_shot
        self.n_query = n_query

        self.classes = []
        self.class_to_images = {}

        for cls in os.listdir(root_dir):
            class_path = os.path.join(root_dir, cls)
            if not os.path.isdir(class_path):
                continue

            clean_images = [os.path.join(class_path, img) for img in os.listdir(class_path)
                            if img.endswith('.jpg') and not os.path.isdir(os.path.join(class_path, img))]

            distortion_path = os.path.join(class_path, 'distortion')
            if os.path.exists(distortion_path):
                distorted_images = [os.path.join(distortion_path, img) for img in os.listdir(distortion_path)
                                    if img.endswith('.jpg')]
            else:
                distorted_images = []

            if len(clean_images) >= self.n_query and len(distorted_images) >= self.n_shot:
                self.classes.append(cls)
                self.class_to_images[cls] = {'clean': clean_images, 'distorted': distorted_images}

        if len(self.classes) == 0:
            raise Exception("No valid classes found. Please check your dataset structure or sampling parameters.")

    def sample_episode(self, n_way=5):
        if len(self.classes) < n_way:
            raise Exception(f"Not enough classes available. Requested: {n_way}, Available: {len(self.classes)}")

        while True:
            selected_classes = random.sample(self.classes, n_way)

            support_images = []
            support_labels = []
            query_images = []
            query_labels = []

            label_mapping = {cls: idx for idx, cls in enumerate(selected_classes)}
            valid_episode = True

            for cls in selected_classes:
                clean_imgs = self.class_to_images[cls]['clean']
                distorted_imgs = self.class_to_images[cls]['distorted']

                if len(clean_imgs) < self.n_query or len(distorted_imgs) < self.n_shot:
                    valid_episode = False
                    break

                support_samples = random.sample(distorted_imgs, self.n_shot)
                query_samples = random.sample(clean_imgs, self.n_query)

                for img_path in support_samples:
                    img = Image.open(img_path).convert('RGB')
                    support_images.append(self.transform(img))
                    support_labels.append(label_mapping[cls])

                for img_path in query_samples:
                    img = Image.open(img_path).convert('RGB')
                    query_images.append(self.transform(img))
                    query_labels.append(label_mapping[cls])

            if valid_episode and support_images and query_images:
                return (torch.stack(support_images), torch.tensor(support_labels),
                        torch.stack(query_images), torch.tensor(query_labels), label_mapping)

# ========== Model ========== #
class ProtoNet(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, embed_dim)

    def forward(self, x):
        x = self.features(x).squeeze()
        return self.fc(x)

# ========== Euclidean Distance and Prediction ========== #
def euclidean_dist(a, b):
    n = a.size(0)
    m = b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return torch.pow(a - b, 2).sum(2)

def predict_protonet(support_embeddings, support_labels, query_embeddings, n_way):
    prototypes = []
    for c in range(n_way):
        class_mask = (support_labels == c)
        class_embeddings = support_embeddings[class_mask]
        class_proto = class_embeddings.mean(0)
        prototypes.append(class_proto)
    prototypes = torch.stack(prototypes)
    dists = euclidean_dist(query_embeddings, prototypes)
    scores = -dists
    return scores

# ========== Test Script ========== #
def test_protonet(model, test_dataset, device, n_way=5, n_shot=5, n_query=1, episodes=250):
    model.eval()
    all_preds = []
    all_labels = []

    for _ in tqdm(range(episodes)):
        support_x, support_y, query_x, query_y, label_mapping = test_dataset.sample_episode(n_way)

        idx_to_class = {v: k for k, v in label_mapping.items()}

        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        with torch.no_grad():
            support_emb = model(support_x)
            query_emb = model(query_x)
            logits = predict_protonet(support_emb, support_y, query_emb, n_way)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(1).cpu().numpy()
            true_labels = query_y.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(true_labels)

        probs = probs.cpu().numpy()
        for i in range(len(preds)):
            true_class = idx_to_class[true_labels[i]]
            pred_class = idx_to_class[preds[i]]
            confidence_score = probs[i][preds[i]]
            match = 1 if preds[i] == true_labels[i] else 0

            print(f"Real: {true_class} | Pred: {pred_class} | Score: {confidence_score:.3f} | Match: {match}")

    # Overall Metrics
    top1_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\nTest Results:")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Macro-averaged F1-Score: {macro_f1:.4f}")

# ========== Main Execution ========== #
if __name__ == "__main__":
    # Load model
    model = ProtoNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    # Load test dataset
    test_dataset = FewShotDataset(TEST_ROOT, transform, n_shot=N_SHOT, n_query=N_QUERY)

    # Run test
    test_protonet(model, test_dataset, DEVICE, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, episodes=TEST_EPISODES)
