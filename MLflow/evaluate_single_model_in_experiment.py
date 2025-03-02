import os

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
os.environ["MLFLOW_ENABLE_ROCM_MONITORING"] = "false"
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets, models
import mlflow
import mlflow.pytorch
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Config:
    img_size = 64
    batch_size = 32
    lr = 3e-4
    epochs = 100
    num_classes = 11  # classes 0...10 corresponding to noise levels 0.0000...0.0100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class PartialNoisyTrainDataset(Dataset):
    def __init__(self, base_dataset, indices, pre_transform=None, final_transform=None):
        self.base_subset = Subset(base_dataset, indices)
        self.pre_transform = pre_transform
        self.final_transform = final_transform
        # Define noise levels for clean images (excluding 0)
        self.aug_noise_levels = [i/1000 for i in range(1, 11)]
        # Partition into clean and non-clean images
        self.clean_indices = []
        self.non_clean_indices = []
        for i in range(len(self.base_subset)):
            path, _ = self.base_subset.dataset.samples[self.base_subset.indices[i]]
            folder = os.path.basename(os.path.dirname(path))
            fval = float(folder)
            if abs(fval - 0.0) < 1e-8:
                self.clean_indices.append(i)
            else:
                self.non_clean_indices.append(i)
        # Total length = non-clean (once) + clean (original + 10 augmented)
        self.total_len = len(self.non_clean_indices) + len(self.clean_indices) * (1 + len(self.aug_noise_levels))
        print("Clean images:", len(self.clean_indices))
        print("Non-clean images:", len(self.non_clean_indices))
        print("Total partial dataset length:", self.total_len)
    def __len__(self):
        return self.total_len
    def parse_folder_label(self, path):
        folder = os.path.basename(os.path.dirname(path))
        floatval = float(folder)
        label = int(round(floatval * 1000))
        return label
    def __getitem__(self, idx):
        nc_count = len(self.non_clean_indices)
        if idx < nc_count:
            real_idx = self.non_clean_indices[idx]
            image, _ = self.base_subset[real_idx]
            path, _ = self.base_subset.dataset.samples[self.base_subset.indices[real_idx]]
            label = self.parse_folder_label(path)
            if self.pre_transform:
                image = self.pre_transform(image)
            if self.final_transform:
                image = self.final_transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, label
        idx2 = idx - nc_count
        if idx2 < len(self.clean_indices):
            real_idx = self.clean_indices[idx2]
            image, _ = self.base_subset[real_idx]
            label = 0
            if self.pre_transform:
                image = self.pre_transform(image)
            if self.final_transform:
                image = self.final_transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, label
        idx3 = idx2 - len(self.clean_indices)
        c_i = idx3 // len(self.aug_noise_levels)
        n_i = idx3 % len(self.aug_noise_levels)
        real_idx = self.clean_indices[c_i]
        image, _ = self.base_subset[real_idx]
        noise_var = self.aug_noise_levels[n_i]
        label = int(round(noise_var * 1000))
        if self.pre_transform:
            image = self.pre_transform(image)
        np_img = np.array(image).astype(np.float32) / 255.0
        noise = np.random.randn(*np_img.shape) * np.sqrt(noise_var)
        noisy_np = np.clip(np_img + noise, 0, 1)
        if len(noisy_np.shape) == 2:
            noisy_pil = Image.fromarray((noisy_np * 255).astype(np.uint8), mode='L')
        else:
            noisy_pil = Image.fromarray((noisy_np * 255).astype(np.uint8))
        if self.final_transform:
            image = self.final_transform(noisy_pil)
        else:
            image = transforms.ToTensor()(noisy_pil)
        return image, label

class NoiseIndexValDataset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_subset = Subset(base_dataset, indices)
        self.transform = transform
    def parse_folder_label(self, path):
        folder = os.path.basename(os.path.dirname(path))
        floatval = float(folder)
        label = int(round(floatval * 1000))
        return label
    def __len__(self):
        return len(self.base_subset)
    def __getitem__(self, idx):
        image, _ = self.base_subset[idx]
        path, _ = self.base_subset.dataset.samples[self.base_subset.indices[idx]]
        label = self.parse_folder_label(path)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label

# Define dataset transformations
pre_transform = transforms.Resize((Config.img_size, Config.img_size))

post_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

clean_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Create datasets
root_dir = r"C:\Users\Qwerty\Downloads\NFFA-EUROPE Noisy v2"  # Set correct path
base_dataset = datasets.ImageFolder(root=root_dir, transform=None)
seed = 42
num_total = len(base_dataset)
train_size = int(0.6 * num_total)
val_size = int(0.2 * num_total)
test_size = num_total - train_size - val_size
indices = list(range(num_total))
train_indices, val_indices, test_indices = torch.utils.data.random_split(
    indices, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(seed)
)

train_dataset = PartialNoisyTrainDataset(base_dataset, train_indices, pre_transform, post_transform)
val_dataset = NoiseIndexValDataset(base_dataset, val_indices, transform=clean_transform)
test_dataset = NoiseIndexValDataset(base_dataset, test_indices, transform=clean_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

# Specify your model URI from MLflow
# Replace <RUN_ID> and <artifact_path> with your actual run ID and artifact path.
model_uri = r"file:///C:\Users\Qwerty\mlruns/719776159845325982/3dcc8f8111cf4d96bf14d1fe5440d98a/artifacts/NoiseNetV2"
model = mlflow.pytorch.load_model(model_uri)
model = model.to(device)
model.eval()

# Initialize lists to hold true labels and predictions
all_true, all_pred = [], []

# Run inference on the test set
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # If your model returns a tuple (e.g., (ordinal_logits, class_logits))
        if isinstance(outputs, tuple):
            _, logits = outputs
        else:
            logits = outputs
        _, preds = torch.max(logits, 1)
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())

# Compute evaluation metrics
accuracy = accuracy_score(all_true, all_pred)
f1 = f1_score(all_true, all_pred, average='weighted')
precision = precision_score(all_true, all_pred, average='weighted')
recall = recall_score(all_true, all_pred, average='weighted')
cm = confusion_matrix(all_true, all_pred)

# Print metrics to the console
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("F1 Score: {:.4f}".format(f1))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("Confusion Matrix:\n", cm)


# Log metrics and confusion matrix artifact into MLflow
with mlflow.start_run() as run:
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Save the confusion matrix as JSON and log as an artifact
    cm_path = "confusion_matrix.json"
    with open(cm_path, "w") as f:
        json.dump(cm.tolist(), f)
    mlflow.log_artifact(cm_path)

    print("Metrics and artifacts logged to MLflow run:", run.info.run_id)

