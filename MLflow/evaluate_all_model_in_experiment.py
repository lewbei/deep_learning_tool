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

# -------------------------------
# Helper Function: Plot Confusion Matrix
# -------------------------------
def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png",
                          title="Confusion Matrix", cmap=plt.cm.Blues):
    """
    Plots the confusion matrix as an image and saves it.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------------------
# MLflow Setup
# -------------------------------
mlflow.enable_system_metrics_logging()
mlflow.set_tracking_uri("file:///C:\\Users\\Qwerty\\mlruns")  # Adjust if necessary
experiment_name = "noise_classification_experiment_dataset1"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id  # e.g. "546529494608093044"

# -------------------------------
# Test Data Configuration
# -------------------------------
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
# -------------------------------
# Retrieve Runs and Build Model URIs
# -------------------------------
runs_df = mlflow.search_runs(experiment_ids=[experiment_id])
print(f"Found {len(runs_df)} run(s) in experiment '{experiment_name}'.")

# Define the local tracking directory (remove file:/// from the URI)
tracking_dir = r"C:\Users\Qwerty\mlruns"

for idx, row in runs_df.iterrows():
    run_id = row["run_id"]

    # Try using the logged parameter "model" as the artifact folder name.
    artifact_folder = row.get("params.model", "")
    # Construct the expected local path for that artifact folder:
    run_artifact_dir = os.path.join(tracking_dir, experiment_id, run_id, "artifacts")
    candidate_path = os.path.join(run_artifact_dir, artifact_folder, "data", "model.pth")
    if not (artifact_folder and os.path.exists(candidate_path)):
        # If the folder isn't valid, search the run's artifact folder for a folder that contains data/model.pth
        found = False
        if os.path.isdir(run_artifact_dir):
            for folder in os.listdir(run_artifact_dir):
                candidate = os.path.join(run_artifact_dir, folder, "data", "model.pth")
                if os.path.exists(candidate):
                    artifact_folder = folder
                    found = True
                    break
        if not found:
            print(f"Skipping run {run_id}: no valid model artifact folder found.")
            continue

    model_uri = f"runs:/{run_id}/{artifact_folder}"
    print(f"\nEvaluating model from run {run_id} with artifact folder: '{artifact_folder}'")
    try:
        model = mlflow.pytorch.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model {model_uri}: {e}")
        continue

    model = model.to(device)
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # If your model returns a tuple (e.g. (ordinal_logits, class_logits)), use classification logits.
            if isinstance(outputs, tuple):
                _, logits = outputs
            else:
                logits = outputs
            _, preds = torch.max(logits, 1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    # -------------------------------
    # Compute Evaluation Metrics
    # -------------------------------
    accuracy = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average='weighted')
    precision = precision_score(all_true, all_pred, average='weighted')
    recall = recall_score(all_true, all_pred, average='weighted')
    cm = confusion_matrix(all_true, all_pred)

    print(f"Results for model {model_uri}:")
    print("  Accuracy: {:.2f}%".format(accuracy * 100))
    print("  F1 Score: {:.4f}".format(f1))
    print("  Precision: {:.4f}".format(precision))
    print("  Recall: {:.4f}".format(recall))
    print("  Confusion Matrix:\n", cm)

    # -------------------------------
    # Log Evaluation Metrics to MLflow
    # -------------------------------
    # Retrieve the model name from parameters (if available)
    model_name = row.get("params.model", "unknown_model")
    with mlflow.start_run(run_name=f"Evaluation_{model_name}_{run_id}") as eval_run:
        mlflow.set_tag("mlflow.runName", f"Evaluation_{model_name}_{run_id}")
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Save confusion matrix as JSON and log as an artifact.
        cm_json_path = f"confusion_matrix_{eval_run.info.run_id}.json"
        with open(cm_json_path, "w") as f:
            json.dump(cm.tolist(), f)
        mlflow.log_artifact(cm_json_path)
        
        # Save the confusion matrix as an image and log as an artifact.
        class_names = [f"{i/1000:.4f}" for i in range(11)]
        cm_img_path = f"confusion_matrix_{eval_run.info.run_id}.png"
        plot_confusion_matrix(all_true, all_pred, class_names, save_path=cm_img_path)
        mlflow.log_artifact(cm_img_path)
        
        # Save the confusion matrix as a pickle file (editable) and log as an artifact.
        cm_pickle_path = f"confusion_matrix_{eval_run.info.run_id}.pkl"
        with open(cm_pickle_path, "wb") as f:
            pickle.dump(cm, f)
        mlflow.log_artifact(cm_pickle_path)
