import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random 
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

def k_fold_train(dataset, model_class, k=5, epochs=10, batch_size=32, lr=1e-3, input_dim=256, hidden_dim=128):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_metrics = []

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold + 1} ---")
        
        # Split dataset
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        model = model_class(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        train_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds = model(x).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        # WE SHOULD MAKE THIS A HIGHER THRESHOLD
        bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
        metrics = {
            'accuracy': accuracy_score(all_labels, bin_preds),
            'precision': precision_score(all_labels, bin_preds),
            'recall': recall_score(all_labels, bin_preds),
            'f1': f1_score(all_labels, bin_preds),
            'roc_auc': roc_auc_score(all_labels, all_preds),
            'losses': train_losses
        }
        all_metrics.append(metrics)

    return all_metrics

def tune_hyperparams(dataset, input_dim, metric_to_optimize='accuracy'):
    from itertools import product
    import pandas as pd

    hidden_dims = [64, 128, 256]
    lrs = [1e-3, 5e-4]
    batch_sizes = [32, 64]
    
    tuning_results = []

    for hidden_dim, lr, batch_size in product(hidden_dims, lrs, batch_sizes):
        print(f"\nTesting hidden_dim={hidden_dim}, lr={lr}, batch_size={batch_size}")
        metrics = k_fold_train(
            dataset=dataset,
            model_class=MLP,
            k=3,
            epochs=5,  # use fewer epochs for tuning
            batch_size=batch_size,
            lr=lr,
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        avg_metrics = {
            'hidden_dim': hidden_dim,
            'lr': lr,
            'batch_size': batch_size,
            'accuracy': np.mean([m['accuracy'] for m in metrics]),
            'f1': np.mean([m['f1'] for m in metrics]),
            'precision': np.mean([m['precision'] for m in metrics]),
            'recall': np.mean([m['recall'] for m in metrics]),
            'roc_auc': np.mean([m['roc_auc'] for m in metrics]),
        }
        tuning_results.append(avg_metrics)

    # Convert to DataFrame for easier sorting/plotting
    results_df = pd.DataFrame(tuning_results)
    best_config = results_df.sort_values(by=metric_to_optimize, ascending=False).iloc[0]
    
    print(f"\nBest config by {metric_to_optimize.upper()}:")
    print(best_config)

    # Plot accuracy (or any other metric) by config
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='hidden_dim', y=metric_to_optimize, hue='batch_size')
    plt.title(f'{metric_to_optimize.upper()} by Hidden Dim and Batch Size')
    plt.ylabel(metric_to_optimize.upper())
    plt.xlabel("Hidden Dimension")
    plt.legend(title="Batch Size")
    plt.show()

    return best_config.to_dict(), results_df


def plot_metric_over_folds(metrics_list, metric_name):
    scores = [m[metric_name] for m in metrics_list]
    sns.lineplot(x=list(range(1, len(scores)+1)), y=scores)
    plt.xlabel('Fold')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} over K-Folds')
    plt.show()

def plot_losses(metrics_list):
    for i, m in enumerate(metrics_list):
        sns.lineplot(x=list(range(len(m['losses']))), y=m['losses'], label=f'Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs per Fold')
    plt.legend()
    plt.show()

def get_genept_embedding(gene_name, gene_embedding_dict):
    embedding_dim = len(next(iter(gene_embedding_dict.values())))
    emb = gene_embedding_dict.get(gene_name, np.zeros(embedding_dim))
    # Example: returns a random vector of size 128
    return emb

def get_data_for_dataset(tsv_path):
    # Read the TSV file into a DataFrame
    data = pd.read_csv(tsv_path, sep="\t", header=0)  # Skip the header row
        
    # Basic cleaning
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    # Add 'Label' column with all interactions having positive labels
    data['Label'] = 1 # set positive labels

    return data

class GenePairDataset(Dataset):
    def __init__(self, df, negative_ratio: float=1.0):
        # Unpickle embedding dict file (read-binary mode)
        with open("genept/GenePT_gene_embedding_ada_text.pickle", "rb") as f:
            self.gene_embedding_dict = pickle.load(f)

        print(f"Make sure your 'input_dim' for MLP is {len(next(iter(self.gene_embedding_dict.values()))) * 2}") # since you concatenate two embeddings

        self.data = df
        
        # NEGATIVE SAMPLING
        # Build a set of all positive pairs (unordered, so both (g1,g2) and (g2,g1) are covered)
        positive_pairs = set(tuple(sorted((row['Gene1'], row['Gene2']))) for _, row in self.data.iterrows())

        # all_genes = self.data['Gene1'].cat.categories.tolist()  # Unique genes
        all_genes = list(pd.unique(self.data['Gene1'].tolist() + self.data['Gene2'].tolist()))

        # Generate negative pairs
        num_negatives = int(self.data.shape[0] * negative_ratio)
        neg_pairs = set()
        attempts = 0
        max_attempts = num_negatives * 10

        while len(neg_pairs) < num_negatives and attempts < max_attempts:
            g1, g2 = random.sample(all_genes, 2)
            pair = tuple(sorted((g1, g2)))
            if pair not in positive_pairs and pair not in neg_pairs:
                neg_pairs.add(pair)
            attempts += 1

        neg_rows = [{'Gene1': g1, 'Gene2': g2, 'Label': 0} for g1, g2 in neg_pairs]
        neg_df = pd.DataFrame(neg_rows)

        # Concatenate positive and negative samples
        self.data = pd.concat([self.data, neg_df], ignore_index=True)

        # Map gene names to embeddings up front
        gene_to_embed = {gene: get_genept_embedding(gene, self.gene_embedding_dict) for gene in all_genes}

        # Create embeddings for Gene1 and Gene2
        self.gene1_embeds = np.stack(self.data['Gene1'].map(gene_to_embed))
        self.gene2_embeds = np.stack(self.data['Gene2'].map(gene_to_embed))
        self.labels = self.data['Label'].astype(np.float32).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # row = self.data.iloc[idx]
        # g1 = row['Gene1']
        # g2 = row['Gene2']
        # emb1 = get_genept_embedding(g1, self.gene_embedding_dict)
        # emb2 = get_genept_embedding(g2, self.gene_embedding_dict)
        # pair_embed = np.concatenate([emb1, emb2])
        # label = self.labels[idx]
        # return torch.tensor(pair_embed, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        pair_embed = np.concatenate([self.gene1_embeds[idx], self.gene2_embeds[idx]])
        label = self.labels[idx]
        return torch.tensor(pair_embed, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=256): # Reflects hyperparameter tuning
        super().__init__()
        
        self.ffm = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.ffm(x).squeeze(-1)

def train(model, dataloader, epochs=10, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from tqdm import tqdm

def train_and_evaluate(model, train_loader, test_loader, epochs, lr, roc_auc_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = (preds >= 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(correct/total):.4f}"})
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={correct/total:.4f}")

    # Evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            preds = model(x).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, bin_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    # Plot ROC-AUC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC-AUC Curve')
    plt.legend(loc="lower right")
    plt.savefig(roc_auc_path)
    plt.close()

def predict_epistatic(model, gene1, gene2, embedding_dict):
    """
    Predict whether two genes are epistatic based on their embeddings.

    Args:
        model (nn.Module): The trained MLP model.
        gene1 (str): The name of the first gene.
        gene2 (str): The name of the second gene.
        embedding_dict (dict): Dictionary containing gene embeddings.

    Returns:
        float: The predicted probability of the two genes being epistatic.
    """
    # Get embeddings for the two genes
    emb1 = get_genept_embedding(gene1, embedding_dict)
    emb2 = get_genept_embedding(gene2, embedding_dict)

    # Concatenate the embeddings
    pair_embed = np.concatenate([emb1, emb2])

    # Convert to tensor
    pair_embed_tensor = torch.tensor(pair_embed, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Move to device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pair_embed_tensor = pair_embed_tensor.to(device)

    # Set model to evaluation mode
    model.eval()

    # Predict
    with torch.no_grad():
        prediction = model(pair_embed_tensor).item()

    return prediction