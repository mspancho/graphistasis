import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import networkx as nx
import tqdm
import itertools


class GraphiStasis(nn.Module):
    """
    GNN model for link prediction (binary edge existence) between gene nodes.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, conv_type='GCN'):
        super(GraphiStasis, self).__init__()
        if conv_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif conv_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels)
        elif conv_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        else:
            raise ValueError("Invalid conv_type")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        # Dot product decoder for link prediction
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def decode_all(self, z):
        # All possible pairs (for full adjacency prediction)
        prob_adj = torch.sigmoid(torch.matmul(z, z.t()))
        return prob_adj
    
    def explain_link(self, data, edge, epochs=200, visualize=True):
        """
        Use GNNExplainer to interpret the prediction for a specific edge (link) between two nodes.
        Args:
            data: PyG Data object
            edge: tuple (u, v) of node indices
            epochs: number of epochs for the explainer
            visualize: if True, plot the explanation subgraph
        Returns:
            edge_mask, node_feat_mask
        """
        explainer = GNNExplainer(self, epochs=epochs)
        edge_mask, node_feat_mask = explainer.explain_link(edge, data.x, data.edge_index)
        if visualize:
            explainer.visualize_subgraph(edge, data.edge_index, edge_mask, y=None, threshold=None)
        return edge_mask, node_feat_mask

def prepare_data(graph: nx.Graph) -> Data:
    """
    Prepare the data for GNN link prediction.
    Returns a PyTorch Geometric Data object.
    """
    graph = nx.convert_node_labels_to_integers(graph)
    pyg_data = pyg.utils.from_networkx(graph)

    # NOTE: One-hot encoding creates feature matrix too large for such a large graph
    # pyg_data.x = torch.eye(nx.number_of_nodes())
    # Build node_type list and idx_to_gene mapping
    node_type_list = []
    idx_to_gene = {}
    for idx, (n, d) in enumerate(graph.nodes(data=True)):
        node_type = d.get('node_type', None) or d.get('type', None)
        node_type_list.append(node_type)
        if node_type == 'gene':
            idx_to_gene[idx] = str(n)
    pyg_data.node_type_list = node_type_list
    pyg_data.idx_to_gene = idx_to_gene

    # Use degree as a feature (recommended for large graphs)
    degrees = torch.tensor([val for (node, val) in graph.degree()], dtype=torch.float32).unsqueeze(1)
    pyg_data.x = degrees
    print("After setting degrees, x shape:", pyg_data.x.shape)

    # Debug prints
    try:
        print("x.shape[0]:", pyg_data.x.shape[0])
        print("edge_index.max():", pyg_data.edge_index.max())
        if pyg_data.x.shape[0] > pyg_data.edge_index.max():
            print("Warning: Edge indices reference node indices not in feature matrix.")
        else:
            print("Aight ur chillin prolly")
    except Exception as e:
        print("Error in prepare_data print statements:", e)

    return pyg_data

def get_gene_gene_edges(data):
    """
    Returns edge_index containing only gene-gene edges.
    """
    mask = []
    node_type_list = getattr(data, 'node_type_list', None)
    if node_type_list is None:
        raise ValueError("node_type_list not found in data. Run prepare_data with node_type attributes.")
    for i in range(data.edge_index.shape[1]):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        if node_type_list[src] == 'gene' and node_type_list[dst] == 'gene':
            mask.append(i)
    if len(mask) == 0:
        raise ValueError("No gene-gene edges found in the graph.")
    return data.edge_index[:, mask]

def train_link_prediction(model, data, optimizer, criterion, epochs=100):
    """
    Train the GraphiStasis GNN model for link prediction.
    """
    model.train()
    losses = []
    from torch_geometric.utils import negative_sampling

    # Only gene-gene edges for training
    pos_edge_index = get_gene_gene_edges(data)

    for epoch in tqdm.trange(epochs, desc="Training"):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)  # Node embeddings

        # Positive edge scores (gene-gene only)
        pos_scores = model.decode(z, pos_edge_index)

        # Negative edge sampling (gene-gene only)
        gene_indices = [i for i, t in enumerate(data.node_type_list) if t == 'gene']
        # All possible gene-gene pairs (excluding self-loops and existing edges)
        all_gene_pairs = set(itertools.combinations(gene_indices, 2))
        existing_gene_edges = set(tuple(sorted((pos_edge_index[0, i].item(), pos_edge_index[1, i].item()))) for i in range(pos_edge_index.shape[1]))
        neg_pairs = list(all_gene_pairs - existing_gene_edges)
        # Sample negatives up to the number of positives
        np.random.shuffle(neg_pairs)
        neg_pairs = neg_pairs[:pos_edge_index.shape[1]]
        if len(neg_pairs) == 0:
            raise ValueError("No negative gene-gene pairs available for sampling.")
        neg_edge_index = torch.tensor(neg_pairs, dtype=torch.long).t().contiguous()
        neg_scores = model.decode(z, neg_edge_index)

        # Labels: 1 for positive, 0 for negative
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(pos_scores.size(0), device=scores.device),
            torch.zeros(neg_scores.size(0), device=scores.device)
        ])

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

def test_link_prediction(model, data, test_epochs=1):
    """
    Evaluate the GraphiStasis GNN model for link prediction.
    Returns ROC AUC and average precision.
    """
    model.eval()
    results = []
    from torch_geometric.utils import negative_sampling

    pos_edge_index = get_gene_gene_edges(data)
    gene_indices = [i for i, t in enumerate(data.node_type_list) if t == 'gene']
    all_gene_pairs = set(itertools.combinations(gene_indices, 2))
    existing_gene_edges = set(tuple(sorted((pos_edge_index[0, i].item(), pos_edge_index[1, i].item()))) for i in range(pos_edge_index.shape[1]))
    neg_pairs = list(all_gene_pairs - existing_gene_edges)
    np.random.shuffle(neg_pairs)
    neg_pairs = neg_pairs[:pos_edge_index.shape[1]]
    if len(neg_pairs) == 0:
        raise ValueError("No negative gene-gene pairs available for sampling.")
    neg_edge_index = torch.tensor(neg_pairs, dtype=torch.long).t().contiguous()

    with torch.no_grad():
        for _ in tqdm.trange(test_epochs, desc="Testing"):
            z = model(data.x, data.edge_index)
            pos_scores = torch.sigmoid(model.decode(z, pos_edge_index))
            neg_scores = torch.sigmoid(model.decode(z, neg_edge_index))

            y_true = torch.cat([
                torch.ones(pos_scores.size(0), device=pos_scores.device),
                torch.zeros(neg_scores.size(0), device=neg_scores.device)
            ])
            y_scores = torch.cat([pos_scores, neg_scores])

            roc_auc = roc_auc_score(y_true.cpu().numpy(), y_scores.cpu().numpy())
            precision, recall, _ = precision_recall_curve(y_true.cpu().numpy(), y_scores.cpu().numpy())
            avg_precision = np.trapz(precision, recall)

            results.append({'roc_auc': roc_auc, 'avg_precision': avg_precision})

    return {'roc_auc': roc_auc, 'avg_precision': avg_precision}

def predict_gene_gene_links(model, data, threshold=0.5, topk=None):
    """
    Predict gene-gene interactions only.
    Returns a list of (gene1_name, gene2_name, score) tuples above the threshold or topk highest.
    """
    model.eval()
    gene_indices = [i for i, t in enumerate(data.node_type_list) if t == 'gene']
    idx_to_gene = getattr(data, 'idx_to_gene', None)
    pairs = list(itertools.combinations(gene_indices, 2))
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()

    with torch.no_grad():
        z = model(data.x, data.edge_index)
        scores = torch.sigmoid(model.decode(z, edge_index))

        # Filter by threshold or topk
        if topk is not None:
            top_scores, top_idx = torch.topk(scores, topk)
            selected = [(edge_index[0, i].item(), edge_index[1, i].item(), top_scores[i].item()) for i in top_idx]
        else:
            mask = scores > threshold
            selected = [(edge_index[0, i].item(), edge_index[1, i].item(), scores[i].item()) for i in mask.nonzero(as_tuple=False).flatten()]

        # Map indices to gene names if available
        if idx_to_gene:
            selected = [(idx_to_gene.get(i, i), idx_to_gene.get(j, j), score) for i, j, score in selected]

    return selected

def plot_roc_curve_link(y_true, y_scores):
    """
    Plot the ROC curve for link prediction.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Link Prediction)')
    plt.show()

def plot_precision_recall_link(y_true, y_scores):
    """
    Plot the precision-recall curve for link prediction.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Link Prediction)')
    plt.show()