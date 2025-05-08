import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
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

    node_type_list = []
    idx_to_gene = {}
    idx_to_disease = {}
    for idx, (n, d) in enumerate(graph.nodes(data=True)):
        node_type = d.get('node_type', None) or d.get('type', None)
        node_type_list.append(node_type)
        if node_type == 1:
            idx_to_gene[idx] = d.get('name', str(n))
        if node_type == 2:
            idx_to_disease[idx] = d.get('name', str(n))
    print("Idx -> [Gene, Disease] mapping done")

    # Use degree as a feature (recommended for large graphs)
    degrees = torch.tensor([val for (_, val) in graph.degree()], dtype=torch.float32).unsqueeze(1)
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

    # for k, v in pyg_data.items():
    #     print(f"{k}: {type(v)}")
    #     if isinstance(v, list):
    #         setattr(pyg_data, k, np.array(v))

    if hasattr(pyg_data, 'name'): del pyg_data.name

    if not hasattr(pyg_data, 'name'): print("Name removed from pyg_data")
    transform = RandomLinkSplit(is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(pyg_data)

    return train_data, val_data, test_data, [node_type_list, idx_to_gene, idx_to_disease]

# def get_gene_indices(data):
#     return [i for i, t in enumerate(data.node_type_list) if t == 1]

# def get_disease_indices(data):
#     return [i for i, t in enumerate(data.node_type_list) if t == 0]

# def get_gene_gene_edges(data, node_type_list=None):
#     """
#     Returns edge_index containing only gene-gene edges.
#     """
#     mask = []
#     if node_type_list is None:
#         raise ValueError("node_type_list not prepared from data. Run prepare_data with node_type attributes.")
#     for i in range(data.edge_index.shape[1]):
#         src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
#         if node_type_list[src] == 1 and node_type_list[dst] == 1:
#             mask.append(i)
#     if len(mask) == 0:
#         raise ValueError("No gene-gene edges found in the graph.")
#     return data.edge_index[:, mask]

def mini_batch_train_link_prediction(model: GraphiStasis,
                                     data: Data,
                                     optimizer: optim.Adam,
                                     criterion: nn.BCELoss,
                                     properties: list,
                                     epochs: int=10,
                                     batch_size: int=1024,
                                     num_neighbors: list=[10, 10],
                                     ) -> list:
    """
    Train the GraphiStasis GNN model for gene-gene link prediction using mini-batch training
    on a graph containing only gene nodes.
    """
    model.train()
    losses = []
    node_type_list, idx_to_gene, idx_to_disease = properties # node_type_list & idx_to_disease are not used in the core loop now

    gene_nodes_original_indices_list = list(idx_to_gene.keys())
    if not gene_nodes_original_indices_list:
        print("Warning: No gene nodes found in 'idx_to_gene'. Cannot train.")
        return losses

    gene_nodes_original_indices = torch.tensor(gene_nodes_original_indices_list, dtype=torch.long)

    # Ensure original data tensors are on a consistent device for subgraph operation
    # and that gene_nodes_original_indices is also on that device.
    # Assuming data.x and data.edge_index are on the same device.
    original_data_device = data.x.device
    if data.edge_index.device != original_data_device:
        # This case should ideally not happen if Data object is consistent
        data.edge_index = data.edge_index.to(original_data_device)
    
    gene_nodes_original_indices = gene_nodes_original_indices.to(original_data_device)

    # Filter out original indices not present in data.num_nodes (e.g., if idx_to_gene is stale)
    valid_gene_indices_mask = gene_nodes_original_indices < data.num_nodes
    gene_nodes_original_indices = gene_nodes_original_indices[valid_gene_indices_mask]

    if gene_nodes_original_indices.numel() == 0:
        print("Warning: No valid gene nodes remaining after filtering. Cannot train.")
        return losses
        
    # Create a new Data object containing only gene nodes and their edges
    gene_only_data = data.subgraph(gene_nodes_original_indices)

    if gene_only_data.num_nodes == 0:
        print("Warning: Gene-only subgraph is empty. Cannot train.")
        return losses

    # NeighborLoader expects input_nodes to be relative to the data it's given
    # and data to be on CPU.
    loader = NeighborLoader(
        gene_only_data.cpu(), 
        input_nodes=None, # Samples all nodes from gene_only_data
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True
    )

    model_device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
            if batch.num_nodes == 0: # Should not happen if gene_only_data is not empty
                continue

            optimizer.zero_grad() # Clear gradients for this batch

            batch = batch.to(model_device)
            z = model(batch.x, batch.edge_index)

            # All edges in the batch are positive gene-gene edges
            pos_edge_index = batch.edge_index
            if pos_edge_index.shape[1] == 0: # No positive edges in this batch
                # If there are nodes, but no edges, we might still want to learn something (e.g. predict no links)
                # or skip if loss calculation requires positive edges.
                # For now, if no positive edges, we can't form positive samples.
                if batch.num_nodes >= 2: # Check if negative sampling is possible
                    batch_local_node_indices = list(range(batch.num_nodes))
                    all_possible_pairs_in_batch = set(itertools.combinations(batch_local_node_indices, 2))
                    neg_pairs_local = list(all_possible_pairs_in_batch) # All pairs are negative
                    np.random.shuffle(neg_pairs_local)
                    # Sample a reasonable number of negatives, e.g., batch_size or a fixed number
                    neg_pairs_local = neg_pairs_local[:batch_size] 
                    if not neg_pairs_local:
                        continue
                    neg_edge_index = torch.tensor(neg_pairs_local, dtype=torch.long, device=z.device).t().contiguous()
                    neg_scores = model.decode(z, neg_edge_index)
                    
                    scores = neg_scores
                    labels = torch.zeros(neg_scores.size(0), device=scores.device)
                else: # Not enough nodes for any pairs
                    continue
            else:
                pos_scores = model.decode(z, pos_edge_index)

                # Negative sampling within the batch (all nodes are genes)
                batch_local_node_indices = list(range(batch.num_nodes))
                
                if batch.num_nodes < 2: # Not enough nodes to form pairs beyond existing
                    scores = pos_scores
                    labels = torch.ones(pos_scores.size(0), device=scores.device)
                else:
                    all_possible_pairs_in_batch = set(itertools.combinations(batch_local_node_indices, 2))
                    
                    existing_edges_in_batch = set()
                    for i in range(pos_edge_index.shape[1]):
                        u, v = pos_edge_index[0, i].item(), pos_edge_index[1, i].item()
                        existing_edges_in_batch.add(tuple(sorted((u, v))))
                    
                    neg_pairs_local = list(all_possible_pairs_in_batch - existing_edges_in_batch)
                    np.random.shuffle(neg_pairs_local)
                    # Sample as many negatives as positives, or up to a max
                    neg_pairs_local = neg_pairs_local[:pos_edge_index.shape[1]] 
                    
                    if not neg_pairs_local and pos_scores.numel() == 0: # No positive and no negative
                        continue
                    elif not neg_pairs_local: # Only positive scores
                        scores = pos_scores
                        labels = torch.ones(pos_scores.size(0), device=scores.device)
                    else: # Both positive and negative scores
                        neg_edge_index = torch.tensor(neg_pairs_local, dtype=torch.long, device=z.device).t().contiguous()
                        neg_scores = model.decode(z, neg_edge_index)
                        scores = torch.cat([pos_scores, neg_scores])
                        labels = torch.cat([
                            torch.ones(pos_scores.size(0), device=scores.device),
                            torch.zeros(neg_scores.size(0), device=scores.device)
                        ])
            
            if scores.numel() == 0: # No scores to compute loss on
                continue

            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / (batch_idx + 1) if (batch_idx + 1) > 0 else 0 # len(loader) might be 0 if gene_only_data is too small
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    return losses

def test_link_prediction(model: GraphiStasis,
                         data: Data,
                         properties: list,
                         batch_size=1024,
                         num_neighbors=[10, 10],
                         test_epochs=1): # test_epochs is usually 1 for full evaluation
    """
    Evaluate the GraphiStasis GNN model for gene-gene link prediction using mini-batch inference
    on a graph containing only gene nodes.
    Returns ROC AUC and average precision.
    """
    model.eval()
    node_type_list, idx_to_gene, idx_to_disease = properties # node_type_list & idx_to_disease are not used

    gene_nodes_original_indices_list = list(idx_to_gene.keys())
    if not gene_nodes_original_indices_list:
        print("Warning: No gene nodes found in 'idx_to_gene'. Cannot test.")
        return {'roc_auc': 0.0, 'avg_precision': 0.0}

    gene_nodes_original_indices = torch.tensor(gene_nodes_original_indices_list, dtype=torch.long)
    
    original_data_device = data.x.device
    if data.edge_index.device != original_data_device:
        data.edge_index = data.edge_index.to(original_data_device)
    gene_nodes_original_indices = gene_nodes_original_indices.to(original_data_device)

    valid_gene_indices_mask = gene_nodes_original_indices < data.num_nodes
    gene_nodes_original_indices = gene_nodes_original_indices[valid_gene_indices_mask]

    if gene_nodes_original_indices.numel() == 0:
        print("Warning: No valid gene nodes remaining after filtering. Cannot test.")
        return {'roc_auc': 0.0, 'avg_precision': 0.0}

    gene_only_data = data.subgraph(gene_nodes_original_indices)

    if gene_only_data.num_nodes == 0:
        print("Warning: Gene-only subgraph is empty. Cannot test.")
        return {'roc_auc': 0.0, 'avg_precision': 0.0}

    loader = NeighborLoader(
        gene_only_data.cpu(),
        input_nodes=None, # Samples all nodes
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False 
    )

    all_y_true = []
    all_y_scores = []
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(test_epochs): 
            for batch in tqdm.tqdm(loader, desc="Testing"):
                if batch.num_nodes == 0:
                    continue

                batch = batch.to(model_device)
                z = model(batch.x, batch.edge_index)

                pos_edge_index = batch.edge_index
                pos_scores = torch.tensor([], device=z.device) # Initialize as empty
                
                if pos_edge_index.shape[1] > 0:
                    pos_scores = torch.sigmoid(model.decode(z, pos_edge_index))

                neg_scores = torch.tensor([], device=z.device) # Initialize as empty
                if batch.num_nodes >= 2:
                    batch_local_node_indices = list(range(batch.num_nodes))
                    all_possible_pairs_in_batch = set(itertools.combinations(batch_local_node_indices, 2))
                    
                    existing_edges_in_batch = set()
                    if pos_edge_index.shape[1] > 0:
                        for i in range(pos_edge_index.shape[1]):
                            u, v = pos_edge_index[0, i].item(), pos_edge_index[1, i].item()
                            existing_edges_in_batch.add(tuple(sorted((u, v))))

                    neg_pairs_local = list(all_possible_pairs_in_batch - existing_edges_in_batch)
                    np.random.shuffle(neg_pairs_local)
                    
                    # For testing, it's common to evaluate on all or a larger sample of negatives.
                    # Here, we sample as many as positives for consistency, or up to batch_size if no positives.
                    num_neg_samples = pos_edge_index.shape[1] if pos_edge_index.shape[1] > 0 else batch_size
                    neg_pairs_local = neg_pairs_local[:num_neg_samples]
                    
                    if neg_pairs_local:
                        neg_edge_index = torch.tensor(neg_pairs_local, dtype=torch.long, device=z.device).t().contiguous()
                        neg_scores = torch.sigmoid(model.decode(z, neg_edge_index))

                if pos_scores.numel() > 0 or neg_scores.numel() > 0:
                    y_true_batch = torch.cat([
                        torch.ones(pos_scores.size(0), device=pos_scores.device),
                        torch.zeros(neg_scores.size(0), device=neg_scores.device)
                    ])
                    y_scores_batch = torch.cat([pos_scores, neg_scores])

                    all_y_true.append(y_true_batch.cpu())
                    all_y_scores.append(y_scores_batch.cpu())

    if not all_y_true:
        print("No valid batches processed for evaluation. Returning zero metrics.")
        return {'roc_auc': 0.0, 'avg_precision': 0.0}

    all_y_true_np = torch.cat(all_y_true).numpy()
    all_y_scores_np = torch.cat(all_y_scores).numpy()

    if len(all_y_true_np) == 0:
        print("Evaluation resulted in no samples. Returning zero metrics.")
        return {'roc_auc': 0.0, 'avg_precision': 0.0}
    
    unique_labels = np.unique(all_y_true_np)
    if len(unique_labels) < 2:
        print(f"Warning: Evaluation data contains only one class ({unique_labels}). ROC AUC and Avg Precision are undefined or misleading.")
        # Return 0.5 for ROC AUC if all labels are the same (chance level for a common interpretation)
        # Avg precision is more complex; 1.0 if all are positive and predicted as positive, 0.0 if all are negative.
        # For simplicity, returning 0.0 or a representative value.
        roc_auc_val = 0.5 
        avg_precision_val = float(unique_labels[0]) if len(unique_labels) == 1 and unique_labels[0] == 1 and np.all(all_y_scores_np > 0.5) else 0.0 # crude
        return {'roc_auc': roc_auc_val, 'avg_precision': avg_precision_val}

    roc_auc = roc_auc_score(all_y_true_np, all_y_scores_np)
    # precision_recall_curve can error if only one class in y_true or if y_scores is constant.
    try:
        precision, recall, _ = precision_recall_curve(all_y_true_np, all_y_scores_np)
        # trapz can fail if precision or recall are empty or have single values.
        avg_precision = np.trapz(recall, precision) if len(recall) > 1 and len(precision) > 1 else 0.0
        if not np.isfinite(avg_precision): avg_precision = 0.0 # Handle NaN/inf from trapz
    except ValueError:
        print("Warning: Could not compute precision-recall curve. Returning 0 for avg_precision.")
        avg_precision = 0.0


    return {'roc_auc': roc_auc, 'avg_precision': avg_precision}

def visualize_loss(losses, title="Loss Curve", xlabel="Epochs", ylabel="Loss"):
    """
    Visualize the training loss over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def predict_gene_gene_links(model: GraphiStasis, data, threshold=0.5, topk=None):
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