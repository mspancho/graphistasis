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
import pickle

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
        # Get edge mask and node feature mask for the given edge
        edge_mask, node_feat_mask = explainer.explain_link(edge, data.x, data.edge_index)
        if visualize: # Visualize the explanation
            explainer.visualize_subgraph(edge, data.edge_index, edge_mask, y=None, threshold=None)
        return edge_mask, node_feat_mask

def prepare_data(graph: nx.Graph) -> Data:
    """
    Prepare the data for GNN link prediction.
    Returns a PyTorch Geometric Data object.
    """

    # Unpickle embedding dict file (read-binary mode)
    with open("genept/GenePT_gene_embedding_ada_text.pickle", "rb") as f:
        gene_embedding_dict = pickle.load(f)

    # Convert NetworkX graph to PyTorch Geometric Data object
    graph = nx.convert_node_labels_to_integers(graph)
    pyg_data = pyg.utils.from_networkx(graph)

    # Keep track of graph attributes
    node_type_list = []
    idx_to_gene = {}
    idx_to_disease = {}
    embedding_dim = len(next(iter(gene_embedding_dict.values())))
    embeddings = []
    for idx, (n, d) in enumerate(graph.nodes(data=True)):
        node_type = d.get('node_type', None) or d.get('type', None)
        node_type_list.append(node_type)
        node_name = d.get('name', str(n))
        if node_type == 1:
            idx_to_gene[idx] = node_name
            emb = gene_embedding_dict.get(node_name, np.zeros(embedding_dim))
        if node_type == 2:
            idx_to_disease[idx] = node_name
            emb = np.zeros(embedding_dim)
        embeddings.append(emb)
    print("Idx -> [Gene, Disease] mapping done")
    pyg_data.x = torch.tensor(np.stack(embeddings), dtype=torch.float32)
    print("After setting embeddings, x shape:", pyg_data.x.shape)

    # Use degree as a feature bc large graph
    # degrees = torch.tensor([val for (_, val) in graph.degree()], dtype=torch.float32).unsqueeze(1)
    # pyg_data.x = degrees
    # print("After setting degrees, x shape:", pyg_data.x.shape)

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

    if hasattr(pyg_data, 'name'): del pyg_data.name

    if not hasattr(pyg_data, 'name'): print("Name removed from pyg_data")
    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(pyg_data)

    return train_data, val_data, test_data, [node_type_list, idx_to_gene, idx_to_disease]

def train_one_epoch(model: GraphiStasis,
                    train_data: Data, # Full training data
                    optimizer: optim.Adam,
                    criterion: nn.Module,
                    device: torch.device) -> float:
    """
    Trains the model for one epoch on the full training graph.
    """
    model.train()
    optimizer.zero_grad()

    # Move data to device
    # train_data should already be on CPU from prepare_data, RandomLinkSplit
    x = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device) # Message-passing edges
    edge_label_index = train_data.edge_label_index.to(device) # Supervision edges
    edge_label = train_data.edge_label.to(device) # Supervision labels

    # GNN forward pass using all training edges for message passing
    node_embeddings = model(x, edge_index)

    # Decode scores for the supervision links
    pred_scores = model.decode(node_embeddings, edge_label_index)
    
    # Compute loss
    loss = criterion(pred_scores, edge_label)
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate_link_predictor(model: GraphiStasis, 
                            data_split: Data, 
                            train_graph_edge_index: torch.Tensor, 
                            device: torch.device,
                            desc: str = "Evaluating", # Description for tqdm
                            eval_batch_size: int = 2048 # Batch size for evaluating links
                            ) -> dict:
    """
    Evaluates the model on a given data split (validation or test).
    Uses the training graph structure for generating node embeddings.
    Includes a progress bar for link decoding.
    """
    model.eval()

    # Generate embeddings for ALL nodes using the full feature set (data_split.x)
    # and the message passing edges from the TRAINING graph
    all_node_embeddings = model(data_split.x.to(device), train_graph_edge_index.to(device))

    # Decode scores for the supervision links in the current data_split
    # data_split.edge_label_index contains (+) and (-) links to evaluate
    
    y_scores_list = []
    y_true_list = []

    num_links_to_eval = data_split.edge_label_index.shape[1]
    
    # Iterate over edge_label_index in chunks for progress bar
    for i in tqdm.tqdm(range(0, num_links_to_eval, eval_batch_size), desc=desc, leave=False):
        batch_edge_label_index = data_split.edge_label_index[:, i:i+eval_batch_size].to(device)
        batch_edge_label = data_split.edge_label[i:i+eval_batch_size] # Corresponding labels

        # Skip empty batches
        if batch_edge_label_index.shape[1] == 0:
            continue

        # Decode scores for the current batch of links
        batch_pred_scores_logits = model.decode(all_node_embeddings, batch_edge_label_index)
        batch_pred_scores_sigmoid = torch.sigmoid(batch_pred_scores_logits)
        
        y_scores_list.append(batch_pred_scores_sigmoid.cpu())
        y_true_list.append(batch_edge_label.cpu())

    if not y_true_list: # No links were evaluated
        print(f"Warning: No links evaluated in {desc}. Returning default metrics.")
        return {'roc_auc': 0.0, 'avg_precision': 0.0}

    # Concatenate all batches
    true_labels = torch.cat(y_true_list).numpy()
    pred_scores_np = torch.cat(y_scores_list).numpy()

    # Check if true_labels contains only one class
    if len(np.unique(true_labels)) < 2:
        roc_auc = 0.5 
        avg_precision = np.mean(true_labels) if len(true_labels) > 0 else 0.0
        print(f"Warning: Evaluation data in '{desc}' contains only one class ({np.unique(true_labels)}). Metrics might be misleading.")
    else: # Normal eval
        roc_auc = roc_auc_score(true_labels, pred_scores_np)
        precision, recall, _ = precision_recall_curve(true_labels, pred_scores_np)
        avg_precision = np.trapz(recall, precision) if len(recall) > 1 and len(precision) > 1 else 0.0
        if not np.isfinite(avg_precision): avg_precision = 0.0

    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'y_true': true_labels,
        'y_scores': pred_scores_np
    }


def run_training_pipeline(
    model_class: GraphiStasis,
    model_args: dict,
    train_data: Data, # From RandomLinkSplit
    val_data: Data,   # From RandomLinkSplit
    test_data: Data,  # From RandomLinkSplit
    epochs: int = 50,
    learning_rate: float = 0.01,
    early_stopping_patience: int = 10, # Set to None to disable
    eval_batch_size: int = 2048 # Batch size for evaluate_link_predictor
    ) -> tuple:
    """
    Main pipeline to train, validate, and test the GNN link predictor using full-graph training.
    """
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Ensure model input dims match the data
    if 'in_channels' not in model_args and hasattr(train_data, 'num_node_features'):
        model_args['in_channels'] = train_data.num_node_features
    elif 'in_channels' not in model_args:
        # If using other features (e.g. GenePT), this needs to be accurate
        num_features = train_data.x.shape[1] if train_data.x is not None and train_data.x.dim() > 1 else 1
        print(f"Warning: 'in_channels' not specified in model_args. Setting to {num_features} based on train_data.x.shape.")
        model_args['in_channels'] = num_features

    # Initialize model, optimizer, and loss fn
    model = model_class(**model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss() 

    # Initialize early stopping variables
    best_val_metric = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_roc_auc': [], 'val_avg_precision': []}

    # edge_index for message passing during eval should be from training graph
    # train_data.edge_index as returned by RandomLinkSplit
    # check correct device for evaluate_link_predictor
    train_graph_message_passing_edges_eval = train_data.edge_index # moved to device in evaluate_link_predictor

    # Main epoch loop with overall progress bar
    for epoch in tqdm.tqdm(range(1, epochs + 1), desc="Overall Epochs"):
        # Pass the full train_data to train_one_epoch
        # train_data moved to device inside train_one_epoch
        avg_train_loss = train_one_epoch(model, train_data, optimizer, criterion, device)
        history['train_loss'].append(avg_train_loss)
        
        # val_data components moved to device inside evaluate_link_predictor
        val_metrics = evaluate_link_predictor(model, val_data, train_graph_message_passing_edges_eval, device, desc="Validating", eval_batch_size=eval_batch_size)
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_avg_precision'].append(val_metrics['avg_precision'])

        # Print epoch summary
        print(f"\nEpoch {epoch:03d} Summary: Train Loss: {avg_train_loss:.4f}, "
              f"Val ROC AUC: {val_metrics['roc_auc']:.4f}, Val Avg Precision: {val_metrics['avg_precision']:.4f}")

        # Early stopping check
        current_val_metric = val_metrics['roc_auc'] 
        if early_stopping_patience is not None:
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                epochs_no_improve = 0
                # torch.save(model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {early_stopping_patience} epochs with no improvement.")
                break
    
    # if early_stopping_patience is not None and os.path.exists('best_model.pth'):
    #     model.load_state_dict(torch.load('best_model.pth'))
    #     print("\nLoaded best model for final testing.")

    print("\nStarting final testing...")
    # test_data components moved to device inside evaluate_link_predictor
    test_metrics = evaluate_link_predictor(model, test_data, train_graph_message_passing_edges_eval, device, desc="Testing", eval_batch_size=eval_batch_size)
    print(f"\nFinal Test Metrics: ROC AUC: {test_metrics['roc_auc']:.4f}, Avg Precision: {test_metrics['avg_precision']:.4f}")

    return model, history, test_metrics

def visualize_loss(history, title="Loss Curve", xlabel="Epochs", ylabel="Loss", save_path=None):
    """
    Visualize the training loss over epochs.
    Accepts the 'history' dict returned by run_training_pipeline.
    If save_path is provided, saves the plot to that file.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_roc_auc' in history:
        plt.plot(history['val_roc_auc'], label='Val ROC AUC', color='orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def predict_gene_gene_links(model: GraphiStasis, data, node_type_list, idx_to_gene, threshold=0.5, topk=None, device='cpu', save_path=None):
    """
    Predict gene-gene interactions only.
    Returns a list of (gene1_name, gene2_name, score) tuples above the threshold or topk highest.
    If save_path is provided, saves a histogram of the top scores to that file.
    """
    model.eval()
    gene_indices = [i for i, t in enumerate(node_type_list) if t == 1]
    # get all pairs of gene indices
    pairs = list(itertools.combinations(gene_indices, 2))
    if not pairs:
        return []
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous().to(device)

    with torch.no_grad(): # no gradient for inference
        # Move data to device
        z = model(data.x.to(device), data.edge_index.to(device))
        scores = torch.sigmoid(model.decode(z, edge_index))

        # Filter by threshold or topk
        if topk is not None:
            # Get topk scores
            top_scores, top_idx = torch.topk(scores, topk)
            # Get corresponding edge indices
            selected = [(edge_index[0, i].item(), edge_index[1, i].item(), top_scores[i].item()) for i in range(topk)]
            # Convert to numpy for plotting 
            plot_scores = top_scores.cpu().numpy()
        else:
            # Filter by threshold
            mask = scores > threshold
            # Get indices of edges above threshold
            selected = [(edge_index[0, i].item(), edge_index[1, i].item(), scores[i].item()) for i in mask.nonzero(as_tuple=False).flatten()]
            plot_scores = scores[mask].cpu().numpy()

        # Map indices to gene names if available
        if idx_to_gene:
            selected = [(idx_to_gene.get(i, i), idx_to_gene.get(j, j), score) for i, j, score in selected]

        # Optionally save histogram of the predicted scores
        if save_path and len(plot_scores) > 0:
            plt.figure()
            plt.hist(plot_scores, bins=20, color='skyblue')
            plt.title("Predicted Gene-Gene Link Scores")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.grid()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    return selected

def plot_roc_curve_link(y_true, y_scores, title='ROC Curve (Link Prediction)', save_path=None):
    """
    Plot the ROC curve for link prediction.
    If save_path is provided, saves the plot to that file.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_precision_recall_link(y_true, y_scores, title='Precision-Recall Curve (Link Prediction)', save_path=None):
    """
    Plot the precision-recall curve for link prediction.
    If save_path is provided, saves the plot to that file.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()