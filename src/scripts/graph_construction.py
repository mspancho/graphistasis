import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
# import pyg_lib as pgl
import torch_scatter as scatter
# import torch_sparse as sparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_tsv_schema(file_path="src/data/DG-Miner-tsv/mine-disease-gene.tsv") -> pd.DataFrame:
    """
    Import the DGMiner TSV file into a pandas DataFrame and return its schema.
    """
    dgm_df = pd.read_csv(file_path, sep='\t')
    print(dgm_df.head()) # Show first few rows
    print()
    
    # Get DataFrame schema
    schema = dgm_df.dtypes.reset_index()
    schema.columns = ['Column Name', 'Data Type']
    
    return schema

def disease_selection(dn: str, 
                      did: str, 
                      file_path="src/data/DG-Miner-tsv/mine-disease-gene.tsv") -> pd.DataFrame:
    """
    Import the DGMiner TSV file into a pandas DataFrame.
    """
    print("Reading in DGMiner TSV file as DataFrame object...")
    dgm_df = pd.read_csv(file_path, sep='\t')

    # Filter out rows with NaN in 'disease_name' or 'disease_id'
    print("Filtering out missing values...")
    dgm_df = dgm_df.dropna()
    
    # Get rid of "MESH:" prefix in disease names
    print("Cleaning disease names...")
    dgm_df['# Disease(MESH)'] = dgm_df['# Disease(MESH)'].str.replace('MESH:', '', regex=False)

    # Select relevant rows
    print(f"Selecting disease '{dn}' with ID '{did}'...")
    dgm_df = dgm_df[dgm_df['# Disease(MESH)'] == did]

    return dgm_df

def graph_construction(dgm_df: pd.DataFrame) -> Data:
    """
    Construct a graph from the DGMiner DataFrame.
    """
    # Combine all unique node names
    nodes = pd.unique(dgm_df[['# Disease(MESH)', 'Gene']].values.ravel())
    node2idx = {node: idx for idx, node in enumerate(nodes)}

    # Map node names to indices for edge_index
    edge_index = dgm_df[['# Disease(MESH)', 'Gene']].map(node2idx.get).values.T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create edge_index tensor
    data = Data(edge_index=edge_index)

    return data

def visualize_graph(data: Data):
    """
    Visualize the graph using NetworkX and Matplotlib.
    """
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().numpy())
    
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
    plt.show()


def main():
    # Test schema extraction
    schema = get_tsv_schema()
    print("Schema of the DGMiner TSV file:")
    print(schema)

    # Test disease selection
    disease_df = disease_selection('schizophrenia', 'D012559')
    print("\nSelected Disease DataFrame:")
    print(disease_df.head())
    print("\nFiltered Disease DataFrame:")
    print(disease_df.head())

    # Test graph construction
    graph_data = graph_construction(disease_df)
    print("\nConstructed Graph Data:")
    print(graph_data)

    # Visualize graph w/ NetworkX
    visualize_graph(graph_data)

if __name__ == "__main__":
    main()