import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
# import pyg_lib as pgl
# import torch_sparse as sparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_tsv_schema(file_path="data/mine-disease-gene.tsv") -> pd.DataFrame:
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

def filter_data(disease_name=None, 
                      disease_id=None, 
                      file_path="data/DG-Miner_miner-disease-gene.tsv",
                      save=False) -> pd.DataFrame:
    """
    Import the DGMiner TSV file into a pandas DataFrame.
    """
    print("Reading in DGMiner TSV file as DataFrame object...")
    df = pd.read_csv(file_path, sep='\t')

    # Filter out rows with NaN in 'disease_name' or 'disease_id'
    print("Filtering out missing values...")
    df = df.dropna()
    
    # Get rid of "MESH:" prefix in disease names
    print("Cleaning disease names...")
    df['# Disease(MESH)'] = df['# Disease(MESH)'].str.replace('MESH:', '', regex=False)

    if disease_id:
        # Filter by disease ID
        print(f"Filtering by disease ID: {disease_id}")
        df = df[df['# Disease(MESH)'] == disease_id]

    if disease_name:
        print(f"Results for: {disease_name}")
        if save:
            # Save the filtered DataFrame to a TSV file
            print(f"Saving filtered DataFrame to {disease_name}_{disease_id}.tsv")
            df.to_csv(f'data/{disease_name}_{disease_id}.tsv', sep='\t', index=False)
    return df

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

def visualize_graph(data: Data, save_path=None):
    """
    Visualize the graph using NetworkX and Matplotlib.
    """
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().numpy())
    
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
    plt.show()

    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")

def explore_data(disease_df: pd.DataFrame):
    """
    Explore the data by printing the first few rows and the number of unique diseases.
    """
    print("Exploring data...")
    print(disease_df.head())
    print(f"Number of unique diseases: {disease_df['# Disease(MESH)'].nunique()}")
    print(f"Number of unique genes: {disease_df['Gene'].nunique()}")
    print(f"Number of disease-gene pairs: {len(disease_df)}")


def main():
    data_present = input("Have you downloaded the data file locally? If so, type 'yes' or 'y'.\n")# Test schema extraction
    
    disease_name = 'schizophrenia'
    disease_id = 'D012559'

    if data_present.lower() in ['yes', 'y']:
        print("Proceeding with schema extraction...")
        schema = get_tsv_schema()
        print("Schema of the DGMiner TSV file:")
        print(schema)

        # Test disease selection
        disease_df = filter_data(disease_name, disease_id)
        print("\nSelected Disease DataFrame:")
        print(disease_df.head())
        print("Saving DataFrame to TSV...")
        disease_df.to_csv(f'data/{disease_name}_{disease_id}.tsv', sep='\t', index=False)
    else:
        disease_df = pd.read_csv(f'data/{disease_name}_{disease_id}.tsv', sep='\t')

    # Test graph construction
    graph_data = graph_construction(disease_df)
    print("\nConstructed Graph Data:")
    print(graph_data)

    # Visualize graph w/ NetworkX
    print("Visualizing the graph...")
    visualize_graph(graph_data)

if __name__ == "__main__":
    main()