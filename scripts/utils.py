from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import torch
import torch_geometric as pyg
from Bio import Entrez
import json

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

    # Drop Diseases that don't start with "D"
    print("Dropping diseases that don't start with 'D'...")
    df = df[df['# Disease(MESH)'].str.startswith('D')]

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

def explore_data(disease_df: pd.DataFrame):
    """
    Explore the data by printing the first few rows and the number of unique diseases.
    """
    print("Exploring data...")
    print(disease_df.head())
    print(f"Number of unique diseases: {disease_df['# Disease(MESH)'].nunique()}")
    print(f"Number of unique genes: {disease_df['Gene'].nunique()}")
    print(f"Number of disease-gene pairs: {len(disease_df)}")

def generate_artificial_data(disease_df, num_interactions=1000):
    # Extract unique gene names
    genes = pd.unique(disease_df['Gene'])

    # Generate a random gene-gene interaction dataset
    # Number of random interactions to generate
    gene1 = np.random.choice(genes, num_interactions)
    gene2 = np.random.choice(genes, num_interactions)

    # Create a DataFrame for the artificial gene-gene interactions
    artificial_data = pd.DataFrame({'Gene1': gene1, 'Gene2': gene2})
    print("Generated artificial gene-gene interaction dataset:")
    print(artificial_data.head())

    return artificial_data

def create_mappings(file_path='data/gene_mapping.tsv'):
    """
    Create mappings between UniProt IDs and gene names.
    """
    # Read the gene mapping file
    df = pd.read_csv(file_path, sep='\t')

    # Create a mapping from UniProt ID to gene name
    from_to = df.set_index('From')['To'].to_dict()

    # Create a reverse mapping from gene name to UniProt ID
    to_from = df.set_index('To')['From'].to_dict()

    return from_to, to_from, df


def fetch_epistatic_interactions(gene_list, access_key, tax_id=9606, batch_size=10):
    """
    Fetches interactions for multiple genes from the BioGRID API in batches.

    Args:
        gene_list (list): A list of gene names to query.
        access_key (str): Your BioGRID API access key.
        tax_id (int): NCBI Taxonomy ID to filter by species (default is 9606 for humans).
        batch_size (int): Number of genes to process per batch.

    Returns:
        dict: A dictionary mapping each gene to its list of interacting genes.
    """
    base_url = "https://webservice.thebiogrid.org/interactions/"
    all_interactions = {}

    # Function to batch the gene list
    def batch_gene_list(gene_list, batch_size):
        for i in range(0, len(gene_list), batch_size):
            yield gene_list[i:i + batch_size]

    # Initialize tqdm progress bar
    total_batches = (len(gene_list) + batch_size - 1) // batch_size  # Calculate total number of batches
    for batch in tqdm(batch_gene_list(gene_list, batch_size), total=total_batches, desc="Processing Batches"):
        params = {
            "geneList": "|".join(batch),  # Join genes in the batch with '|'
            "searchNames": "true",
            "includeInteractors": "true",  # Include first-order interactors
            "accessKey": access_key,
            "format": "json",
            "taxId": tax_id
        }

        try:
            response = requests.get(base_url, params=params)

            if response.status_code != 200:
                print(f"Request URL: {response.url}")
                print(f"Error fetching interactions: {response.status_code}, {response.text}")
                continue  # Skip this batch and move to the next one

            interactions = response.json()

            # Ensure the response is a dictionary
            if not isinstance(interactions, dict):
                print(f"Unexpected response format for batch {batch}: {interactions}")
                continue  # Skip this batch and move to the next one

            # Parse the interactions
            for interaction_id, interaction_data in interactions.items():
                try:
                    gene_a = interaction_data.get("OFFICIAL_SYMBOL_A")
                    gene_b = interaction_data.get("OFFICIAL_SYMBOL_B")

                    # Add interactions to the dictionary
                    if gene_a not in all_interactions:
                        all_interactions[gene_a] = []
                    if gene_b not in all_interactions:
                        all_interactions[gene_b] = []

                    # Avoid self-interactions
                    if gene_a != gene_b:
                        all_interactions[gene_a].append(gene_b)
                        all_interactions[gene_b].append(gene_a)
                except Exception as e:
                    print(f"Error parsing interaction {interaction_id}: {e}")
                    continue  # Skip this interaction and move to the next one

        except Exception as e:
            print(f"Error processing batch {batch}: {e}")
            continue  # Skip this batch and move to the next one

    # Filter the dictionary to only include the genes in the input list
    filtered_interactions = {gene: all_interactions.get(gene, []) for gene in gene_list}

    return filtered_interactions

Entrez.email = "tasawwar_rahman@brown.edu"  # Replace with your email
Entrez.api_key = "b512b74d9e6a81b2ae8c21068512ecfb2308"

def mesh_to_name(mesh_ids):
    """
    Convert multiple MESH IDs to disease names using NCBI Entrez.

    Args:
        mesh_ids (list): A list of MESH IDs to query.

    Returns:
        dict: A dictionary mapping MESH IDs to disease names.
    """
    mesh_to_disease = {}
    for mesh_id in tqdm(mesh_ids, desc="Processing MESH IDs"):
        try:
            # Step 1: Convert MESH ID to numeric UID using esearch
            esearch_handle = Entrez.esearch(db="mesh", term=mesh_id, retmode="xml")
            esearch_record = Entrez.read(esearch_handle)
            esearch_handle.close()

            # Extract the UID from the esearch result
            uid_list = esearch_record.get("IdList", [])
            if not uid_list:
                print(f"No UID found for MESH ID {mesh_id}")
                mesh_to_disease[mesh_id] = None
                continue
            uid = uid_list[0]

            # Step 2: Fetch disease name using esummary and the UID
            esummary_handle = Entrez.esummary(db="mesh", id=uid, retmode="xml")
            esummary_record = Entrez.read(esummary_handle)
            esummary_handle.close()

            # Extract the primary disease name from DS_MeshTerms
            if "DS_MeshTerms" in esummary_record[0]:
                mesh_terms = esummary_record[0]["DS_MeshTerms"]
                if isinstance(mesh_terms, list) and len(mesh_terms) > 0:
                    mesh_to_disease[mesh_id] = mesh_terms[0]  # Use the first term as the primary disease name
                else:
                    print(f"No valid terms found in DS_MeshTerms for UID {uid}")
                    mesh_to_disease[mesh_id] = None
            else:
                print(f"'DS_MeshTerms' field not found for UID {uid}")
                mesh_to_disease[mesh_id] = None
        except Exception as e:
            print(f"Error processing MESH ID {mesh_id}: {e}")
            mesh_to_disease[mesh_id] = None

    return mesh_to_disease

def clean_and_map_data(df, uniprot_to_gene, mesh_to_disease, file_path=None):
    """
    Replace UniProt IDs with gene names and MESH IDs with disease names in the DataFrame.
    Drop rows where mappings are not found.

    Args:
        df (pd.DataFrame): The input DataFrame with UniProt and MESH IDs.
        uniprot_to_gene (dict): Dictionary mapping UniProt IDs to gene names.
        mesh_to (dict): Dictionary mapping MESH IDs to disease names.

    Returns:
        pd.DataFrame: The cleaned and mapped DataFrame.
    """
    # Replace UniProt IDs with gene names
    df['Gene'] = df['Gene'].map(uniprot_to_gene)

    # Replace MESH IDs with disease names
    df['# Disease(MESH)'] = df['# Disease(MESH)'].map(mesh_to_disease)

    # Drop rows where either 'Gene' or '# Disease(MESH)' is NaN (unmapped)
    df = df.dropna(subset=['Gene', '# Disease(MESH)'])

    # Rename Disease column
    df = df.copy()
    df.rename(columns={'# Disease(MESH)': 'Disease'}, inplace=True)

    if file_path:
        df.to_csv(file_path, sep='\t', index=False)

    return df

def generate_epistatic_interactions_tsv(interactions, gene_list, output_path):
    """
    Generate a TSV file of epistatic interactions for the given gene list.

    Args:
        interactions (dict): A dictionary where keys are genes and values are lists of interacting genes.
        gene_list (list): A list of genes to include in the interactions.
        output_path (str): Path to save the generated TSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the epistatic interactions.
    """
    # Filter interactions to include only genes in the gene list
    filtered_interactions = {
        gene: [interactor for interactor in interactors if interactor in gene_list]
        for gene, interactors in interactions.items()
        if gene in gene_list
    }

    # Create a list of interaction pairs
    interaction_pairs = []
    for gene, interactors in tqdm(filtered_interactions.items(), desc="Processing Interactions"):
        for interactor in interactors:
            # Add each interaction as a pair (gene, interactor)
            interaction_pairs.append((gene, interactor))

    # Convert the interaction pairs to a DataFrame
    interaction_df = pd.DataFrame(interaction_pairs, columns=["Gene1", "Gene2"])

    # Save the DataFrame to a TSV file
    if output_path:
        interaction_df.to_csv(output_path, sep="\t", index=False)
        print(f"Epistatic interactions TSV file saved to {output_path}")
    
    return interaction_df

def generate_graph_from_tsv(tsv_path: str, tsv_type: str) -> nx.Graph:
    """
    Generate a graph from a TSV file where each line represents an edge between two nodes.

    Args:
        tsv_path (str): Path to the TSV file.

    Returns:
        nx.Graph: A NetworkX graph object.
    """
    # Read the TSV file into a DataFrame
    df = pd.read_csv(tsv_path, sep="\t", header=0)  # Skip the header row

    # Create a graph
    G = nx.Graph()

    # Add edges to the graph
    if tsv_type == "DG":
        for _, row in df.iterrows():
            disease = row[0]
            gene = row[1]
            G.add_node(disease, node_type=2, name=disease)
            G.add_node(gene, node_type=1, name=gene)
            G.add_edge(disease, gene)
    elif tsv_type == "GG":
        for _, row in df.iterrows():
            gene1 = row[0]
            gene2 = row[1]
            G.add_node(gene1, node_type=1, name=gene1)
            G.add_node(gene2, node_type=1, name=gene2)
            G.add_edge(gene1, gene2)

    return G

def plot_graph(G, title="Graph", save_path=None):
    """
    Plot a NetworkX graph.

    Args:
        G (nx.Graph): The graph to plot.
        title (str): Title of the plot.
        save_path (str): Path to save the plot image. If None, the plot will be shown but not saved.
    """
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G)
    ## Color code
    # colors = ['blue' if G.nodes[n].get('node_type') == 'gene' else 'red' for n in G.nodes()]
    # nx.draw(G, pos, with_labels=True, node_size=50, node_color=colors, font_size=8)
    nx.draw(G, pos, with_labels=True, node_size=50, node_color='blue', font_size=8)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")

    plt.show()

def save_graph(graph, file_path):
    """
    Save a NetworkX graph to a GML file.

    Args:
        graph (nx.Graph): The graph to save.
        file_path (str): The path to save the GML file.
    """
    nx.write_gml(graph, file_path)
    print(f"Graph saved to {file_path}")

def combine_graphs(graph1, graph2):
    """
    Combine two NetworkX graphs into one.

    Args:
        graph1 (nx.Graph): The first graph.
        graph2 (nx.Graph): The second graph.

    Returns:
        nx.Graph: The combined graph.
    """
    combined_graph = nx.compose(graph1, graph2)
    return combined_graph

def load_graph(file_path):
    """
    Load a NetworkX graph from a GML file.

    Args:
        file_path (str): The path to the GML file.

    Returns:
        nx.Graph: The loaded graph.
    """
    graph = nx.read_gml(file_path)
    return graph