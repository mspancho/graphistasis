# Create directories
mkdir -p data
mkdir -p figures
mkdir -p models

# Download the dataset
wget -P data https://snap.stanford.edu/biodata/datasets/10020/files/DG-Miner_miner-disease-gene.tsv.gz

# Unzip the dataset
gzip -d data/DG-Miner_miner-disease-gene.tsv.gz