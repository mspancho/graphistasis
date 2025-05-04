# Unpack DG-Miner data tsv file (tab-separated values)
gzip -d data/DG-Miner_miner-disease-gene.tsv.gz

# Create directories
mkdir -p data/
mkdir -p models
mkdir -p figures
mkdir -p scripts

# Move files to appropriate directories
mv unpack.sh scripts/unpack.sh
mv DG-Miner_miner-disease-gene.tsv data/mine-disease-gene.tsv
mv graph_construction.py scripts/graph_construction.py