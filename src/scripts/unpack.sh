# Unpack DG-Miner data tsv file (tab-separated values)
gzip -d src/data/DG-Miner/DG-Miner_miner-disease-gene.tsv.gz

# Create directories
mkdir -p src
mkdir -p src/data/DG-Miner-tsv
mkdir -p src/models
mkdir -p src/figures
mkdir -p src/scripts

# Move files to appropriate directories
mv unpack.sh src/scripts/unpack.sh
mv DG-Miner_miner-disease-gene.tsv src/data/DG-Miner-tsv/mine-disease-gene.tsv
mv graph_construction.py src/scripts/graph_construction.py