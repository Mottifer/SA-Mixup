# SA-Mixup: Semantic-Aware Adaptive Node Mixup for Graph Neural Networks

This repository accompanies the manuscript **"SA-Mixup: Semantic-Aware
Adaptive Node Mixup for Graph Neural Networks"**.

SA-Mixup is a graph data augmentation framework for semi-supervised node
classification. The method generates virtual nodes through semantic-aware
adaptive mixup and integrates them into the original graph through a
reliability-constrained neighbor connection strategy.

## Overview

SA-Mixup contains three main stages:

1. **Node pair selection:** high-confidence pseudo-labeled nodes are
   selected and paired with labeled nodes from the same class.
2. **Semantic-aware adaptive node mixup:** semantic relation and predictive
   uncertainty are used to determine the mixing coefficient for each node
   pair.
3. **Similarity-guided neighbor connection:** generated nodes are connected to
   reliable candidate neighbors selected according to feature similarity.

## Experimental Environment

The experiments reported in the manuscript were conducted with:

- Python 3.12.10
- PyTorch 2.8.0
- PyTorch Geometric 2.7.0
- CUDA 12.8

The evaluated GNN backbones are GCN, GAT, and GraphSAGE. The benchmark datasets
include Cora, CiteSeer, Pubmed, Coauthor-CS, Coauthor-Physics, ogbn-arxiv, and
Flickr.

## Reproducibility

The released code will include the following materials:

- scripts for dataset preparation and semi-supervised data splits;
- backbone-specific configurations for GCN, GAT, and GraphSAGE;
- scripts for running SA-Mixup and baseline models;
- random-seed settings and evaluation instructions; and
- instructions for reproducing the main, ablation, and sensitivity experiments.

## Citation

Citation information will be added after publication.

## Contact

For questions regarding this work, please contact the corresponding author of
the associated manuscript.
