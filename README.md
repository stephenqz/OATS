## OATS: Outlier-Aware Pruning Througn Sparse and Low Rank Decomposition
This repository contains the code for OATS: Outlier-Aware Pruning through Sparse and Low Rank Decomposition.

## Abstract
The recent paradigm shift to large-scale foundation models has brought about a new era for deep learning that, while has found great success in practice, has also been plagued by prohibitively expensive costs in terms of high memory consumption and compute. To mitigate these issues, there has been a concerted effort in post-hoc neural network pruning techniques that do not require costly retraining. Despite the considerable progress being made, existing methods often exhibit a steady drop in model performance as the compression increases. In this paper, we present a novel approach to compressing large transformers, coined OATS, that compresses the model weights by approximating each weight matrix as the sum of a sparse matrix and a low-rank matrix. Prior to the decomposition, the weights are first scaled by the second moment of their input embeddings, so as to ensure the preservation of  outlier features recently observed in large transformer models. Without retraining, OATS achieves state-of-the-art performance when compressing large language models, such as Llama-3 and Phi-3, and vision transformers, such as Google's ViT and DINOv2, by up to 60%, all while speeding up the model's inference on a CPU by up to 1.37x compared to prior pruning methods.

![vit_viz](figures/vit_viz.png)

## Dependencies
The dependencies used to run the experiments in our paper are:
```
    accelerate==0.29.3
    datasets==2.19.0
    lm_eval==0.4.2
    ml_collections==0.1.1
    torch==2.3.0
    transformers==4.44.1
```

## How to Run
Hyperparameter and experiment specifications are passed via a list of dictionaries in `OATS_configs.py` with each dictionary representing a specific experiment. The `compress` variable for OATS should be set to `False` if evaluating model performance. If this is the case, the sparse plus low-rank matrices are summed and saved as a single dense matrix. File to run is `main.py`. 

## Codebases Utilized
Our code utilizes and takes inspiration from the codebases found at the following GitHub Repos:
```
    SliceGPT: https://github.com/microsoft/TransformerCompression
    SparseGPT: https://github.com/IST-DASLab/sparsegpt
    Wanda: https://github.com/locuslab/wanda
```

## Citation
```
@inproceedings{
zhang2025oats,
title={{OATS}: Outlier-Aware Pruning Through Sparse and Low Rank Decomposition},
author={Stephen Zhang and Vardan Papyan},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=DLDuVbxORA}
}
```
