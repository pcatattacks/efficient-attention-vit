# Project Proposal: Efficient Attention in Vision Transformers

**CS 5787 – Deep Learning**
**Students:** Pranav Dhingra, Shashank Ramachandran
**NetIDs:** pd453, sr2433
**Emails:** pd453@cornell.edu, sr2433@cornell.edu

## Papers Chosen

1. **An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale**
   _Dosovitskiy et al., ICLR 2021_
2. **Linformer: Self-Attention with Linear Complexity**  
   _Wang et al., NeurIPS 2020_
3. **Performer: Rethinking Attention with Performers**  
   _Choromanski et al., ICLR 2021_
4. **Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention**  
   _Xiong et al., AAAI 2021_
5. **Fusion of Regional and Sparse Attention in Vision Transformers**  
   _Ibtehaz et al., 2024_

## Problem and Motivation

The **Vision Transformer (ViT)** achieves strong image classification performance but suffers from quadratic computational complexity in its self-attention mechanism, limiting scalability for high-resolution images.

Recent research has introduced efficient approximations such as:

- Linformer (low-rank projections)
- Performer (kernelized linear attention)
- Nyströmformer (matrix approximation)

These methods reduce computation while retaining accuracy, but their trade-offs vary across datasets and scales.  
We aim to conduct a controlled empirical comparison of these efficient self-attention mechanisms on a more complex dataset, **ImageNet-100** (could be revised to a more complex dataset), where attention length and spatial resolution are large enough to meaningfully test efficiency gains.

Our study will focus on both computational performance and classification accuracy, providing insights into which approximations are most effective for scalable vision models.

## Data and Methods

- We will use a pre-trained Vision Transformer (ViT) backbone implemented in Hugging Face’s library and PyTorch Image Models (timm) to minimize redundant reimplementation.
- For each variant (Linformer, Performer, Nyströmformer), we will fine-tune and benchmark on **ImageNet-100** (other datasets might be explored as well), comparing training speed, inference latency, and top-1 accuracy.
- We will also explore a hybrid attention mechanism that integrates **atrous (dilated) attention** — inspired by ACC-ViT — to reduce computation while maintaining receptive field coverage.
  - This hybrid will be tested under similar conditions for fair comparison.

### Metrics

- **Training and inference time per epoch**
- **GPU memory usage**
- **Classification accuracy (top-1 and top-5)**
- **FLOPs and parameter counts**

## Evaluation

Results will be reported as trade-off curves between accuracy and computational cost.

We will analyze:

- How each efficient attention method scales with input resolution.
- The impact of dilated/hybrid attention on efficiency and accuracy.
- The practical deployability of each model in resource-limited scenarios (if time permits).
