# LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models
[![paper](https://img.shields.io/badge/arxiv-2502.02406-red.svg)](https://arxiv.org/abs/2502.02406)

Repository for ICML 2025 paper [LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models](https://arxiv.org/abs/2502.02406).

LV-XAttn is a distributed, exact, sequence-parallel cross-attention mechanism designed to handle long visual inputs in multimodal large language models (MLLMs).

## TLDR
In cross-attention for MLLMs, the size of key-value blocks is significantly larger than that of the query block. Existing sequence-parallel attention mechanism such as Ring Attention transmit these large key-value blocks among GPUs, involving large communication overhead. On the other hand, LV-XAttn transmits the significantly smaller query block and softmax statistics, resulting in up to 10.62x end-to-end speedup on Llama-3 on 16 A100 GPUs.

In addition, since the visual inputs (and thus the key-value blocks derived from them) are shared and remain unchanged across all cross-attention layers, we can significantly reduce the memory footprint and allow longer visual inputs to be processed by avoiding storage of key-value blocks for the backward pass and recomputing them on-the-fly.

## Structure
* `lv_xattn/`: Contains the implementation of LV-XAttn.
* `ring/`: Contains the implementation of Ring Attention.
* `ring_self/`: Same content as `ring/`, but for self-attention.
* `test_llama3.py`: Example of applying LV-XAttn to `Llama-3.2-11B-Vision-Instruct`. To see how to patch attention layers with QKV recomputation, refer to `patch_cross_attention_forward`. For patching without QKV recomputation, see `patch_self_attention_forward`.
* `test.py`: Contains correctness tests for LV-XAttn, comparing it with the PyTorch attention implementation.
* `scripts/`: Contains scripts to run the programs.

## Acknowledgements
The code is heavliy based on [DistFlashAttn](https://github.com/RulinShao/LightSeq).


## Citation
```bibtex
@article{chang2025lvxattn,
      title={LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models}, 
      author={Tzu-Tao Chang and Shivaram Venkataraman},
      year={2025},
      eprint={2502.02406},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.02406}, 
}
```

