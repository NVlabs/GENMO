# GENMO: A Generalist Model for Human Motion

[![Project Page](https://img.shields.io/badge/Project-Page-0099cc)](https://research.nvidia.com/labs/dair/genmo/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.01425-b31b1b.svg)](https://arxiv.org/abs/2505.01425)

**GENMO** is a unified generalist model for **human motion generation and estimation**, developed by NVIDIA Research.  
It bridges motion *generation* and *estimation* in a single framework that can synthesize and reconstruct human motion across multiple modalities — video, text, audio, and keypoints.

---

## 📰 News

- **[October 2025]** 📢 The **GENMO** codebase will be **released soon!**  
  Stay tuned for the official open-source release, including pretrained models and evaluation scripts.  
  Follow the [project page](https://research.nvidia.com/labs/dair/genmo/) for updates and announcements.


---

## 🚀 Highlights

GENMO introduces a **unified generative framework** that connects motion estimation and generation through shared objectives.

- **Unified framework:** Reframes motion estimation as *constrained generation*, allowing a single model to perform both tasks.  
- **Regression × Diffusion synergy:** Combines the accuracy of regression models with the diversity of diffusion-based generation.  
- **Estimation-guided training:** Trains effectively on in-the-wild datasets using only 2D or textual supervision.  
- **Multimodal conditioning:** Supports video, text, audio, 2D/3D keyframes, or even time-varying mixed inputs (e.g., video → text → video).  
- **Arbitrary-length motion:** Generates continuous, coherent sequences of any duration in one diffusion pass.  
- **State-of-the-art performance:** Achieves leading results on diverse motion estimation and generation benchmarks.

For more details, visit the **[GENMO project page →](https://research.nvidia.com/labs/dair/genmo/)**

---

## 📖 Paper & Citation

**Paper:**  
[GENMO: Generative Models for Human Motion Synthesis](https://arxiv.org/abs/2505.01425)  
*Jiefeng Li, Jinkun Cao, Haotian Zhang, Davis Rempe, Jan Kautz, Umar Iqbal, Ye Yuan*  
arXiv preprint, 2025

**BibTeX:**
```bibtex
@inproceedings{genmo2025,
  title     = {GENMO: Generative Models for Human Motion Synthesis},
  author    = {Li, Jiefeng and Cao, Jinkun and Zhang, Haotian and Rempe, Davis and Kautz, Jan and Iqbal, Umar and Yuan, Ye},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
