# AIComposer: Any Style and Content Image Composition via Feature Integration (ICCV 2025)

<a href='https://arxiv.org/abs/2507.20721'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
<a href='https://huggingface.co/sherlhw/AIComposer'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-green'></a>
<a href='https://huggingface.co/datasets/sherlhw/AIComposer-benchmark'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>


## Installation

```bash
# install diffusers
pip install diffusers==0.29.2

# download the models
cd AIComposer
git lfs install
git clone https://huggingface.co/sherlhw/AIComposer
mv AIComposer mlp_model
```

## Download Models

- [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

## Running AIComposer

### Data Preparation

Several input samples are available under `./examples` directory. Each sample involves one background (background_image), one foreground (ref_image), one segmentation mask for the foreground (ref_mask), and one user mask that denotes the desired composition location (background_mask). 

More samples are available in [AIComposer Test Benchmark](https://huggingface.co/datasets/sherlhw/AIComposer-benchmark) or you can customize them. 


### Image Composition

```
python demo.py # SDXL版本
```


## Acknowledgments
Our work is standing on the shoulders of giants. We thank the following contributors that our code is based on: [diffusers](https://github.com/huggingface/diffusers), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [TF-ICON](https://github.com/Shilin-LU/TF-ICON/) and [Magic Insert](https://magicinsert.github.io/?ref=aiartweekly).


## Citation
If you find AIComposer useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{li2025aicomposerstylecontentimage,
      title={AIComposer: Any Style and Content Image Composition via Feature Integration}, 
      author={Haowen Li and Zhenfeng Fan and Zhang Wen and Zhengzhou Zhu and Yunjin Li},
      booktitle={arXiv preprint arxiv:2507.20721},
      year={2025}
}
```