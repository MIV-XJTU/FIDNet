## [ACMMM 2024] Bridging Fourier and Spatial-Spectral Domains for Hyperspectral Image Denoising

The official PyTorch implementation of [FIDNet](https://dl.acm.org/doi/abs/10.1145/3664647.3681461).

## Model
The primary implementation of the FIDNet can be found in the following directory:

```
model/FIDNet.py
```

## Running
For training and testing, you can use the code provided in the [RAS2S](https://github.com/MIV-XJTU/RAS2S). 
Simply place the model file `model/FIDNet.py` into the `basic/models/competing_methods` directory, and ensure that the path to checkpoint/fidnet.pth is correctly set.

## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@inproceedings{xiao2024bridging,
  title={Bridging Fourier and Spatial-Spectral Domains for Hyperspectral Image Denoising},
  author={Xiao, Jiahua and Liu, Yang and Zhang, Shizhou and Wei, Xing},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024}
}
@inproceedings{xiao2024region,
  title={Region-Aware Sequence-to-Sequence Learning for Hyperspectral Denoising},
  author={Xiao, Jiahua and Liu, Yang and Wei, Xing},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
