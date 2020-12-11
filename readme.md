## Coresets for Robust Training of Neural Networks against Noisy Labels
Baharan Mirzasoleiman, Kaidi Cao, Jure Leskovec
_________________

This is the official implementation of crust in the paper [Coresets for Robust Training of Neural Networks against Noisy Labels](https://proceedings.neurips.cc/paper/2020/file/8493eeaccb772c0878f99d60a0bd2bb3-Paper.pdf) in PyTorch.

### Dependency

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.7
- [scikit-learn](https://scikit-learn.org/stable/)

### Training 

We provide a training example with this repo:

```bash
python robust_cifar_train.py --gpu 0 --use_crust
```

### Reference

If you find our paper and repo useful, please cite as

```
@article{mirzasoleiman2020coresets,
  title={Coresets for Robust Training of Neural Networks against Noisy Labels},
  author={Mirzasoleiman, Baharan and Cao, Kaidi and Leskovec, Jure},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```