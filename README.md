# CNNIQA
PyTorch 0.4 implementation of the following paper:
[Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf)

### Note
- The optimizer is chosen as Adam here, instead of the SGD with momentum in the paper.
- the mat files in data/ are the information extracted from the datasets and the index information about the train/val/test split. The subjective scores of LIVE is from the [realigned data](http://live.ece.utexas.edu/research/Quality/release2/dmos_realigned.mat).

## Training

**You can get the dataset from https://pan.baidu.com/s/18y5zswF_rKaQbf0ZPuP5Bg [1ih5] , and then put them into "data/".**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --exp_id=0 --database=LIVE
```
Before training, the `im_dir` in `config.yaml` must to be specified.
Train/Val/Test split ratio in intra-database experiments can be set in `config.yaml` (default is 0.6/0.2/0.2).

### Visualization
```bash
tensorboard --logdir=tensorboard_logs --port=6006 # in the server
ssh -L 6006:localhost:6006 user@host # in your PC, then see the visualization in your PC
```
