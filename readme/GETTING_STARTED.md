# Getting Started

This document provides tutorials to train and evaluate Rocky-CenterNet. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).

## Benchmark evaluation

First, download the models you want to evaluate from our [model zoo](MODEL_ZOO.md) and put them in `RockyCenterNet_ROOT/models/`.

### Rock-COCO

To evaluate Rock detection DLA (on Front Scale Dataset)
run

~~~
python test.py rockdet --exp_id rockdet_rock_coco_dla_2x --load_model ../models/rockdet_rock_coco_dla_2x.pth --dataset rock_coco --folder rock_front_dataset --trainval
~~~

This will give an AP of `58.8` (against ellipses) in the Scaled Front Dataset, if setup correctly.
Similarly to evaluate CenterNet:

~~~
python test.py ctdet --exp_id det_rock_coco_dla_2x --load_model ../models/ctdet_rock_coco_dla_2x.pth --flip_test --trainval
~~~

This will give an AP of `66.0` (vs bboxes) in the Scaled Front Dataset, if setup correctly.

## Training

We have packed all the training scripts in the [experiments](../experiments) folder.
The experiment names are correspond to the model name in the [model zoo](MODEL_ZOO.md).
The number of GPUs for each experiments can be found in the scripts and the model zoo.
In the case that you don't have 8 GPUs, you can follow the [linear learning rate rule](https://arxiv.org/abs/1706.02677) to scale the learning rate as batch size.
For example, to train COCO object detection with dla on 2 GPUs, run

~~~
python main.py ctdet --exp_id coco_dla --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 0,1
~~~

The default learning rate is `1.25e-4` for batch size `32` (on 2 GPUs).
By default, pytorch evenly splits the total batch size to each GPUs.
`--master_batch` allows using different batchsize for the master GPU, which usually costs more memory than other GPUs.
If it encounters GPU memory out, using slightly less batch size (e.g., `112` of `128`) with the same learning is fine.

If the training is terminated before finishing, you can use the same commond with `--resume` to resume training. It will found the lastest model with the same `exp_id`.
