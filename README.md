# 3d-pc-AE-GAN
This code was maded based on PointNet.(repository : https://github.com/charlesq34/pointnet)
And I used chamfer distance + repulsion distance as a point cloud loss function.
Repulsion distance was used for uniform distribution of the point cloud. (paper : https://arxiv.org/abs/1801.06761)
Also I used the autoencoder model as a generator of GAN.

To use tf_grouping type make. Also to use tf_nn_distance change makefile_dist to makefile and type make.
Before train, you must download shapenet datasets and make data dir and copy or move the shapenet datasets data dir.
 <a href="https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip" target="_blank">(Download datasets)</a>
 
To train model type "python train.py".

If you want to see the scattered point cloud, you must download visdom and type "python -m visdom.server".

Here, we will be able to use the GAN model to reconstruct a damaged 3D point cloud 
by classifying it using feature distance to find out what shape of a point cloud it is.

## Before train
![prediction example](https://github.com/skoo9500/3d-pc-AE-GAN/blob/master/ae-gan/screenshot/before_train1.png)
![prediction example](https://github.com/skoo9500/3d-pc-AE-GAN/blob/master/ae-gan/screenshot/before_train2.png)
## After train
![prediction example](https://github.com/skoo9500/3d-pc-AE-GAN/blob/master/ae-gan/screenshot/airplane.png)
![prediction example](https://github.com/skoo9500/3d-pc-AE-GAN/blob/master/ae-gan/screenshot/chair.png)
![prediction example](https://github.com/skoo9500/3d-pc-AE-GAN/blob/master/ae-gan/screenshot/desk.png)

This is a screenshot of the test data applied to the model.
