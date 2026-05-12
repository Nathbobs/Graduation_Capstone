# Graduation Capstone Project
### Project's Goal:
To build a 3D reconstruction pipeline using [Nerf](https://github.com/bmild/nerf) and [3D Guassian splatting (3DGs)](https://github.com/graphdeco-inria/gaussian-splatting) to render photorealistic 3D scenes from processed multi-view image datasets. Modifying the approaches of the authors of [Nerf](https://github.com/bmild/nerf) and [3D Guassian splatting](https://github.com/graphdeco-inria/gaussian-splatting),  we were able to achieve state of the art results on a lower-end consumer GPU. We also experimented with the following variations of 3DGs such as [FeatureGS](),[ScaffoldGS](). (To be continued)

## Our Specs

Nvidia RTX 2070
16GB RAM
8GB Vram

Ubuntu 24.04 LTS

#### vram 8 gigabyte

python train.py -s data/truck -m output/truck \
  --iterations 7000 \
  --densify_grad_threshold 0.01 \
  --densification_interval 400 \
  --densify_until_iter 4000 \
  --test_iterations -1
