# Graduation_Capstone
Graduation project

#### vram 8 gigabyte

python train.py -s data/truck -m output/truck \
  --iterations 7000 \
  --densify_grad_threshold 0.01 \
  --densification_interval 400 \
  --densify_until_iter 4000 \
  --test_iterations -1
