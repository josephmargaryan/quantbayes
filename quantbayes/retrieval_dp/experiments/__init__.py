"""
How to run:

python -m quantbayes.retrieval_dp.experiments.exp_cifar10_private_retrieval \
  --out_dir ./runs/cifar10_retrieval_ball_dp \
  --l2_normalize \
  --mechanism gaussian \
  --score neg_l2 \
  --n_per_class_list 100,2000,5000
"""
