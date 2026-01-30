"""
How to run:

python -m quantbayes.retrieval_dp.experiments.exp_cifar10_private_retrieval \
  --out_dir ./runs/cifar10_retrieval_ball_dp \
  --l2_normalize \
  --mechanism gaussian \ # or laplace
  --sigma_method analytic \ # or classic,
  --delta 1e-5 \
  --n_per_class_list 100,2000,5000 \
  --eps_list 0.05,0.1,0.2,0.5,1,2,5,10
"""
