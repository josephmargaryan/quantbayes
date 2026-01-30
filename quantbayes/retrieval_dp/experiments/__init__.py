"""
How to run:

# ANALYTIC
!python -m quantbayes.retrieval_dp.experiments.exp_cifar10_private_retrieval \
  --out_dir ./runs/analytic/cifar10_retrieval_ball_dp \
  --l2_normalize \
  --mechanism gaussian \
  --sigma_method analytic \
  --n_per_class_list 100,2000,5000

# CLASSIC
!python -m quantbayes.retrieval_dp.experiments.exp_cifar10_private_retrieval \
  --out_dir ./runs/classic/cifar10_retrieval_ball_dp \
  --l2_normalize \
  --mechanism gaussian \
  --sigma_method classic \
  --n_per_class_list 100,2000,5000

"""
