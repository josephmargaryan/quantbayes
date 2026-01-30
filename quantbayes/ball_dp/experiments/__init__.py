"""
How to run

# Run from repo root:
python -m quantbayes.ball_dp.experiments.exp_cifar10_prototypes \
  --out_dir paper/figures \
  --l2_normalize \
  --sigma_method analytic \
  --n_per_class_list 100,2000,5000

python -m quantbayes.ball_dp.experiments.exp_cifar10_logreg \
  --out_dir paper/figures \
  --l2_normalize \
  --sigma_method analytic \
  --lam 1e-2 \
  --n_per_class_list 100,2000,5000

python -m quantbayes.ball_dp.experiments.exp_cifar10_attacks \
  --out_dir paper/figures \
  --l2_normalize \
  --sigma_method analytic \
  --n_pairs 5000 \
  --band 0.10

python -m quantbayes.ball_dp.experiments.exp_radius_coverage \
  --out_dir paper/figures \
  --l2_normalize

python -m quantbayes.ball_dp.experiments.exp_privacy_profile \
  --out_dir paper/figures \
  --l2_normalize \
  --r_percentiles 10,25,50,75,90 \
  --nn_sample_per_class 400 \
  --B_quantile 0.999

"""

"""
python -m quantbayes.ball_dp.experiments.exp_cifar10_prototypes \
  --out_dir paper/figures/classic \
  --l2_normalize \
  --sigma_method classic \
  --n_per_class_list 100,2000,5000

python -m quantbayes.ball_dp.experiments.exp_cifar10_logreg \
  --out_dir paper/figures/classic \
  --l2_normalize \
  --sigma_method classic \
  --lam 1e-2 \
  --n_per_class_list 100,2000,5000

python -m quantbayes.ball_dp.experiments.exp_cifar10_attacks \
  --out_dir paper/figures/classic \
  --l2_normalize \
  --sigma_method classic \
  --n_pairs 5000 \
  --band 0.10
"""
