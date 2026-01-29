"""
How to run

# 1) Prototype experiment (produces acc_vs_eps_prototypes.png + results.csv)
python -m quantbayes.ball_dp.experiments.exp_cifar10_prototypes \
  --out_dir ./runs/cifar10_prototypes_ball_dp \
  --l2_normalize \
  --n_per_class_list 100,2000,5000

# 2) Coverage curve figure (coverage_curve.png)
python -m quantbayes.ball_dp.experiments.exp_radius_coverage \
  --out_dir ./runs/cifar10_prototypes_ball_dp \
  --l2_normalize

# 3) Attack/audit figure (attack_audit_vs_eps.png + audit_results.csv)
python -m quantbayes.ball_dp.experiments.exp_cifar10_attacks \
  --out_dir ./runs/cifar10_prototypes_ball_dp \
  --l2_normalize \
  --n_trials 2000

# 4) Logistic regression head experiment (acc_vs_eps_logreg.png + results.csv)
python -m quantbayes.ball_dp.experiments.exp_cifar10_logreg \
  --out_dir ./runs/cifar10_logreg_ball_dp \
  --l2_normalize

!python -m quantbayes.ball_dp.experiments.exp_privacy_profile \
  --out_dir ./runs/cifar10_privacy_profile \
  --l2_normalize \
  --r_percentiles 10,25,50,75,90 \
  --nn_sample_per_class 400


"""
