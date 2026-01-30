"""
How to run

# ERM
# ANALYTIC
!python -m quantbayes.ball_dp.experiments.exp_cifar10_prototypes \
  --out_dir ./runs/analytic/cifar10_prototypes_ball_dp \
  --l2_normalize \
  --sigma_method analytic \
  --n_per_class_list 100,2000,5000

# CLASSIC
!python -m quantbayes.ball_dp.experiments.exp_cifar10_prototypes \
  --out_dir ./runs/classic/cifar10_prototypes_ball_dp \
  --l2_normalize \
  --sigma_method classic \
  --n_per_class_list 100,2000,5000

# ANALYTIC
!python -m quantbayes.ball_dp.experiments.exp_cifar10_logreg \
  --out_dir ./runs/analytic/cifar10_logreg_ball_dp \
  --l2_normalize \
  --sigma_method analytic \
  --lam 1e-2 \
  --weight_decay 1e-2 \
  --n_per_class_list 100,2000,5000

# CLASSIC
!python -m quantbayes.ball_dp.experiments.exp_cifar10_logreg \
  --out_dir ./runs/classic/cifar10_logreg_ball_dp \
  --l2_normalize \
  --sigma_method classic \
  --lam 1e-2 \
  --weight_decay 1e-2 \
  --n_per_class_list 100,2000,5000

  
# Attacks 
# MAIN (analytic)
!python -m quantbayes.ball_dp.experiments.exp_cifar10_attacks \
  --out_dir ./runs/analytic/cifar10_attacks_ball_dp \
  --l2_normalize \
  --sigma_method analytic \
  --n_pairs 5000 \
  --band 0.10

# APPENDIX (classic)
!python -m quantbayes.ball_dp.experiments.exp_cifar10_attacks \
  --out_dir ./runs/classic/cifar10_attacks_ball_dp \
  --l2_normalize \
  --sigma_method classic \
  --n_pairs 5000 \
  --band 0.10

!python -m quantbayes.ball_dp.experiments.exp_cifar10_attacks \
  --out_dir ./runs/analytic/cifar10_attacks_ball_dp \
  --l2_normalize \
  --sigma_method analytic \
  --n_pairs 5000 \
  --band 0.10 \
  --make_multiplier_plot \
  --mult_eps 1.0


# Radius coverage and privacy profiles 

!python -m quantbayes.ball_dp.experiments.exp_radius_coverage \
  --out_dir ./runs/shared/cifar10_coverage \
  --l2_normalize \

!python -m quantbayes.ball_dp.experiments.exp_privacy_profile \
  --out_dir ./runs/shared/cifar10_privacy_profile \
  --l2_normalize \
  --r_percentiles 10,25,50,75,90 \
  --nn_sample_per_class 400 \
  --B_quantile 0.999

"""
