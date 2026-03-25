# QuickStart
## A. Built-in theorem-backed model
```python
from quantbayes.ball_dp.api import ImageTask, fit_private_image_classifier

task = ImageTask.from_name("mnist", normalize="zero_one", train_limit=12000, test_limit=2000)

run = fit_private_image_classifier(
    task,
    privacy="ball_dp",
    architecture="cnn",
    epsilon=6.0,
    delta=1e-5,
    clip=1.0,
    epochs=8,
    lr=0.15,
)
print(run.summary())
run.plots()
```
## B. Custom dataset, standard RDP, custom model, no `provided_lz`
```python
run = fit_private_image_classifier(
    task,
    privacy="standard_rdp",
    architecture="custom",
    model=model,
    state=state,
    noise_std=2.0,      # explicit RDP run
    clip=1.0,           # required for standard accounting
    epochs=20,
    lr=0.05,
    spectral_target=None,
)```

## C. Custom dataset, Ball RDP, custom model, proved `L_z`
```python
run = fit_private_image_classifier(
    task,
    privacy="ball_rdp",
    architecture="custom",
    model=model,
    state=state,
    provided_lz=proved_Lz,
    noise_std=2.0,
    clip=None,          # allowed when your theorem gives a global L_z
    epochs=20,
    lr=0.05,
    spectral_target=None,
)```
## D. Truly noiseless custom baseline
```python
run = fit_private_image_classifier(
    task,
    privacy="noiseless",
    architecture="custom",
    model=model,
    state=state,
    clip=None,          # truly unclipped
    epochs=20,
    lr=0.05,
    spectral_target=None,
)
```
## E. Shadow attack with a custom architecture
```python
import equinox as eqx
import jax.random as jr

def model_factory(seed: int, task):
    key = jr.PRNGKey(seed)
    model, state = eqx.nn.make_with_state(
        JAXDPBaselineCNN
    )(key=key, num_classes=2)
    return model, state

attack_run = run_shadow_attack(
    task,
    privacy="ball_rdp",
    architecture="custom",
    model_factory=model_factory,
    provided_lz=proved_Lz,
    loss_name="softmax_cross_entropy",   # or "binary_logistic" if model outputs one logit
    noise_std=2.0,
    clip=1.0,
    epochs=5,
    batch_size=128,
    shadow_trials=24,
    shadow_dataset_size=128,
    side_info_regime="known_label",
)
print(attack_run.summary())
attack_run.plots()
attack_run.plot_rero()```



