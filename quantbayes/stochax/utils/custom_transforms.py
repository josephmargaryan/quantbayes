import jax
import jax.scipy.ndimage as ndi 
import jax.numpy as jnp
import jax.random as jr
from augmax.base import Transformation, InputType

__all__ = [
    "TumourGuidedCrop",
    "MaskGuidedCrop",
    "RandomChoice",
    "RandomScale",
    "AdditiveGaussianNoise",
    "CoarseDropout",
]

"""
Example usage:
```python

rand_crop = RandomChoice(
    MaskGuidedCrop(256, 256, p_focus=0.9),
    TumourGuidedCrop(256, 256, focus_ratio=0.02, p_focus=0.7),
    augmax.RandomCrop(256, 256),
    p=[0.5, 0.3, 0.2],
)

transform = augmax.Chain(
    rand_crop,
    RandomScale(min_s=0.9, max_s=1.1, p=0.5), 
    augmax.HorizontalFlip(p=0.5),
    augmax.Rotate(angle_range=10, p=0.6),
    AdditiveGaussianNoise(sigma_min=0.0, sigma_max=0.02, p=0.3), 
    augmax.RandomGamma(range=(0.9, 1.2), p=0.5),
    CoarseDropout(patch=32, dropout_prob=0.15, p=0.2),          
    augmax.GaussianBlur(sigma=1.2, p=0.3),
    input_types=[augmax.InputType.IMAGE, augmax.InputType.MASK],
)
augment_fn = make_augmax_augment(transform)
```

"""


class TumourGuidedCrop(Transformation):
    """
    Random crop that (with probability p_focus) is centred on one of the
    brightest `focus_ratio` pixels in the PET-CT slice.

    Inputs : [image, mask]   – each [H, W, C]
    Output : [cropped_image, cropped_mask]
    """

    def __init__(
        self,
        crop_h: int = 256,
        crop_w: int = 256,
        *,
        focus_ratio: float = 0.05,
        p_focus: float = 1.0,
        input_types=(InputType.IMAGE, InputType.MASK),
    ):
        super().__init__(input_types=input_types)
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)
        self.focus_ratio = float(focus_ratio)
        self.p_focus = float(p_focus)

    def apply(self, rng, values, input_types, invert: bool = False):
        if invert:
            return values  

        img, mask = values  
        H, W = img.shape[:2]  
        Npix = H * W

        rng, k_focus, k_pick = jr.split(rng, 3)
        use_focus = jr.bernoulli(k_focus, self.p_focus)  

        K = max(1, int(self.focus_ratio * Npix))  
        intensity = img.max(axis=-1).reshape(-1)  
        _, idx_top = jax.lax.top_k(intensity, K)  

        idx_from_top = idx_top[jr.randint(k_pick, (), 0, K)]
        idx_random = jr.randint(k_pick, (), 0, Npix)
        flat_idx = jnp.where(use_focus, idx_from_top, idx_random)

        y_c = flat_idx // W
        x_c = flat_idx % W

        top = jnp.clip(y_c - self.crop_h // 2, 0, H - self.crop_h)
        left = jnp.clip(x_c - self.crop_w // 2, 0, W - self.crop_w)

        img_crop = jax.lax.dynamic_slice(
            img, (top, left, 0), (self.crop_h, self.crop_w, img.shape[-1])
        )
        mask_crop = jax.lax.dynamic_slice(
            mask, (top, left, 0), (self.crop_h, self.crop_w, mask.shape[-1])
        )

        return [img_crop, mask_crop] 


class MaskGuidedCrop(Transformation):
    """
    Crop (crop_h × crop_w) around a pixel chosen from the tumour mask
    with probability p_focus; random elsewhere otherwise.

    Expected inputs : [image, mask]   where both are [H, W, C].
    Returns         : [cropped_image, cropped_mask]  (same list type).

    Parameters
    ----------
    crop_h, crop_w : int
        Output window size.
    p_focus : float
        Chance of forcing the centre inside the mask (0 ≤ p ≤ 1).
    """

    def __init__(
        self,
        crop_h: int = 256,
        crop_w: int = 256,
        *,
        p_focus: float = 0.8,
        input_types=(InputType.IMAGE, InputType.MASK),
    ):
        super().__init__(input_types=input_types)
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)
        self.p_focus = float(p_focus)

    def apply(self, rng, values, input_types, invert: bool = False):
        if invert:
            return values  

        img, mask = values 
        H, W = img.shape[:2]  
        N = H * W

        rng, k_pick, k_gumbel, k_rand = jr.split(rng, 4)

        pos = (mask[..., 0] > 0).reshape(-1)  
        has_pos = jnp.any(pos)  

        gumbel = -jnp.log(-jnp.log(jr.uniform(k_gumbel, (N,))))
        scores = jnp.where(pos, gumbel, -jnp.inf)
        idx_from_mask = jnp.argmax(scores)  

        idx_random = jr.randint(k_rand, (), 0, N) 


        idx_focus = jax.lax.cond(
            has_pos, lambda _: idx_from_mask, lambda _: idx_random, operand=None
        )

        use_focus = jr.bernoulli(k_pick, self.p_focus)
        flat_idx = jnp.where(use_focus, idx_focus, idx_random)

        y_c = flat_idx // W
        x_c = flat_idx % W

        top = jnp.clip(y_c - self.crop_h // 2, 0, H - self.crop_h)
        left = jnp.clip(x_c - self.crop_w // 2, 0, W - self.crop_w)

        img_crop = jax.lax.dynamic_slice(
            img, (top, left, 0), (self.crop_h, self.crop_w, img.shape[-1])
        )

        mask_crop = jax.lax.dynamic_slice(
            mask, (top, left, 0), (self.crop_h, self.crop_w, mask.shape[-1])
        )

        return [img_crop, mask_crop] 


class RandomChoice(Transformation):
    """
    Randomly selects and applies exactly one transform from a list of provided transforms.

    Parameters
    *transforms : Transformation
        One or more instances of augmax `Transformation` (or subclasses).
    p : sequence of float, optional
        Probabilities associated with each transform. Must be the same length
        as `transforms` and sum to 1.0. If `None`, all transforms are chosen
        uniformly at random.

    Example
    -------
    ```python
    from augmax.base import InputType
    import augmax
    from your_module import MaskGuidedCrop 

    # 1) Define a RandomChoice that either does a mask-guided crop or a random crop
    rand_crop = RandomChoice(
        MaskGuidedCrop(crop_h=256, crop_w=256, p_focus=0.8),
        augmax.RandomCrop(256, 256),
        p=[0.5, 0.5]
    )

    # 2) Chain that with other augmentations
    transform = augmax.Chain(
        rand_crop,
        augmax.HorizontalFlip(p=0.5),
        augmax.HorizontalFlip(p=0.5),
        augmax.Rotate(angle_range=10, p=0.6),
        augmax.Warp(strength=3, coarseness=16),
        augmax.RandomContrast(range=(-0.2, 0.2), p=0.5),
        augmax.RandomGamma(range=(0.8, 1.2), p=0.5),
        augmax.GaussianBlur(sigma=2, p=0.3),
        input_types=[InputType.IMAGE, InputType.MASK],
    )

    # 3) Build your augment function
    augment_fn = make_augmax_augment(transform)
    ```
    """

    def __init__(self, *transforms, p=None):
        if not transforms:
            raise ValueError("Need one or more transforms")
        self.transforms = list(transforms)
        if p is not None:
            p = jnp.array(p, dtype=jnp.float32)
            if p.shape != (len(transforms),) or not jnp.isclose(p.sum(), 1.0):
                raise ValueError("`p` must have length = #transforms and sum to 1.")
        self.p = p

    def apply(self, rng, inputs, input_types, invert=False):
        rng, sample_key, *rest = jax.random.split(rng, num=len(self.transforms) + 2)
        if self.p is None:
            idx = jax.random.randint(sample_key, (), 0, len(self.transforms))
        else:
            idx = jax.random.choice(sample_key, a=len(self.transforms), p=self.p)

        branch_keys = jnp.stack(
            rest
        )  
        branches = [
            lambda _inp, t=t, key=branch_keys[i]: t.apply(
                key, _inp, input_types, invert=invert
            )
            for i, t in enumerate(self.transforms)
        ]
        return jax.lax.switch(idx, branches, inputs)

class RandomScale(Transformation):
    def __init__(self, min_s=0.9, max_s=1.1, p=0.5,
                 input_types=(InputType.IMAGE, InputType.MASK)):
        super().__init__(input_types=input_types)
        self.min_s, self.max_s, self.p = float(min_s), float(max_s), float(p)

    def apply(self, rng, values, input_types, invert=False):
        if invert:
            return values
        img, mask = values
        rng, k_p, k_s = jr.split(rng, 3)
        do = jr.bernoulli(k_p, self.p)
        s  = jr.uniform(k_s, (), minval=self.min_s, maxval=self.max_s)
        img = jnp.where(do, img * s, img)
        return [img, mask]


class AdditiveGaussianNoise(Transformation):
    def __init__(self, sigma_min=0.0, sigma_max=0.02, p=0.3,
                 input_types=(InputType.IMAGE, InputType.MASK)):
        super().__init__(input_types=input_types)
        self.sigma_min, self.sigma_max, self.p = float(sigma_min), float(sigma_max), float(p)

    def apply(self, rng, values, input_types, invert=False):
        if invert:
            return values
        img, mask = values
        rng, k_p, k_n = jr.split(rng, 3)
        do = jr.bernoulli(k_p, self.p)
        sigma = jr.uniform(k_n, (), minval=self.sigma_min, maxval=self.sigma_max)
        noise = jr.normal(k_n, img.shape) * sigma
        img = jnp.where(do, img + noise, img)
        return [img, mask]


class CoarseDropout(Transformation):
    """
    Zero-out square patches on the *image only*.

    Parameters
    ----------
    patch : int
        Edge length (pixels) of the square patch that each dropout “seed”
        expands into.
    dropout_prob : float
        Probability that any pixel becomes a seed (0–1).
    p : float
        Probability of applying the transform at all (0–1).
    """

    def __init__(self, patch: int = 32, dropout_prob: float = 0.15, p: float = 0.2,
                 input_types=(InputType.IMAGE, InputType.MASK)):
        super().__init__(input_types=input_types)
        self.patch         = int(patch)
        self.dropout_prob  = float(dropout_prob)
        self.p             = float(p)

    def apply(self, rng, values, input_types, invert: bool = False):
        if invert:
            return values                             

        img, mask = values
        H, W = img.shape[:2]

        rng, k_apply, k_seed = jr.split(rng, 3)
        do = jr.bernoulli(k_apply, self.p)

        seeds = jr.uniform(k_seed, (H, W)) < self.dropout_prob   

        window = (self.patch, self.patch)           
        seeds_f32 = seeds.astype(jnp.float32)        

        counts = jax.lax.reduce_window(
            seeds_f32,
            0.0,
            jax.lax.add,                
            window_dimensions=window,
            window_strides=(1, 1),
            padding='SAME'
        )
        dropout_mask = counts > 0.0                 
        dropout_mask = dropout_mask[..., None]     

        img = jnp.where(do & dropout_mask, 0.0, img)
        return [img, mask]