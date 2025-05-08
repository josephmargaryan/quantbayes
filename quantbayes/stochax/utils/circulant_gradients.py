import jax
import jax.numpy as jnp


@jax.custom_jvp
def circulant_matmul(weight_vector, x):
    # Forward pass using FFT
    w_fft = jnp.fft.fft(weight_vector)
    x_fft = jnp.fft.fft(x, axis=-1)
    y_fft = x_fft * w_fft[None, :].conjugate()
    return jnp.fft.ifft(y_fft, axis=-1).real


@circulant_matmul.defjvp
def circulant_matmul_jvp(primals, tangents):
    w, x = primals
    dw, dx = tangents
    # forward = circulant_matmul(w, x)   # shape (batch, b)

    # derivative wrt w => circ(dL/dy) x
    # derivative wrt x => circ(dL/dy) w
    # We can do it in the frequency domain directly

    # Just to illustrate the idea:
    w_fft = jnp.fft.fft(w)
    x_fft = jnp.fft.fft(x, axis=-1)
    dw_fft = jnp.fft.fft(dw)
    dx_fft = jnp.fft.fft(dx, axis=-1)

    # Forward pass again
    primal_y_fft = x_fft * w_fft.conjugate()
    primal_y = jnp.fft.ifft(primal_y_fft, axis=-1).real

    # JVP
    # d/dw => IRFFT( conj(x_fft) * d(w_fft) ) + IRFFT(...) ???
    # We'll keep it short in the example:
    tangent_y_fft = (x_fft * dw_fft.conjugate()) + (dx_fft * w_fft.conjugate())
    tangent_y = jnp.fft.ifft(tangent_y_fft, axis=-1).real

    return primal_y, tangent_y
