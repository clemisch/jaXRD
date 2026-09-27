import jax
import jax.numpy as jnp
import equinox as eqx

from jax.flatten_util import ravel_pytree

from .model import _get_histogram, _get_Ns



@jax.jit(static_argnames="Ns")
def get_lm_step(active, passive, ttheta_deg, y_obs, Ns, damping):
    active_flat_init, unravel = ravel_pytree(active)

    def get_residuals(active_flat):
        histogram = eqx.combine(unravel(active_flat), passive)
        fwd = _get_histogram(histogram, ttheta_deg, Ns)
        residuals = fwd - y_obs
        return residuals

    r = get_residuals(active_flat_init)
    J = jax.jacfwd(get_residuals)(active_flat_init)
    H = J.T @ J + damping * jnp.eye(active_flat_init.size)
    
    active_flat_new = active_flat_init + jnp.linalg.solve(H, -J.T @ r)
    active_new = unravel(active_flat_new)

    return active_new, jnp.sum(r**2)


@jax.jit(static_argnames="Ns")
def get_rwp(histogram, ttheta_deg, y_obs, Ns):
    fwd = _get_histogram(histogram, ttheta_deg, Ns)
    residual = fwd - y_obs
    nom = jnp.sum(jnp.square(residual))
    denom = jnp.sum(jnp.square(y_obs))
    rwp = 100. * jnp.sqrt(nom / denom)

    return rwp


def optimize_schedule(
    histogram, schedule, 
    ttheta_deg, y_obs, 
    n_fwhm, max_iter, rwp_tol=1e-6, damping=1e-4,
):
    active_init, passive = eqx.partition(histogram, schedule)

    rwp_prev = jnp.inf
    active_opt = active_init
    histogram_opt = histogram

    for i_iter in range(max_iter):
        Ns = _get_Ns(histogram_opt, ttheta_deg, n_fwhm)
        active_opt, rtot = get_lm_step(
            active_opt, passive, 
            ttheta_deg, y_obs, Ns, damping
        )
        histogram_opt = eqx.combine(active_opt, passive)
        rwp_new = get_rwp(histogram_opt, ttheta_deg, y_obs, Ns)
        print(rwp_new)

        if abs(rwp_new - rwp_prev) < rwp_tol:
            break
        rwp_prev = rwp_new

    return histogram_opt, i_iter+1, rwp_new
