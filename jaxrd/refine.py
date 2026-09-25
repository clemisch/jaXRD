import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from jax.flatten_util import ravel_pytree



@jax.jit(static_argnames=("Ns", "fun_fwd"))
def get_lm_step(active, passive, ttheta_deg, y_obs, Ns, damping, fun_fwd):
    active_flat_init, unravel = ravel_pytree(active)

    def get_residuals(active_flat):
        model = eqx.combine(unravel(active_flat), passive)
        fwd = fun_fwd(model, xx, Ns)
        residuals = fwd - yy
        return residuals

    r = get_residuals(active_flat_init)
    J = jax.jacfwd(get_residuals)(active_flat_init)
    H = J.T @ J + damping * jnp.eye(active_flat_init.size)
    
    active_flat_new = active_flat_init + jnp.linalg.solve(H, -J.T @ r)
    active_new = unravel(active_flat_new)

    return active_new, jnp.sum(r**2)


@jax.jit(static_argnames=("Ns", "fun_fwd"))
def get_rwp(model, xx, yy, Ns, fun_fwd):
    fwd = fun_fwd(model, xx, Ns)
    residual = fwd - yy
    nom = jnp.sum(jnp.square(residual))
    denom = jnp.sum(jnp.square(yy))
    rwp = 100. * jnp.sqrt(nom / denom)

    return rwp


def optimize_schedule(
    model, schedule, 
    xx, yy, 
    n_fwhm, fun_fwd, 
    max_iter, rwp_tol=1e-6,
    damping=1e-4,
):
    active_init, passive = eqx.partition(model, schedule)

    rwp_prev = np.inf
    active_opt = active_init
    model_opt = model

    for i_iter in range(max_iter):
        Ns = get_Ns(model_opt, xx, n_fwhm)
        active_opt, rtot = get_lm_step(
            active_opt, passive, xx, yy, Ns, damping, fun_fwd
        )
        model_opt = eqx.combine(active_opt, passive)
        rwp_new = get_rwp(model_opt, xx, yy, Ns, fun_fwd)
        print(rwp_new)

        if abs(rwp_new - rwp_prev) < rwp_tol:
            break
        rwp_prev = rwp_new

    return model_opt, i_iter+1, rwp_new
