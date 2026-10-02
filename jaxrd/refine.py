import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from jax.flatten_util import ravel_pytree

from .model import _get_histogram, _get_Ns



@jax.jit(static_argnames="Ns")
def get_lm_step(
    active, 
    passive, 
    ttheta_deg, 
    y_obs, 
    Ns, 
    damping,
    limits=None

):
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
    if limits is not None:
        active_new = clip_parameters(active_new, limits)

    return active_new


@jax.jit(static_argnames="Ns")
def _get_rwp(histogram, ttheta_deg, y_obs, Ns):
    fwd = _get_histogram(histogram, ttheta_deg, Ns)
    residual = fwd - y_obs
    nom = jnp.sum(jnp.square(residual))
    denom = jnp.sum(jnp.square(y_obs))
    rwp = 100. * jnp.sqrt(nom / denom)

    return rwp


def get_schedule(histogram):
    schedule = jax.tree_util.tree_map(lambda _: False, histogram)

    return schedule


def get_limits(histogram, apply_default=True):
    limits = jax.tree_util.tree_map(lambda _: (None, None), histogram)

    if not apply_default:
        return limits

    limits.zero_deg = (-1., 1.)
    for name in limits.phases:
        limits.scales[name] = (0., None)
        limits.phases[name].intensities = (0., None)
        limits.phases[name].profile.U_deg2 = (None, None)
        limits.phases[name].profile.V_deg2 = (None, None)
        limits.phases[name].profile.W_deg2 = (1e-8, 1.)
        limits.phases[name].profile.eta = (0., 1.)

        lattice = histogram.phases[name].lattice
        limits.phases[name].lattice.a = (0.98 * lattice.a, 1.02 * lattice.a)
        limits.phases[name].lattice.b = (0.98 * lattice.b, 1.02 * lattice.b)
        limits.phases[name].lattice.c = (0.98 * lattice.c, 1.02 * lattice.c)
        limits.phases[name].lattice.alpha = (lattice.alpha - 5., lattice.alpha + 5.)
        limits.phases[name].lattice.beta = (lattice.beta - 5., lattice.beta + 5.)
        limits.phases[name].lattice.gamma = (lattice.gamma - 5., lattice.gamma + 5.)

    return limits


@jax.jit
def clip_parameters(parameters, limits):
    clipped = jax.tree_util.tree_map(
        lambda x, bound: None if x is None else jnp.clip(x, bound[0], bound[1]),
        parameters, limits,
        is_leaf=lambda x: x is None,
    )

    return clipped


def optimize_schedule(
    histogram, schedule, 
    ttheta_deg, y_obs, 
    n_fwhm, max_iter, rwp_tol=1e-6, damping=1e-8,
    silent=False, 
    limits=None,
):
    active_init, passive = eqx.partition(histogram, schedule)

    if limits is not None:
        limits = eqx.filter(limits, schedule, is_leaf=lambda x: isinstance(x, tuple))
        active_init = clip_parameters(active_init, limits)
        histogram = eqx.combine(active_init, passive)

    active_opt = active_init
    histogram_opt = histogram
    rwp_lst = []
    rwp_prev = np.inf

    for i_iter in range(max_iter):
        Ns = _get_Ns(histogram_opt, ttheta_deg, n_fwhm)
        active_opt = get_lm_step(
            active_opt, passive, 
            ttheta_deg, y_obs, Ns, damping, 
            limits
        )
        histogram_opt = eqx.combine(active_opt, passive)

        rwp_new = _get_rwp(histogram_opt, ttheta_deg, y_obs, Ns)
        rwp_lst.append(rwp_new)

        if not silent:
            print(rwp_new)

        if abs(rwp_new - rwp_prev) < rwp_tol:
            break

        if rwp_new > rwp_prev:
            break

        rwp_prev = rwp_new

    histogram_opt = jax.block_until_ready(histogram_opt)

    return histogram_opt, i_iter+1, rwp_lst


###############################################################################
# Wrappers for n_fwhm
###############################################################################

def get_rwp(histogram, ttheta_deg, y_obs, *, n_fwhm=5.):
    Ns = _get_Ns(histogram, ttheta_deg, n_fwhm)
    return _get_rwp(histogram, ttheta_deg, y_obs, Ns)
