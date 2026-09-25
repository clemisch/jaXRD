import jax
import equinox as eqx


@jax.jit(static_argnames="Ns")
def get_cost_schedule(params_active, params_passive, xx, yy, Ns):
    model = eqx.combine(params_active, params_passive)
    cost = _get_cost_windowed(model, xx, yy, Ns)

    return cost


@jax.jit(static_argnames="Ns")
def _get_grad_schedule(params_active, params_passive, xx, yy, Ns):
    fn_grad = eqx.filter_grad(_get_cost_schedule)
    grad = fn_grad(params_active, params_passive, xx, yy, Ns)
    return grad


def get_cost_schedule(params_active, params_passive, xx, yy, n_fwhm):
    model = eqx.combine(params_active, params_passive)
    Ns = _get_Ns(model, xx, n_fwhm)
    cost = _get_cost_schedule(params_active, params_passive, xx, yy, Ns)

    return cost


def get_grad_schedule(params_active, params_passive, xx, yy, n_fwhm):
    model = eqx.combine(params_active, params_passive)
    Ns = _get_Ns(model, xx, n_fwhm)
    grad = _get_grad_schedule(params_active, params_passive, xx, yy, Ns)

    return grad
