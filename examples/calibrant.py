import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import xrdmap
import jaxrd

from jax.flatten_util import ravel_pytree


lambda_A = 0.9510908133875442
fname_data = "/scratch/data/calib/calib_al2o3.xye"
fname_cif = "/scratch/data/calib/Corundum.cif"

ttheta_deg, y_obs, y_err = np.loadtxt(fname_data).T

ttheta_lo, ttheta_hi = 15., 53.
i_lo, i_hi = np.searchsorted(ttheta_deg, (ttheta_lo, ttheta_hi))
sl = np.s_[i_lo:i_hi]

ttheta_deg = ttheta_deg[sl]
y_obs = y_obs[sl]
y_err = y_err[sl]

bkg = xrdmap.background.get_background(y_obs)
y_corr = y_obs - bkg

phase = jaxrd.model.Phase.from_cif(
    fname_cif, 
    lambda_A, 
    (ttheta_lo, ttheta_hi),
    profile=jaxrd.model.Profile(W_deg2=2e-2)
)
hist = jaxrd.model.Histogram.from_phases([phase], lambda_A)

schedule = jax.tree_util.tree_map(lambda _: False, hist)
schedule.scales["Corundum"] = True
schedule.zero_deg = True


refine_options = dict(n_fwhm=5., max_iter=100, rwp_tol=1e-4, damping=1e-3)

hist_opt, n_iter, rwp_opt = jaxrd.refine.optimize_schedule(
    hist, schedule, 
    ttheta_deg, y_corr, 
    **refine_options
)
y_calc = jaxrd.model.get_histogram(hist_opt, ttheta_deg, n_fwhm=5.)

schedule.phases["Corundum"].profile.W_deg2 = True
schedule.phases["Corundum"].profile.eta = True

hist_opt2, n_iter, rwp_opt = jaxrd.refine.optimize_schedule(
    hist_opt, schedule, 
    ttheta_deg, y_corr, 
    **refine_options
)
y_calc2 = jaxrd.model.get_histogram(hist_opt2, ttheta_deg, n_fwhm=5.)

schedule.phases["Corundum"].intensities = True
hist_opt3, n_iter, rwp_opt = jaxrd.refine.optimize_schedule(
    hist_opt, schedule, 
    ttheta_deg, y_corr, 
    **refine_options
)
y_calc3 = jaxrd.model.get_histogram(hist_opt3, ttheta_deg, n_fwhm=5.)
