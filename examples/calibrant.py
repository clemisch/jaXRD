import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import xrdmap
import jaxrd

from jax.flatten_util import ravel_pytree

jax.config.update('jax_enable_x64', True)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


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
)
hist = jaxrd.model.Histogram.from_phases([phase], lambda_A)



schedule = jaxrd.refine.get_schedule(hist)
schedule.scales["Corundum"] = True
schedule.zero_deg = True


refine_options = dict(n_fwhm=5., max_iter=10, rwp_tol=1e-4, damping=1e-3)

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
schedule.scales["Corundum"] = False
hist_opt3, n_iter, rwp_opt = jaxrd.refine.optimize_schedule(
    hist_opt2, schedule, 
    ttheta_deg, y_corr, 
    **refine_options
)
y_calc3 = jaxrd.model.get_histogram(hist_opt3, ttheta_deg, n_fwhm=5.)



import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, layout="tight")
ax.plot(ttheta_deg, y_corr, "o", mfc="white", mec="black", mew=0.8, ms=5.)
ax.plot(ttheta_deg, y_calc2, color="C0")
ax.plot(ttheta_deg, y_calc3, color="C1")
ax.grid()
