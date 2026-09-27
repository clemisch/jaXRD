import numpy as np
import jax
import jax.numpy as jnp

import jaxrd


lambda_A = 1.
ttheta_lo, ttheta_hi = 10., 50.

# fname = "/scratch/cif/standardized/Corundum_PDF5.cif"
fname = "/scratch/data/calib/Corundum_PDF5.cif"
phase = jaxrd.model.Phase.from_cif(
    fname, 
    lambda_A, 
    (ttheta_lo, ttheta_hi),
    profile=jaxrd.model.Profile(W_deg2=2e-2)
)

phase2 = jaxrd.model.Phase.from_cif(
    fname, 
    lambda_A, 
    (ttheta_lo, ttheta_hi),
    profile=jaxrd.model.Profile(W_deg2=2e-2),
    name="bla"
)
phase2.lattice.a *= 0.9


ttheta_deg = np.linspace(ttheta_lo, ttheta_hi, 2000)
hist = jaxrd.model.Histogram.from_phases([phase, phase2], lambda_A, scales=[1., 2.])
y_calc = jaxrd.model.get_histogram(hist, ttheta_deg)
