import numpy as np
import jax
import jax.numpy as jnp

import jaxrd

peaks = jaxrd.model.Peaks.from_args(
    np.array((1., 2., 3.)), 
    profile=jaxrd.model.Profile(eta=0.1)
)
model = {
    "bla": peaks
}

xx = linspace(0, 10, 1000)
fwd = jaxrd.model.get_model(model, xx)



fname = "/scratch/cif/standardized/Corundum_PDF5.cif"
peaks = jaxrd.model.Peaks.from_cif(fname, 1.0, (10., 50.))
