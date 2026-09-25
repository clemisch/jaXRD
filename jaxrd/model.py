import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from dataclasses import dataclass
from functools import partial



@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "a", "b", "c", 
        "alpha", "beta", "gamma", 
    ],
    meta_fields=["crystal_system", "space_group_symbol", "space_group_number"],
)
@dataclass(frozen=True)
class Lattice:
    crystal_system: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group_symbol: str
    space_group_number: int


# @dataclass(frozen=True)
# class PhasePeaks:
#     name: str
#     ttheta_deg: jax.Array
#     intensity: jax.Array
#     hkl: tuple[tuple[int, ...] | None, ...]
#     d_hkl_angstrom: torch.Tensor
#     wavelength_angstrom: float | None = None
#     lattice: Lattice | None = None


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["U_deg2", "V_deg2", "W_deg2", "eta"],
    meta_fields=[],
)
@dataclass
class Profile:
    U_deg2: float = 0.0
    V_deg2: float = 0.0
    W_deg2: float = 1e-2
    eta: float = 0.1


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "ttheta_deg", 
        "intensities", 
        "profile",
        "lattice",
    ],
    meta_fields=["name"],
)
@dataclass
class Peaks:
    name: str
    ttheta_deg: jax.Array
    intensities: jax.Array
    profile: Profile
    lattice: Lattice | None = None

    @classmethod
    def from_args(cls, ttheta_deg, intensities=None, profile=None, lattice=None):
        if intensities is None:
            intensities = jnp.ones_like(ttheta_deg)
        if profile is None:
            profile = Profile()

        assert ttheta_deg.ndim == 1
        assert ttheta_deg.shape == intensities.shape

        peaks = cls(
            name="phase",
            ttheta_deg=ttheta_deg,
            intensities=intensities,
            profile=profile,
            lattice=lattice,
        )

        return peaks


@jax.jit
def get_fwhms(peaks):
    profile = peaks.profile
    fwhms2 = profile.U_deg2 * jnp.square(peaks.ttheta_deg)
    fwhms2 += profile.V_deg2 * peaks.ttheta_deg
    fwhms2 += profile.W_deg2
    fwhms = jnp.sqrt(fwhms2)

    return fwhms


@jax.jit
def get_peak_gauss(center, sigma, area, xx):
    peak = jnp.exp(-0.5 * jnp.square(xx - center) / jnp.square(sigma))
    peak = area * peak / np.sqrt(2. * np.pi) / sigma

    return peak


@jax.jit
def get_peak_lorentz(center, gamma, area, xx):
    peak = gamma / (jnp.square(xx - center) + jnp.square(gamma))
    peak = peak * area / jnp.pi

    return peak


@jax.jit(static_argnames="N")
def get_peaks(peaks, ttheta_deg, N):
    fwhms = get_fwhms(peaks)
    sigmas = fwhms / 2.3548
    gammas = fwhms / 2.

    # TODO: potentially many peaks => vmap/scan
    out = jnp.zeros_like(ttheta_deg)
    for center, intensity, sigma, gamma in zip(
        peaks.ttheta_deg, peaks.intensities, sigmas, gammas
    ):
        i_lo = jnp.searchsorted(ttheta_deg, center) - N//2
        idx = jnp.arange(N) + i_lo

        ttheta_window = ttheta_deg[idx]
        gaussian = get_peak_gauss(center, sigma, intensity, ttheta_window)
        lorentzian = get_peak_lorentz(center, gamma, intensity, ttheta_window)
        peak = peaks.profile.eta * lorentzian + (1 - peaks.profile.eta) * gaussian

        out = out.at[idx].add(
            peak, 
            mode="drop", 
            wrap_negative_indices=False,
            indices_are_sorted=True,
            unique_indices=True,
        )

    return out


@jax.jit(static_argnames="Ns")
def get_model(model, ttheta_deg, Ns):
    fwd = jnp.zeros_like(ttheta_deg)
    for key, N in Ns:
        fwd += get_peaks(model[key], ttheta_deg, N)
    return fwd


def _get_max_size(dx, n_fwhm, fwhms):
    fwhm = np.max(fwhms)
    radius = n_fwhm * fwhm
    size = 2 * radius / dx + 1
    size_binned = np.exp2(np.ceil(np.log2(size)))

    return int(size_binned)


def _get_Ns(model, xx, n_fwhm):
    dx = xx[1] - xx[0]
    Ns = tuple(
        (key, _get_max_size(dx, n_fwhm, model[key].sigmas)) 
        for key in sorted(model)
    )

    return Ns
