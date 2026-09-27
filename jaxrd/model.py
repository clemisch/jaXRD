import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from dataclasses import dataclass
from functools import partial
from pathlib import Path

# TODO: move this to io.py
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "a", "b", "c", 
        "alpha", "beta", "gamma", 
    ],
    meta_fields=["crystal_system", "spg_str", "spg_int"],
)
@dataclass(frozen=True)
class Lattice:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    crystal_system: str
    spg_str: str
    spg_int: int


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
class Phase:
    name: str
    ttheta_deg: jax.Array
    intensities: jax.Array
    profile: Profile
    lattice: Lattice | None = None

    @classmethod
    def from_args(cls, ttheta_deg, intensities=None, profile=None, lattice=None, name=None):
        if intensities is None:
            intensities = jnp.ones_like(ttheta_deg)
        if profile is None:
            profile = Profile()
        if name is None:
            name = "phase"

        assert ttheta_deg.ndim == 1
        assert ttheta_deg.shape == intensities.shape

        phase = cls(
            name=name,
            ttheta_deg=jnp.array(ttheta_deg),
            intensities=jnp.array(intensities),
            profile=profile,
            lattice=lattice,
        )

        return phase


    @classmethod 
    def from_cif(cls, cif_path, lambda_A, ttheta_range_deg=None, profile=None, name=None):
        cif_path = Path(cif_path)

        if ttheta_range_deg is None:
            ttheta_range_deg = (1e-3, 180. - 1e-3)
        if profile is None:
            profile = Profile()
        if name is None: 
            name = cif_path.stem

        structure = Structure.from_file(cif_path)
        analyzer = SpacegroupAnalyzer(structure)
        calculator = XRDCalculator(wavelength=lambda_A)

        pattern = calculator.get_pattern(
            structure,
            two_theta_range=ttheta_range_deg,
            scaled=True
        )
        assert len(pattern.x) > 0

        hkls = []
        for entries in pattern.hkls:
            if not entries:
                hkls.append(None)
                continue
            hkls.append(entries[0]["hkl"])

        lattice = Lattice(
            a=structure.lattice.a,
            b=structure.lattice.b,
            c=structure.lattice.c,
            alpha=structure.lattice.alpha,
            beta=structure.lattice.beta,
            gamma=structure.lattice.gamma,
            crystal_system=analyzer.get_crystal_system(),
            spg_str=analyzer.get_space_group_symbol(),
            spg_int=analyzer.get_space_group_number(),
        )

        phase = cls(
            name=name,
            ttheta_deg=jnp.array(pattern.x),
            intensities=jnp.array(pattern.y),
            profile=profile,
            lattice=lattice,
        )

        return phase


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "phases",
        "scales",
    ],
    meta_fields=["names"],
)
@dataclass
class Histogram:
    phases: [Phase]
    scales: jax.Array
    lambda_A: float
    names: [str]

    @classmethod
    def from_phases(cls, phases, lambda_A, scales=None, names=None):
        if scales is None:
            scales = jnp.ones(len(phases))
        if names is None:
            names = [phase.name for phase in phases]

        assert len(names) == len(phases)
        assert len(set(names)) == len(names)

        histogram = cls(
            phases=phases,
            scales=scales,
            lambda_A=lambda_A,
            names=names,
        )

        return histogram



@jax.jit
def _get_fwhms(phase):
    profile = phase.profile
    fwhms2 = profile.U_deg2 * jnp.square(phase.ttheta_deg)
    fwhms2 += profile.V_deg2 * phase.ttheta_deg
    fwhms2 += profile.W_deg2
    fwhms = jnp.sqrt(fwhms2)

    return fwhms


def _get_max_size(dx, n_fwhm, fwhms):
    fwhm = np.max(fwhms)
    radius = n_fwhm * fwhm
    size = 2 * radius / dx + 1
    size_binned = np.exp2(np.ceil(np.log2(size)))

    return int(size_binned)


def _get_N(phase, ttheta_deg, n_fwhm):
    delta = ttheta_deg[1] - ttheta_deg[0]
    fwhms = _get_fwhms(phase)
    N = _get_max_size(delta, n_fwhm, fwhms)

    return N


def _get_Ns(model, ttheta_deg, n_fwhm):
    Ns = tuple(
        (key, _get_N(model[key], ttheta_deg, n_fwhm)) 
        for key in sorted(model)
    )

    return Ns


@jax.jit
def _get_peak_gauss(center, sigma, area, xx):
    peak = jnp.exp(-0.5 * jnp.square(xx - center) / jnp.square(sigma))
    peak = area * peak / np.sqrt(2. * np.pi) / sigma

    return peak


@jax.jit
def _get_peak_lorentz(center, gamma, area, xx):
    peak = gamma / (jnp.square(xx - center) + jnp.square(gamma))
    peak = peak * area / jnp.pi

    return peak


@jax.jit(static_argnames="N")
def _get_phase(phase, ttheta_deg, N):
    fwhms = _get_fwhms(phase)
    sigmas = fwhms / 2.3548
    gammas = fwhms / 2.

    # TODO: potentially many peaks => vmap/scan
    out = jnp.zeros_like(ttheta_deg)
    for center, intensity, sigma, gamma in zip(
        phase.ttheta_deg, phase.intensities, sigmas, gammas
    ):
        i_lo = jnp.searchsorted(ttheta_deg, center) - N//2
        idx = jnp.arange(N) + i_lo

        ttheta_window = ttheta_deg[idx]
        gaussian = _get_peak_gauss(center, sigma, intensity, ttheta_window)
        lorentzian = _get_peak_lorentz(center, gamma, intensity, ttheta_window)
        peak = phase.profile.eta * lorentzian + (1 - phase.profile.eta) * gaussian

        out = out.at[idx].add(
            peak, 
            mode="drop", 
            wrap_negative_indices=False,
            indices_are_sorted=True,
            unique_indices=True,
        )

    return out


@jax.jit(static_argnames="Ns")
def _get_histogram(histogram, ttheta_deg, Ns):
    fwd = jnp.zeros_like(ttheta_deg)
    for key, N in Ns:
        fwd += _get_phase(histogram[key], ttheta_deg, N)
    return fwd


###############################################################################
# Wrappers for n_fwhm
###############################################################################

def get_phase(phase, ttheta_deg, *, n_fwhm=4.):
    N = _get_N(phase, ttheta_deg, n_fwhm)
    return _get_phase(phase, ttheta_deg, N)


def get_histogram(histogram, ttheta_deg, *, n_fwhm=4.):
    Ns = _get_Ns(histogram, ttheta_deg, n_fwhm)
    return _get_histogram(histogram, ttheta_deg, Ns)
