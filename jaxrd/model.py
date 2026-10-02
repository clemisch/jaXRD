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
@dataclass
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


    @jax.jit
    def get_center(self, h, k, l, lambda_A):
        a = self.a
        b = self.b
        c = self.c

        if self.crystal_system == "cubic":
            d2_inv = (h * h + k * k + l * l) / (a * a)
        elif self.crystal_system == "tetragonal":
            d2_inv = (h * h + k * k) / (a * a) + (l * l) / (c * c)
        elif self.crystal_system == "orthorhombic":
            d2_inv = (h * h) / (a * a) + (k * k) / (b * b) + (l * l) / (c * c)
        elif self.crystal_system in {"hexagonal", "trigonal"}:
            d2_inv = (4.0 / 3.0) * (h * h + h * k + k * k) / (a * a) + (l * l) / (c * c)
        elif self.crystal_system in {"monoclinic", "triclinic"}:
            alpha = jnp.radians(self.alpha)
            beta = jnp.radians(self.beta)
            gamma = jnp.radians(self.gamma)
            metric = jnp.array((
                a * a, 
                a * b * jnp.cos(gamma), 
                a * c * jnp.cos(beta),
                a * b * jnp.cos(gamma), 
                b * b, 
                b * c * jnp.cos(alpha),
                a * c * jnp.cos(beta), 
                b * c * jnp.cos(alpha),
                c * c,
            )).reshape((3, 3))
            hkl = jnp.stack((h, k, l))
            d2_inv = hkl @ jnp.linalg.solve(metric, hkl)
        else:
            raise ValueError(f"Unknown crystal system: {self.crystal_system!r}")

        d_hkl = 1. / jnp.sqrt(d2_inv)
        centers_rad = 2. * jnp.arcsin(lambda_A / 2. / d_hkl)
        centers_deg = jnp.degrees(centers_rad)

        return centers_deg


    @jax.jit
    def get_centers(self, hkls, lambda_A):
        hs = hkls[:, 0]
        ks = hkls[:, 1]
        ls = hkls[:, -1]
        centers_deg = jax.vmap(self.get_center, (0, 0, 0, None))(hs, ks, ls, lambda_A)

        return centers_deg


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
        "intensities", 
        "hkls", 
        "profile",
        "lattice",
    ],
    meta_fields=["name"],
)
@dataclass
class Phase:
    name: str
    intensities: jax.Array
    hkls: jax.Array
    profile: Profile
    lattice: Lattice


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
            scaled=False
        )
        assert len(pattern.x) > 0

        hkls = []
        for entries in pattern.hkls:
            if not entries:
                hkls.append(None)
                continue
            hkls.append(entries[0]["hkl"])
        hkls = jnp.asarray(hkls, dtype="int32")

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
            intensities=jnp.array(pattern.y),
            hkls=hkls,
            profile=profile,
            lattice=lattice,
        )

        return phase


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "phases",
        "scales",
        "lambda_A",
        "zero_deg",
    ],
    meta_fields=[],
)
@dataclass
class Histogram:
    phases: dict[str, Phase]
    scales: dict[str, float]
    lambda_A: float
    zero_deg: float = 0.0


    def __repr__(self):
        phases = ", ".join(self.phases)
        return f"Histogram[{phases}]"


    @classmethod
    def from_phases(cls, phases, lambda_A, scales=None, names=None, zero_deg=0.0):
        if scales is None:
            scales = jnp.ones(len(phases))
        if names is None:
            names = [phase.name for phase in phases]

        assert len(names) == len(phases)
        assert len(set(names)) == len(names)

        histogram = cls(
            phases=dict(zip(names, phases)),
            scales=dict(zip(names, scales)),
            lambda_A=lambda_A,
            zero_deg=zero_deg
        )

        return histogram



@jax.jit
def _get_fwhms(phase, lambda_A):
    centers_deg = phase.lattice.get_centers(phase.hkls, lambda_A)

    fwhms2 = phase.profile.U_deg2 * jnp.square(centers_deg)
    fwhms2 += phase.profile.V_deg2 * centers_deg
    fwhms2 += phase.profile.W_deg2
    fwhms = jnp.sqrt(fwhms2)

    return fwhms


def _get_max_size(dx, n_fwhm, fwhms):
    fwhm = np.max(fwhms)
    radius = n_fwhm * fwhm
    size = 2 * radius / dx 
    size_binned = np.exp2(np.ceil(np.log2(size)))  # round to next power of 2

    return int(size_binned)


def _get_N(phase, ttheta_deg, lambda_A, n_fwhm):
    delta = ttheta_deg[1] - ttheta_deg[0]
    # TODO: this might become expensive for every LM iterations
    fwhms = _get_fwhms(phase, lambda_A)
    N = _get_max_size(delta, n_fwhm, fwhms)

    return N


def _get_Ns(histogram, ttheta_deg, n_fwhm):
    Ns = tuple(
        (key, _get_N(histogram.phases[key], ttheta_deg, histogram.lambda_A, n_fwhm)) 
        for key in sorted(histogram.phases)
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
def _get_phase(phase, ttheta_deg, lambda_A, N):

    def worker(carry, x):
        center, intensity, sigma, gamma = x

        i_lo = jnp.searchsorted(ttheta_deg, center) - N//2
        idx = jnp.arange(N) + i_lo

        ttheta_window = ttheta_deg[idx]
        gaussian = _get_peak_gauss(center, sigma, intensity, ttheta_window)
        lorentzian = _get_peak_lorentz(center, gamma, intensity, ttheta_window)
        peak = phase.profile.eta * lorentzian + (1 - phase.profile.eta) * gaussian

        carry = carry.at[idx].add(
            peak, 
            mode="drop", 
            wrap_negative_indices=False,
            indices_are_sorted=True,
            unique_indices=True,
        )

        return carry, None


    fwhms = _get_fwhms(phase, lambda_A)
    sigmas = fwhms / 2.3548
    gammas = fwhms / 2.

    centers_deg = phase.lattice.get_centers(phase.hkls, lambda_A)
    
    out = jnp.zeros_like(ttheta_deg)
    out, _ = jax.lax.scan(
        worker, 
        out, 
        (centers_deg, phase.intensities, sigmas, gammas),
        unroll=1
    )

    return out


@jax.jit(static_argnames="Ns")
def _get_histogram(histogram, ttheta_deg, Ns):
    ttheta_deg = ttheta_deg + histogram.zero_deg

    fwd = jnp.zeros_like(ttheta_deg)
    for key, N in Ns:
        fwd += histogram.scales[key] * _get_phase(
            histogram.phases[key], 
            ttheta_deg, 
            histogram.lambda_A, 
            N,
        )

    return fwd


###############################################################################
# Wrappers for n_fwhm
###############################################################################

def get_phase(phase, ttheta_deg, lambda_A, *, n_fwhm=5.):
    N = _get_N(phase, ttheta_deg, lambda_A, n_fwhm)
    return _get_phase(phase, ttheta_deg, lambda_A, N)


def get_histogram(histogram, ttheta_deg, *, n_fwhm=5.):
    Ns = _get_Ns(histogram, ttheta_deg, n_fwhm)
    return _get_histogram(histogram, ttheta_deg, Ns)
