import numpy as np
import random
from scipy.stats import norm, cauchy
from scipy.special import wofz


class ProfileGenerator:
    def __init__(
        self,
        profile_resolution: int,
        min_sigma: float = 0.1,
        max_sigma: float = 0.5,
        min_gamma: float = 0.1,
        max_gamma: float = 0.5,
        max_int: int = 65000
    ):
        self.profile_resolution = profile_resolution
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.max_int = max_int
        self.PROFILE_WIDTH = 5

    def generate(self) -> np.array:
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)
        x = np.linspace(-self.PROFILE_WIDTH, self.PROFILE_WIDTH, self.profile_resolution)
        profile = random.sample([
            norm.pdf(x, loc=0, scale=sigma),
            cauchy.pdf(x, scale=gamma),
            voigt(x, sigma=sigma, gamma=gamma)
        ],1)
        profile -= np.min(profile)
        return np.squeeze(profile) * np.random.randint(self.max_int)


def voigt(
    x: np.ndarray,
    sigma: float,
    gamma: float
) -> np.ndarray:
    z = (x + 1j*gamma)/(sigma*np.sqrt(2))
    return np.real(wofz(z))/(sigma*np.sqrt(2*np.pi))

def get_line_boundaries(line_profile: np.array) -> list:
    PERCENTILE_AT_BOUNDARY = .15
    line_max = np.max(line_profile)
    boundary_value = line_max * PERCENTILE_AT_BOUNDARY
    return np.where(boundary_value <= line_profile)[0]

def add_profile_to_spectrum(
    spectrum: np.array,
    profile: np.array,
    shift: int
):
    spectrum[shift:(shift + len(profile))] += profile
    return spectrum

def generate_spectrum_with_profiles(
    wavelengths: np.array,
    line_count: int,
    profile_generator: ProfileGenerator
) -> tuple:
    spectrum = np.zeros_like(wavelengths)
    ground_truth = np.zeros_like(wavelengths)
    line_maxima = []
    for _ in range(line_count):
        line_intensities = profile_generator.generate()
        line_maxima.append(np.max(line_intensities))
        profile_start = np.random.randint(1, len(spectrum) - len(line_intensities))
        ground_truth[get_line_boundaries(line_intensities) + profile_start] += 1
        spectrum = add_profile_to_spectrum(
            spectrum,
            line_intensities,
            shift=profile_start
        )
    return (spectrum, ground_truth, line_maxima)

def add_noise_to_spectrum(
    spectrum: np.array,
    noise_std: float,
    noise_center: float,
):
    noise = noise_center + (noise_std * np.random.randn(len(spectrum)))
    return spectrum + noise

def add_polynomial_baseline_to_spectrum(
    spectrum: np.array,
    scaling_factor: float
) -> np.array:
    baseline_degree = np.random.randint(5,15)
    baseline = np.polynomial.chebyshev.chebval(
        x=np.linspace(0,.95,len(spectrum)),
        c=np.random.random(baseline_degree),
    )
    baseline -= np.min(baseline)
    baseline /= np.max(baseline)
    baseline *= scaling_factor
    return spectrum + baseline

def add_blackbody_continuum_to_spectrum(
    spectrum: np.array,
    wavelengths: np.array,
    blackbody_temperature: float,
    scaling_factor: float,
) -> np.array:
    k_b = 1.380649e-23
    h = 6.626070e-34
    c = 299792458
    blackbody_spectrum = ((2*np.pi*h*(c**2))/((wavelengths*1e-9)**5))*(np.exp((h*c)/(wavelengths*(1e-9)*k_b*blackbody_temperature))-1)**(-1)
    blackbody_spectrum /= np.max(blackbody_spectrum)
    blackbody_spectrum *= scaling_factor
    return spectrum + blackbody_spectrum

def add_recombination_background_to_spectrum(
    spectrum: np.array,
    blurr_width: int = 500,
) -> np.array:
    blurring_profile = norm.pdf(
        x=np.linspace(
            -blurr_width,
            blurr_width,
            blurr_width*2
        ),
        loc=0,
        scale=blurr_width
    ) * 5
    recombination_background = np.convolve(
        spectrum,
        blurring_profile,
        mode='same'
    )
    return spectrum + recombination_background
