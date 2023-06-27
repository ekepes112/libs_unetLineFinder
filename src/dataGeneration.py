import numpy as np
import random
from scipy.stats import norm, cauchy
from scipy.special import wofz

def voigt(
    x: np.ndarray,
    sigma: float,
    gamma: float
) -> np.ndarray:
    z = (x + 1j*gamma)/(sigma*np.sqrt(2))
    return np.real(wofz(z))/(sigma*np.sqrt(2*np.pi))

def get_line_boundaries(
    line_profile: np.array
) -> list:
    PERCENTILE_AT_BOUNDARY = .05
    line_max = np.max(line_profile)
    boundary_value = line_max * PERCENTILE_AT_BOUNDARY
    return (line_max, np.where(boundary_value <= line_profile)[0][[0,-1]])

def generate_random_profile(
    profile_length: int,
) -> np.array:
    sigma = np.random.uniform(0, 1)
    gamma = np.random.uniform(0, 1)
    x = np.linspace(-5, 5, profile_length)
    max_int = np.random.randint(65000)
    profile = random.sample([
        norm.pdf(x, loc=0, scale=sigma),
        cauchy.pdf(x, scale=gamma),
        voigt(x, sigma=sigma, gamma=gamma)
    ],1)

    return np.squeeze(profile) * max_int

def add_profile_to_spectrum(
    spectrum: np.array,
    profile: np.array,
    shift: int
) -> np.array:
    spectrum[shift:(shift + len(profile))] += profile
    return spectrum

# def generate_spectrum_with_profiles(
#     wavelengths: np.array,
#     line_count: int,
#     profile_widths: int
# ) -> tuple:
#     spectrum = np.zeros_like(wavelengths)
#     box_coordinates = []
#     line_maxima = []
#     for _ in range(line_count):
#         line_intensities = generate_random_profile(profile_widths)
#         profile_start = np.random.randint(1, len(spectrum) - len(line_intensities))
#         profile_box = get_line_boundaries(line_intensities)
#         box_coordinates.append(
#             profile_box[1] + profile_start
#         )
#         line_maxima.append(profile_box[0])
#         spectrum = add_profile_to_spectrum(
#             spectrum,
#             line_intensities,
#             shift=profile_start
#         )
#     return (spectrum, box_coordinates, line_maxima)

def generate_spectrum_with_profiles(
    wavelengths: np.array,
    line_count: int,
    profile_widths: int
) -> tuple:
    spectrum = np.zeros_like(wavelengths)
    ground_truth = np.zeros_like(wavelengths)
    for _ in range(line_count):
        line_intensities = generate_random_profile(profile_widths)
        profile_start = np.random.randint(1, len(spectrum) - len(line_intensities))
        ground_truth = add_profile_to_spectrum(
            ground_truth,
            line_intensities != 0,
            shift=profile_start
        )
        spectrum = add_profile_to_spectrum(
            spectrum,
            line_intensities,
            shift=profile_start
        )
    return (spectrum, ground_truth)
