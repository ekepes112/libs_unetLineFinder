import numpy as np
import random
from scipy.stats import norm, cauchy
from scipy.special import wofz
import src.config as config


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
        """
        Generate a random profile using the given parameters.

        Returns:
            np.array: A numpy array representing the generated profile.
        """
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
    """
    Calculate the Voigt profile for a given array of values.

    Parameters:
        x (np.ndarray): The array of values for which to compute the Voigt profile.
        sigma (float): The sigma parameter of the Voigt profile.
        gamma (float): The gamma parameter of the Voigt profile.

    Returns:
        np.ndarray: The calculated Voigt profile for the input array of values.
    """
    z = (x + 1j*gamma)/(sigma*np.sqrt(2))
    return np.real(wofz(z))/(sigma*np.sqrt(2*np.pi))

def get_line_boundaries(
    line_profile: np.array,
    percentile_at_boundary: float = .95,
) -> list:
    """
    Generate the line boundaries based on the given line profile.

    Args:
        line_profile (np.array): The line profile to analyze.
        percentile_at_boundary (float, optional): The percentile at which to set the boundary value.
            Defaults to .95.

    Returns:
        list: The indices of the line boundaries.
    """
    line_max = np.max(line_profile)
    boundary_value = line_max * percentile_at_boundary
    return np.where(boundary_value <= line_profile)[0]

def add_profile_to_spectrum(
    spectrum: np.array,
    profile: np.array,
    shift: int
):
    """
    Add a profile to the given spectrum at a specified shift.

    Parameters:
        spectrum (np.array): The spectrum array to modify.
        profile (np.array): The profile array to add to the spectrum.
        shift (int): The shift value to use for adding the profile.

    Returns:
        np.array: The modified spectrum array.
    """
    spectrum[shift:(shift + len(profile))] += profile
    return spectrum

def generate_spectrum_with_profiles(
    wavelengths: np.array,
    line_count: int,
    profile_generator: ProfileGenerator,
    percentile_at_boundary: float = .95,
) -> tuple:
    """
	Generate a spectrum with profiles.

	Parameters:
	    wavelengths (np.array): An array of wavelengths.
	    line_count (int): The number of lines to generate.
	    profile_generator (ProfileGenerator): An instance of the ProfileGenerator class used to generate synthetic emission line profiles.
	    percentile_at_boundary (float, optional): The percentile at which to determine the boundaries of the synthetic emission line. Defaults to 0.95.

	Returns:
	    tuple: A tuple containing the generated spectrum, the ground truth spectrum, and the line maxima.
	"""
    spectrum = np.zeros_like(wavelengths)
    ground_truth = np.zeros_like(wavelengths)
    line_maxima = []
    for _ in range(line_count):
        line_intensities = profile_generator.generate()
        line_maxima.append(np.max(line_intensities))
        profile_start = np.random.randint(1, len(spectrum) - len(line_intensities))
        ground_truth[
            get_line_boundaries(
                line_intensities,
                percentile_at_boundary
            ) + profile_start
        ] += 1
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
    """
    Add noise to a given spectrum.

    Args:
        spectrum (np.array): The spectrum to which noise will be added.
        noise_std (float): The standard deviation of the noise.
        noise_center (float): The center value of the noise distribution.

    Returns:
        np.array: The spectrum with added noise.
    """
    noise = noise_center + (noise_std * np.random.randn(len(spectrum)))
    return spectrum + noise

def add_polynomial_baseline_to_spectrum(
    spectrum: np.array,
    scaling_factor: float
) -> np.array:
    """
    Add a polynomial baseline to a spectrum.

    Parameters:
        spectrum (np.array): The input spectrum.
        scaling_factor (float): The scaling factor for the baseline.

    Returns:
        np.array: The spectrum with the polynomial baseline added.
    """
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
    """
    Adds a blackbody continuum to a given spectrum.

    Args:
        spectrum (np.array): The original spectrum.
        wavelengths (np.array): The wavelengths corresponding to the spectrum.
        blackbody_temperature (float): The temperature of the blackbody in Kelvin.
        scaling_factor (float): The scaling factor to apply to the blackbody spectrum.

    Returns:
        np.array: The spectrum with the added blackbody continuum.
    """
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
    """
    Adds recombination background to a given spectrum. The background is generated by smoothing the input spectrum using a wide Gaussian.

    Parameters:
        spectrum (np.array): The spectrum to which the recombination background will be added.
        blurr_width (int, optional): The width of the blurring profile used for convolution. Defaults to 500.

    Returns:
        np.array: The spectrum with the recombination background added.
    """
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

def augment_spectrum(
    spectrum: np.array,
    **kwargs
) -> np.array:
    """
    Augments the given spectrum by adding recombination background, noise, polynomial baseline, and blackbody continuum.

    Parameters:
        spectrum (np.array): The input spectrum to be augmented.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        np.array: The augmented spectrum.

    """
    augmented_spectrum = add_recombination_background_to_spectrum(spectrum)
    augmented_spectrum = add_noise_to_spectrum(
        augmented_spectrum,
        noise_std=kwargs['noise_std'],
        noise_center=kwargs['noise_center']
    )
    augmented_spectrum = add_polynomial_baseline_to_spectrum(
        augmented_spectrum,
        scaling_factor=kwargs['scaling_factor_for_polynomial']
    )
    augmented_spectrum = add_blackbody_continuum_to_spectrum(
        augmented_spectrum,
        wavelengths=kwargs['wavelengths'],
        blackbody_temperature=np.random.randint(
            config.MIN_TEMPERATURE,
            config.MAX_TEMPERATURE
        ),
        scaling_factor=kwargs['scaling_factor_for_blackbody']
    )
    augmented_spectrum -= np.min(augmented_spectrum)
    augmented_spectrum /= np.max(augmented_spectrum)
    return augmented_spectrum
