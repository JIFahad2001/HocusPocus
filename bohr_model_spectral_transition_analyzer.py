"""
Bohr Model Spectral Transition Analyzer
========================================
Calculates the physical properties of photons emitted or absorbed during electron
transitions in hydrogen-like atoms under the Bohr model, and evaluates relativistic
kinematics for an electron carrying equivalent energy.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple


# Fundamental Physical Constants (CODATA recommended values)
SPEED_OF_LIGHT: float = 299_792_458.0              # m/s (c)
PLANCK_CONSTANT: float = 6.62607015e-34            # J*s (h)
ELEMENTARY_CHARGE: float = 1.602176634e-19         # C (e)
ELECTRON_REST_MASS: float = 9.1093837015e-31       # kg (m_e)
RYDBERG_CONSTANT_H: float = 1.09677583e7           # m^-1 (R_H for Hydrogen)
RYDBERG_WAVELENGTH_NM: float = 1e9 / RYDBERG_CONSTANT_H  # ~91.1753 nm (1 / R_H)


# Spectral series mapped by lower principal quantum number (n_lower)
SPECTRAL_SERIES = {
    1: "Lyman (UV)",
    2: "Balmer (Visible / Near-UV)",
    3: "Paschen (Near-IR)",
    4: "Brackett (Mid-IR)",
    5: "Pfund (Mid-IR)",
    6: "Humphreys (Far-IR)",
}

# Electromagnetic spectrum classification (wavelength in nanometers -> description)
SPECTRUM_REGIONS = [
    (0.01, 10.0, "Extreme Ultraviolet / X-ray"),
    (10.0, 380.0, "Ultraviolet (UV)"),
    (380.0, 450.0, "Visible: Violet"),
    (450.0, 475.0, "Visible: Blue"),
    (475.0, 495.0, "Visible: Cyan (Sky Blue)"),
    (495.0, 570.0, "Visible: Green"),
    (570.0, 590.0, "Visible: Yellow"),
    (590.0, 620.0, "Visible: Orange"),
    (620.0, 750.0, "Visible: Red"),
    (750.0, 2500.0, "Near-Infrared (NIR)"),
    (2500.0, 25000.0, "Mid-Infrared (MIR)"),
    (25000.0, 1000000.0, "Far-Infrared (FIR)"),
]


@dataclass(frozen=True)
class TransitionResult:
    """Holds analytical results for an atomic transition and equivalent electron kinematics."""
    n_initial: int
    n_final: int
    transition_type: str
    series_name: str
    spectral_region: str
    wavelength_nm: float
    frequency_hz: float
    photon_energy_j: float
    photon_energy_ev: float
    electron_velocity_mps: float
    lorentz_factor: float
    relativistic_mass_kg: float


def identify_spectrum_region(wavelength_nm: float) -> str:
    """Determine the region of the electromagnetic spectrum for a given wavelength."""
    for lower_bound, upper_bound, label in SPECTRUM_REGIONS:
        if lower_bound <= wavelength_nm < upper_bound:
            return label
    if wavelength_nm < 0.01:
        return "Gamma / High Energy X-ray"
    return "Microwave / Radio"


def calculate_transition(n_from: int, n_to: int) -> TransitionResult:
    """
    Calculates optical and relativistic parameters for a Bohr transition.

    Args:
        n_from: Initial principal quantum number.
        n_to: Final principal quantum number.

    Returns:
        TransitionResult: Complete populated results dataclass.
    """
    if n_from == n_to:
        raise ValueError("Initial and final orbit numbers must be distinct.")
    if n_from < 1 or n_to < 1:
        raise ValueError("Principal quantum numbers must be positive integers (n >= 1).")

    is_emission = n_from > n_to
    n_lower = min(n_from, n_to)
    n_upper = max(n_from, n_to)
    transition_type = "Emission (photon released)" if is_emission else "Absorption (photon absorbed)"

    # Rydberg formula for Hydrogen: 1/lambda = R_H * (1/n_lower^2 - 1/n_upper^2)
    inv_lambda = RYDBERG_CONSTANT_H * ((1.0 / (n_lower ** 2)) - (1.0 / (n_upper ** 2)))
    wavelength_m = 1.0 / inv_lambda
    wavelength_nm = wavelength_m * 1e9

    # Photon properties
    frequency = SPEED_OF_LIGHT / wavelength_m
    photon_energy_j = PLANCK_CONSTANT * frequency
    photon_energy_ev = photon_energy_j / ELEMENTARY_CHARGE

    # Categorization
    series_name = SPECTRAL_SERIES.get(n_lower, f"High-n Series (n_lower = {n_lower})")
    spectral_region = identify_spectrum_region(wavelength_nm)

    # Relativistic Kinematics:
    # If a free stationary electron acquired this kinetic energy: E_k = photon_energy_j
    # Total relativistic energy: E_total = m_0 * c^2 + E_k = gamma * m_0 * c^2
    # gamma = 1 + (E_k / (m_0 * c^2))
    # v = c * sqrt(1 - 1 / gamma^2)
    rest_energy_j = ELECTRON_REST_MASS * (SPEED_OF_LIGHT ** 2)
    gamma = 1.0 + (photon_energy_j / rest_energy_j)
    beta = math.sqrt(1.0 - (1.0 / (gamma ** 2)))
    velocity_mps = beta * SPEED_OF_LIGHT
    relativistic_mass = gamma * ELECTRON_REST_MASS

    return TransitionResult(
        n_initial=n_from,
        n_final=n_to,
        transition_type=transition_type,
        series_name=series_name,
        spectral_region=spectral_region,
        wavelength_nm=wavelength_nm,
        frequency_hz=frequency,
        photon_energy_j=photon_energy_j,
        photon_energy_ev=photon_energy_ev,
        electron_velocity_mps=velocity_mps,
        lorentz_factor=gamma,
        relativistic_mass_kg=relativistic_mass,
    )


def print_report(res: TransitionResult) -> None:
    """Prints a structured, scientific report of the transition analysis."""
    divider = "=" * 65
    subdivider = "-" * 65

    print(f"\n{divider}")
    print(f"       HYDROGEN BOHR MODEL TRANSITION REPORT: n = {res.n_initial} -> n = {res.n_final}")
    print(f"{divider}")
    print(f" Transition Type       : {res.transition_type}")
    print(f" Spectral Series       : {res.series_name}")
    print(f" Classification        : {res.spectral_region}")
    print(f"{subdivider}")
    print(" [ Emitted / Absorbed Photon Properties ]")
    print(f"  * Wavelength         : {res.wavelength_nm:,.4f} nm  ({res.wavelength_nm * 10:,.4f} Å)")
    print(f"  * Frequency          : {res.frequency_hz / 1e12:,.4f} THz  ({res.frequency_hz:.4e} Hz)")
    print(f"  * Photon Energy (eV) : {res.photon_energy_ev:.4f} eV")
    print(f"  * Photon Energy (J)  : {res.photon_energy_j:.4e} J")
    print(f"{subdivider}")
    print(" [ Hypothetical Electron Kinematics (E_k = E_photon) ]")
    print(f"  * Relativistic Speed : {res.electron_velocity_mps / 1000:,.3f} km/s "
          f"({(res.electron_velocity_mps / SPEED_OF_LIGHT) * 100:.4f}% of c)")
    print(f"  * Lorentz Factor (γ) : {res.lorentz_factor:.8f}")
    print(f"  * Relativistic Mass  : {res.relativistic_mass_kg:.6e} kg")
    print(f"  * Rest Mass (m_0)    : {ELECTRON_REST_MASS:.6e} kg")
    print(f"  * Relativistic Gain  : {((res.lorentz_factor - 1.0) * 100):.6e}%")
    print(f"{divider}\n")


def get_positive_int(prompt: str) -> Optional[int]:
    """Safely prompts for a positive integer or allows exit."""
    while True:
        raw = input(prompt).strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        try:
            val = int(raw)
            if val < 1:
                print("  [!] Please enter a positive integer greater than 0.")
                continue
            return val
        except ValueError:
            print("  [!] Invalid input. Please enter a whole integer number or 'q' to quit.")


def main() -> None:
    """Main interactive driver loop."""
    print("=" * 65)
    print("   BOHR ATOMIC MODEL TRANSITION ANALYZER (HYDROGEN ATOM)")
    print("=" * 65)
    print("Type 'q' or 'exit' at any prompt to quit.\n")

    while True:
        try:
            n_from = get_positive_int("Enter starting orbit (n_from / electron origin) : ")
            if n_from is None:
                break

            n_to = get_positive_int("Enter target orbit   (n_to   / electron arrival): ")
            if n_to is None:
                break

            if n_from == n_to:
                print("  [!] Initial and target orbits cannot be the same (no energy transition). Try again.\n")
                continue

            result = calculate_transition(n_from=n_from, n_to=n_to)
            print_report(result)

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as err:
            print(f"\n[Error]: {err}\n")

    print("\nThank you for exploring atomic physics! Happy learning!")


if __name__ == "__main__":
    main()