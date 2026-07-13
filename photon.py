import math

class Photon:
    def __init__(self, wavelength_m):
        # Constants
        self.c = 299792458.0  # Speed of light in m/s
        self.h = 6.62607015e-34  # Planck constant in J/Hz
        self.hbar = self.h / (2 * math.pi)  # Reduced Planck constant
        
        # Properties
        self.wavelength = wavelength_m
        self.k = (2 * math.pi) / self.wavelength  # Wave number
        self.omega = self.c * self.k  # Angular frequency (omega = c * k)
        
        # Energy and Momentum calculations
        self.energy = self.hbar * self.omega  # E = hbar * omega
        self.momentum = self.hbar * self.k  # p = hbar * k
        
    def verify_logic(self):
        # Testing E = c * p
        energy_from_momentum = self.c * self.momentum
        logic_holds = math.isclose(self.energy, energy_from_momentum, rel_tol=1e-12)
        
        return self.energy, energy_from_momentum, logic_holds

if __name__ == "__main__":
    # Test for a 500 nm (visible light) photon
    photon = Photon(500e-9)
    e, e_p, is_valid = photon.verify_logic()
    print("Photon initialized.")
    print(f"Energy (E = \\hbar * \\omega):      {e:.4e} J")
    print(f"Energy (E = p * c):              {e_p:.4e} J")
    print(f"Logic E = pc validated:          {is_valid}")
