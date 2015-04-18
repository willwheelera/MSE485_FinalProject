import numpy as np
import numpy.linalg as LA
import math

# This file generates starting wavefunctions for a particular configuration.
# 1. Write functions analytically
# 2. Load lookup table from txt

# Hydrogen H2 Molecule from atomic wavefunctions
# Atomic units
a_B = 1.0 # Bohr radius
m_e = 1.0 # electron mass
q_e = 1.0 # electron charge
hbar = 1.0 # reduced planck constant
k_e = 1.0 # Coulomb's constant (1/4\pi\epsilon_0)


# Define atom positions
R1 = np.array([-0.5,0,0])*a_B
R2 = np.array([0.5,0,0])*a_B

# Spherical Harmonics
def Y00(rvec):
  return  (4.0*math.pi)**(-0.5)

# Radial functions
def R10(rvec):
  r = np.sqrt(np.sum(rvec*rvec))
  return 2*a_B**(-1.5)*np.exp(-r/a_B)

# S orbitals
def psi_s1(rvec):
  return Y00(rvec-R1)*R10(rvec-R1)

def psi_s2(rvec):
  return Y00(rvec-R2)*R10(rvec-R2)


# Retrieve the functions
def getH2Functions():
  return (psi_s1, psi_s2)
