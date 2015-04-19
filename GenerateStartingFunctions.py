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
ion_positions = np.zeros((2,3))
ion_sep = 2.0
R1 = np.array([-0.5*ion_sep,0,0])*a_B
R2 = np.array([0.5*ion_sep,0,0])*a_B
ion_positions[0,:] = R1
ion_positions[1,:] = R2
ion_charges = np.array([1.0,1.0])

# Spherical Harmonics
Y00 = (4.0*math.pi)**(-0.5)

# Radial functions
# Note: this function requires that position coordinates are along dimension 1 of the array (not 0)
def R10(rvec):
  r = np.sqrt(np.sum(rvec*rvec,1))
  return 2*a_B**(-1.5)*np.exp(-r/a_B)

def Laplacian_R10(rvec):
  r = np.sqrt(np.sum(rvec*rvec,1))
  return 2*a_B**(-3.5)*np.exp(-r/a_B)

# S orbitals
def psi_s1(rvec):
  return Y00 * R10(rvec-R1)

def psi_s2(rvec):
  return Y00 * R10(rvec-R2)

# Laplacian of S orbitals
def Lpsi_s1(rvec):
  return Y00 * Laplacian_R10(rvec-R1) # TODO

def Lpsi_s2(rvec):
  return Y00 * Laplacian_R10(rvec-R2)  # TODO


# Retrieve the functions
def getH2Functions():
  return np.array([psi_s1, psi_s2])

# Retrieve the Laplacians of the wavefunctions (can be defined analytically)
def getH2Laplacians():
  return np.array([Lpsi_s1, Lpsi_s2])

def getIonPositions():
  return ion_positions

def getIonCharges():
  return ion_charges

def InitializeElectrons():
    e_positions = ion_positions + np.random.randn(2,3) * a_B # generate array of electron positions
    return e_positions

