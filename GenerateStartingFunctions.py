import numpy as np
import numpy.linalg as LA
import math

# This file generates starting wavefunctions for a particular configuration.
# 1. Write functions analytically
# 2. Load lookup table from txt

# Hydrogen H2 Molecule from atomic wavefunctions
# Atomic units
m_e = 1.0 #* 9.11e-31 # electron mass
q_e = 1.0 #* 1.602e-19 # electron charge
hbar = 1.0 #* 1.0546e-34 # reduced planck constant
k_e = 1.0 #/ (4*np.pi * 8.854e-12) # Coulomb's constant (1/4\pi\epsilon_0)
a_B = 1.0 #* hbar*hbar / (k_e*m_e*q_e**2) # Bohr radius

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
def R10(rvec, Z=1.0):
  r = np.sqrt(np.sum(rvec*rvec,1))
  a = a_B/Z
  return 2*a**(-1.5)*np.exp(-r/a)

def R20(rvec, Z=1.0):
  a = 2*a_B/Z
  rb = np.sqrt(np.sum(rvec*rvec,1)) / a
  return 2*a**(-1.5)*(1.0-rb)*np.exp(-rb)
  
def Laplacian_R10(rvec, Z=1.0):
  a = a_B/Z
  rb = np.sqrt(np.sum(rvec*rvec,1)) / a
  #return 2*a**(-3.5)*np.exp(-r/a)
  return 2*a**(-3.5)*(1.0 - 2.0/rb) * np.exp(-rb)

def Laplacian_R20(rvec, Z=1.0):
  a = 2*a_B/Z
  rb = np.sqrt(np.sum(rvec*rvec,1)) / a # change of vars makes formula simpler: rb = r/a = r*Z / 2*a_B
  #return 4*(2*a_B)**(-3.5)*(1.5-(r/(4.0*a_B)))*np.exp(-2.0*r/a_B)
  #return (1.0/(2.0*math.sqrt(2.0)))*a_B**(-3.5)*(1.5-(r/(4.0*a_B)))*np.exp(-2.0*r/a_B)
  return -2*a**(-3.5)/rb * (rb-4)*(rb-1)*np.exp(-rb)

####################################
# 

# S orbitals
def psi_1s(e_pos_vec,i_pos):
  return Y00 * R10(e_pos_vec - i_pos)

def psi_s1(rvec):
  return Y00 * R10(rvec-R1)

def psi_s2(rvec):
  return Y00 * R10(rvec-R2)

# Laplacian of S orbitals
def Lpsi_1s(e_pos_vec, i_pos):
  return Y00 * Laplacian_R10(e_pos_vec - i_pos) * (-hbar*hbar*0.5/m_e)

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



