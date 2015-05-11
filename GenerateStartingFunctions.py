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
ion_sep = 1.0
R1 = np.array([-0.5*ion_sep,0,0])*a_B
R2 = np.array([0.5*ion_sep,0,0])*a_B
ion_positions[0,:] = R1
ion_positions[1,:] = R2
ion_charges = np.array([1.0,1.0])

# Spherical Harmonics, indexed by l, m
Y00 = (4.0*math.pi)**(-0.5)

def Y1all(rvec, Z=1.0):
  rmag = np.sqrt(np.sum(rvec*rvec,1))
  return np.sqrt(0.75/np.pi) * rvec/rmag

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

def R21(rvec, Z=1.0):
  a = 2*a_B/Z
  rb = np.sqrt(np.sum(rvec*rvec,1)) / a
  return 2*3**(-.5) * a**(-3.5) * rb * np.exp(-rb)
  
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
def psi_1s(e_pos_vec,i_pos,Z=1.0):
  return Y00 * R10(e_pos_vec - i_pos, Z)

#def psi_s1(rvec):
#  return Y00 * R10(rvec-R1)
#
#def psi_s2(rvec):
#  return Y00 * R10(rvec-R2)

def psi_2s(e_pos_vec, i_pos, Z=1.0):
  return Y00 * R20(e_pos_vec - i_pos, Z)

def psi_2p_all(e_pos_vec, i_pos, Z=1.0):
  return Y1all(e_pos_vec - i_pos) * R21(e_pos_vec - i_pos, Z)

def psi_2px(e_pos_vec, i_pos, Z=1.0):
  return Y1all(e_pos_vec - i_pos)[0] * R21(e_pos_vec - i_pos, Z)

def psi_2py(e_pos_vec, i_pos, Z=1.0):
  return Y1all(e_pos_vec - i_pos)[1] * R21(e_pos_vec - i_pos, Z)

def psi_2pz(e_pos_vec, i_pos, Z=1.0):
  return Y1all(e_pos_vec - i_pos)[2] * R21(e_pos_vec - i_pos, Z)

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

class H_atom:
  Z = 1.0
  i_pos = np.zeros(3)
  
  def __init__(self,pos=np.array([0,0,0])):
    self.i_pos = pos

  def setPosition(self, pos):
    self.i_pos = pos

  def psi_1s(self, e_pos_vec):
    return Y00 * R10(e_pos_vec - self.i_pos, self.Z)

class Atom:
  Z = 1.0
  i_pos = np.zeros(3)

  def __init__(self, pos=np.array([0,0,0]), Z=1.0):
    self.i_pos = pos
    self.Z = Z

  last_e_vec = np.zeros(3)
  last_2p_vec = np.zeros(3)
  
  def setPosition(self, pos):
    self.i_pos = pos

  def psi_1s(self, e_pos_vec):
    return Y00 * R10(e_pos_vec - self.i_pos, self.Z)

  def psi_2s(self, e_pos_vec):
    return Y00 * R20(e_pos_vec - self.i_pos, self.Z)
  
  def psi_2p_all(self, e_pos_vec):
    return Y1all(e_pos_vec - self.i_pos) * R21(e_pos_vec - self.i_pos, self.Z)
  
  def psi_2px(self, e_pos_vec):
    if (last_e_vec == e_pos_vec).all():
      return last_2p_vec[0]
    else:
      last_e_vec = e_pos_vec.copy()
      last_2p_vec = Y1all(e_pos_vec - self.i_pos)[0] * R21(e_pos_vec - self.i_pos, self.Z)
      return last_2p_vec[0]
  
  def psi_2py(self, e_pos_vec):
    if (last_e_vec == e_pos_vec).all():
      return last_2p_vec[1]
    else:
      last_e_vec = e_pos_vec.copy() 
      last_2p_vec = Y1all(e_pos_vec - self.i_pos)[0] * R21(e_pos_vec - self.i_pos, self.Z)
      return last_2p_vec[1]

  def psi_2pz(self, e_pos_vec):
    if (last_e_vec == e_pos_vec).all():
      return last_2p_vec[2]
    else:
      last_e_vec = e_pos_vec.copy() 
      last_2p_vec = Y1all(e_pos_vec - self.i_pos)[0] * R21(e_pos_vec - self.i_pos, self.Z)
      return last_2p_vec[2]

