import numpy as np
import numpy.linalg as LA
import GenerateStartingFunctions as GSF
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# GenerateTrialFunctions.py
# This contains a function that takes the starting wavefunctions and creates a trial wavefunction
# 1. Compute a Slater determinant
# 2. Multiply a Jastrow factor

# Define the atomic wavefunctions
psi_array = GSF.getH2Functions()  #generate array of objective basis states
psi_laplacian = GSF.getH2Laplacians() # get kinetic energy terms of wavefunctions (including hbar^2/2m)

def setAtomicWavefunctions(wfnArray):
    psi_array = wfnArray

def SlaterMatrix(e_positions,psi_array):
    slater_matrix = np.zeros((len(e_positions),(len(e_positions))))
    for j in range(0, len(psi_array)):
        slater_matrix[j,:] = psi_array[j](e_positions) #build slater determinant
    return slater_matrix

def SlaterDeterminant(slater_matrix):
    WF = (1/sqrt(np.math.factorial(len(e_positions)))) * LA.det(slater_matrix)
    return WF

def PsiManyBody(e_positions):
    slater_matrix = SlaterMatrix(e_positions)
    return Slater_Determinant(slater_matrix)

##########################################
# TODO don't need this function
def LaplacianSlater(e_positions,i):
    slater_mat = SlaterMatrix(e_positions)
    second_derivs = numpy.zeros(len(e_positions))
    for j in range(len(psi_laplacian)):
        second_derivs[j] = psi_laplacian[j](np.array([e_positions[i]]))
    slater_mat[i,:] = second_derivs
    return slater_mat

# This is for calculating kinetic energy
# TODO don't need this 
def LaplacianPsi(e_positions,i):
    slater_mat = LaplacianSlater(e_positions,i)
    return SlaterDeterminant(slater_mat)

###########################################

def KineticTerm(e_positions):
    # We can compute all of the kinetic energy terms given the positions
    # This might be hard to debug...
    # Apparently LA.det will compute determinants of all matrices stacked along dimension 2 at once
    # I am not sure this is any faster... but less for loops :)
    
    deriv_mat = SlaterMatrix(e_positions,psi_laplacian) # the slater matrix of the laplacians
    N = len(e_positions) # 
    allSlaterMats = numpy.repeat(SlaterMatrix(e_positions,psi_array),N,axis=2) # copy this matrix N times along dimension 2
    for i in range(N):
        allSlaterMats[i,:,i] = deriv_mat[i,:] # set the "diagonal rows" of this NxNxN matrix to be the second derivatives
    localKineticEnergy = numpy.sum(LA.det(allSlaterMats)) # add together the determinants of each derivative matrix
    return localKineticEnergy
