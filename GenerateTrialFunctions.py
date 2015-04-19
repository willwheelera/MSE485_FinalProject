import numpy as np
import numpy.linalg as LA
import GenerateStartingFunctions as GSF
from numpy import random

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


# GenerateTrialFunctions.py
# This contains a function that takes the starting wavefunctions and creates a trial wavefunction
# 1. Compute a Slater determinant
# 2. Multiply a Jastrow factor

# Define the atomic wavefunctions
psi_array = GSF.getH2Functions()  #generate array of objective basis states
psi_laplacian = GSF.getH2Laplacians() # get kinetic energy terms of wavefunctions (including hbar^2/2m)
R = GSF.ion_positions
Z = GSF.ion_charges  
N = len(R)

def setAtomicWavefunctions(wfnArray):
    psi_array = wfnArray

def SlaterMatrix(e_positions,psi_array):
    slater_matrix = np.zeros((len(e_positions),(len(e_positions))))
    for j in range(0, len(psi_array)):
        slater_matrix[j,:] = psi_array[j](e_positions)  #build slater determinant
    return slater_matrix

def SlaterDeterminant(slater_matrix):
    Nfact = np.math.factorial(len(slater_matrix))
    WF = (1/np.sqrt(Nfact)) * LA.det(slater_matrix)
    return WF

def PsiManyBody(e_positions):
    slater_matrix = SlaterMatrix(e_positions,psi_array)
    print (e_positions)
    print(R)
    disp_matrix = np.repeat([e_positions],N,axis=0).transpose() - np.repeat([R],N,axis=0)
    # return Psi_MB and (r-R)Psi_MB (MB = many-body)
    return SlaterDeterminant(slater_matrix), SlaterDeterminant(slater_matrix*disp_matrix)

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
# TODO move these functions to another file

def KineticTerm(e_positions):
    # We can compute all of the kinetic energy terms given the positions
    # This might be hard to debug...
    # Apparently LA.det will compute determinants of all matrices stacked along dimension 2 at once
    # I am not sure this is any faster... but less for loops :)
    
    deriv_mat = SlaterMatrix(e_positions,psi_laplacian) # the slater matrix of the laplacians
    N = len(e_positions) # 
    
    allSlaterMats = np.repeat([SlaterMatrix(e_positions,psi_array)],N,axis=0) # copy this matrix N times along dimension 0
        
    if np.version.version > '1.8':
        for i in range(N):
            allSlaterMats[i,i,:] = deriv_mat[i,:] # set the "diagonal rows" of this NxNxN matrix to be the second derivatives
            # First index: slice
            # Second index: matrix row (which position)
            # Third index: matrix column (which wavefunction)
    
        localKineticEnergy = np.sum(LA.det(allSlaterMats)) # add together the determinants of each derivative matrix
    else:
        dets = np.zeros(N)
        for i in range(N):
            allSlaterMats[i,i,:] = deriv_mat[i,:] # set the "diagonal rows" of this NxNxN matrix to be the second derivatives
            dets[i] = LA.det(allSlaterMats[i,:,:])
        localKineticEnergy = np.sum(dets)

    return localKineticEnergy

def PotentialTerm(e_positions):
    
    q_e2 = GSF.q_e**2

    for i in range(len(e_positions)):
        # electron-ion terms
        S = np.repeat([e_positions[i,:]],len(R),axis=0)
        ion_displacements = S-R
        ion_distances = np.sqrt(np.sum(ion_displacements*ion_displacements,axis=1))
        V_ion = np.sum(Z/ion_distances) * q_e2
         
        # electron-electron terms
        S = np.repeat([e_positions[i,:]],len(e_positions)-i-1,axis=0)
        e_displacements = S-e_positions[i+1:,:] # only calculate distances to e- not already counted
        e_distances = np.sqrt(np.sum(e_displacements*e_displacements,axis=1))
        V_e = np.sum(1.0/e_distances) * q_e2

    return V_ion + V_e





