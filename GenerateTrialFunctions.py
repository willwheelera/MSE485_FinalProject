import numpy as np
import numpy.linalg as LA
import GenerateStartingFunctions as GSF
from numpy import random

# GenerateTrialFunctions.py
# This contains a function that takes the starting wavefunctions and creates a trial wavefunction
# 1. Compute a Slater determinant
# 2. Multiply a Jastrow factor TODO

def H2Molecule(ion_sep):
    # ion_sep is in atomic units of Bohr radius
    psi_array = np.array([GSF.psi_1s, GSF.psi_1s])
    psi_laplacian = np.array([GSF.Lpsi_1s, GSF.Lpsi_1s])
    ion_positions = np.array([
        [-0.5*ion_sep, 0, 0],
        [0.5*ion_sep, 0, 0]]) * GSF.a_B
    ion_charges = np.array([1.0, 1.0]) # just Z number
    
    wf = WaveFunctionClass()
    wf.setAtomicWavefunctions(psi_array)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setIonPositions(ion_positions)
    wf.setIonCharges(ion_charges)
    return wf

class WaveFunctionClass:
    # Define the atomic wavefunctions
    psi_array = GSF.getH2Functions()  #generate array of objective basis states
    psi_laplacian = GSF.getH2Laplacians() # get kinetic energy terms of wavefunctions (including hbar^2/2m)
    ion_positions = GSF.ion_positions
    ion_charges = GSF.ion_charges  
    N = len(ion_positions)
    
    def setAtomicWavefunctions(self, wfnArray):
        self.psi_array = wfnArray
   
    def setAtomicLaplacians(self, lapArray):
        self.psi_laplacian = lapArray

    def setIonPositions(self, pos):
        self.ion_positions = pos
        self.N = len(pos)
    
    def setIonCharges(self, charges):
        self.ion_charges = charges

    # MANY-BODY WAVEFUNCTION
    def PsiManyBody(self, e_positions):
        slater_matrix = SlaterMatrix(e_positions, self.ion_positions, self.psi_array)
        return SlaterDeterminant(slater_matrix)
    
    ##########################################
    # LOCAL ENERGY
    def LocalEnergy(self, e_positions, psi_at_rvec):
        # KINETIC TERM
        # We can compute all of the kinetic energy terms given the positions
        # This might be hard to debug...
        # Apparently LA.det will compute determinants of all matrices stacked along dimension 2 at once
        # I am not sure this is any faster... but less for loops :)
        
        deriv_mat = SlaterMatrix(e_positions, self.ion_positions, self.psi_laplacian) # the slater matrix of the laplacians
        N = len(e_positions) # 
        
        allSlaterMats = np.repeat([SlaterMatrix(e_positions, self.ion_positions, self.psi_array)],N,axis=0) # copy this matrix N times along dimension 0
            
        if np.version.version > '1.8':
            for i in range(N):
                # set the "diagonal rows" of this NxNxN matrix to be the second derivatives
                allSlaterMats[i,i,:] = deriv_mat[i,:]
                # First index: slice
                # Second index: matrix row (which position)
                # Third index: matrix column (which wavefunction)
        
            localKineticEnergy = np.sum(LA.det(allSlaterMats)) / psi_at_rvec # add together the determinants of each derivative matrix
        else:
            dets = np.zeros(N)
            for i in range(N):
                 # set the "diagonal rows" of this NxNxN matrix to be the second derivatives
                allSlaterMats[i,i,:] = deriv_mat[i,:]
                dets[i] = LA.det(allSlaterMats[i,:,:])
            localKineticEnergy = np.sum(dets) / psi_at_rvec
    
        # POTENTIAL TERM
        q_e2 = GSF.q_e**2
    
        for i in range(len(e_positions)):
            # electron-ion terms
            S = np.repeat([e_positions[i,:]],N,axis=0)
            ion_displacements = S - self.ion_positions
            ion_distances = np.sqrt(np.sum(ion_displacements*ion_displacements,axis=1))
            V_ion = -np.sum(self.ion_charges/ion_distances) * q_e2
             
            # electron-electron terms
            S = np.repeat([e_positions[i,:]],len(e_positions)-i-1,axis=0)
            e_displacements = S - e_positions[i+1:,:] # only calculate distances to e- not already counted
            e_distances = np.sqrt(np.sum(e_displacements*e_displacements,axis=1))
            V_e = np.sum(1.0/e_distances) * q_e2
    
        return V_ion + V_e + localKineticEnergy


# SLATER DETERMINANT    
def SlaterMatrix(e_positions,ion_positions,fn_array):
    slater_matrix = np.zeros((len(e_positions),(len(e_positions))))
    for j in range(0, len(fn_array)):
        slater_matrix[j,:] = fn_array[j](e_positions,ion_positions[j,:])  #build slater matrix
    return slater_matrix

def SlaterDeterminant(slater_matrix):
    Nfact = np.math.factorial(len(slater_matrix))
    WF = (1/np.sqrt(Nfact)) * LA.det(slater_matrix)
    return WF

