import numpy as np
import numpy.linalg as LA
import GenerateStartingFunctions as GSFs
from numpy import random
from numpy import sqrt, sin, cos, exp


# GenerateTrialFunctions.py
# This contains a function that takes the starting wavefunctions and creates a trial wavefunction
# 1. Compute a Slater determinant
# 2. Multiply a Jastrow factor

#Define the electron starting positions
e_positions = GSF.getIonPositions() + np.random.randn(2,3) * GSF.a_B # generate array of electron positions

psi_array = GSF.getH2Functions()  #generate array of objective basis states

def Slater_Determinant(e_positions,psi_array):
    slater_matrix = np.zeros((len(e_positions),(len(e_positions))))
    for j in range(0, len(psi_array)):
        slater_matrix[j,:] = psi_array[j](e_positions) #build slater determinant
    WF = (1/sqrt(np.math.factorial(len(e_positions))))*LA.det(slater_matrix)
    return(WF)











