import numpy as np
import numpy.linalg as LA
import GenerateStartingFunctions
from GenerateStartingFunctions import *
from numpy import random
from numpy import sqrt, sin, cos, exp


# GenerateTrialFunctions.py
# This contains a function that takes the starting wavefunctions and creates a trial wavefunction
# 1. Compute a Slater determinant
# 2. Multiply a Jastrow factor

#Define the electron starting positions

rvec1 = R1+np.random.randn(1,3) #position of first electron
rvec2 = R2+np.random.randn(1,3) #position of second electron
e_positions = np.vstack((rvec1,rvec2))#generate array of electron positions

psi_array = np.array([psi_s1,psi_s2])#,psi_s2,psi_s2]) #generate array of objective basis states
def Slater_Determinant(e_positions,psi_array):
    a = np.zeros((len(e_positions),(len(e_positions))))
    for i in range(0,len(e_positions)):
        for j in range(0, len(psi_array)):
            a[j,i] = psi_array[j](e_positions[i]) #build slater determinate
    WF = (1/sqrt(len(e_positions)))*LA.det(a)
    return(WF)

#print((SD(e_positions,psi_array))**2)

def UpdatePosition(R,i,simga): #move the electron at the i'th position
    R_update = R
    x = np.random.normal(0.0,sigma)
    y = np.random.normal(0.0,sigma)
    z = np.random.normal(0.0,sigma)
    R_update[i,0] = R[i,0]+x
    R_update[i,1] = R[i,1]+y
    R_update[i,2] = R[i,2]+z
    return(R_update)

# STARTING MAIN LOOP FOR VQMC
sigma = 0.5
steps = 2000
moves_accepted = 0.0
for t in range(0,steps):
    for i in range(0,len(e_positions)):
        e_positions_old = e_positions.copy() #generate array of old electron positions
        e_positions_new = UpdatePosition(e_positions,i,sigma) #generate array of new electron poisitons
        prob_old = Slater_Determinant(e_positions_old,psi_array)**2 #get modulus^2 of old wave function
        prob_new = Slater_Determinant(e_positions_new,psi_array)**2 #get modulus^2 of new wave function
        ratio = prob_new/prob_old #take the ratio of the squares
        A = min(1,ratio) # Acceptance crtiera
        ran = np.random.random()
        if A > ran:
            e_positions = e_positions_new
            moves_accepted += 1.0
        else:
            e_positions = e_positions_old
print((moves_accepted/(2.0*t))*100.0)









