import numpy as np
import numpy.linalg as LA
import math
import matplotlib.pyplot as plt
import GenerateTrialFunctions as GTF


def MC_loop(Psi, N):
  # Psi is the trial wavefunction. This function finds the energy of Psi using Monte Carlo
  # N is the number of electrons
  R = InitializePositions(N)




def InitializePositions(N):
  R = numpy.zeros((N,3))
  return R


def UpdatePosition(R,i,simga): #move the electron at the i'th position
    R_update = R
    dr = np.random.randn(3)*sigma
    R_update[i,:] = R_update[i,:] + dr
    return(R_update)

# STARTING MAIN LOOP FOR VQMC
sigma = 0.5
steps = 2000
moves_accepted = 0.0
e_positions_old = numpy.zeros(


def MC_loop(Psi, N):
    # Psi is the trial wavefunction. This function finds the energy of Psi using Monte Carlo
    # N is the number of electrons
    
    # STARTING MAIN LOOP FOR VQMC
    sigma = 0.5
    steps = 2000
    moves_accepted = 0.0
    e_positions_old = numpy.zeros((N,3))
    
    R = InitializePositions(N)
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


