import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import GenerateTrialFunctions as GTF
import GenerateStartingFunctions as GSF
import timing
import sys
from scipy import optimize

def UpdatePosition(R,i,sigma): #move the electron at the i'th position
    R_update = R.copy()
    dr = np.random.randn(3)*sigma
    R_update[i,:] = R_update[i,:] + dr
    return(R_update, 1.0) # return new_position, T_ratio
# TODO: is copying the whole array less efficient than the version in the HW?

# TODO finish  Force-bias moves
def ForceBiasMove(wf,e_positions,i,sigma):
    # calculate force (needs WaveFunctionClass to have gradient function)
    # generate move
    # calculate T ratio
    new_positions = e_positions.copy() # TODO
    T_ratio = 1.0 # TODO
    return new_positions, T_ratio # TODO: is copying whole arrays less efficient than in HW?

#############################################################
# STARTING MAIN LOOP FOR VQMC
#############################################################
sigma_default = 0.5 
steps_default = 4000

bond_distance = 1.0
WF = GTF.H2Molecule(2.0)

def MC_loop(steps=1000, sig=0.5):
    
    sigma = sig * GSF.a_B # scale the move distance by Bohr radius
    moves_accepted = 0.0
    
    e_positions = WF.InitializeElectrons()
    e_positions_new = e_positions.copy()
    N = len(e_positions)
    collection_of_positions = np.zeros((2*N*steps,3))
    
    Psi = WF.PsiManyBody(e_positions)
    prob_old = Psi**2
    print('initial prob: ', prob_old)
    E = np.zeros(steps)
    index = 0

    for t in range(0,steps):
        for i in range(0,len(e_positions)):
            
            e_positions_new, T_ratio = UpdatePosition(e_positions,i,sigma) #generate array of new electron poisitons
            
            Psi_new =  WF.PsiManyBody(e_positions_new)
            prob_new = Psi_new**2 #get modulus^2 of new wave function
            ratio = prob_new/prob_old #take the ratio of the squares

            ## A = min(1,ratio) # Acceptance crtierion - this is automatically satisfied in our probability checking
            A = ratio

            ran = np.random.random()
            if A > ran:
                e_positions = e_positions_new
                moves_accepted += 1.0
                prob_old = prob_new
                Psi = Psi_new

            collection_of_positions[index:index+2,:] = e_positions
            index += 2
        
        E[t] = WF.LocalEnergy(e_positions, Psi)
        printtime = 5000
        if (t+1)%printtime == 0: print 'Finished step '+str(t+1)

    print('Acceptance Rate:',(moves_accepted/(2.0*t))*100.0)
    
    return collection_of_positions, E


#############################################################
# Golden Section Search Search in 1D
#############################################################
# a scalar function to call

def Etot(L):
    steps_input=5000
    WF = GTF.H2Molecule(L)
    Eion= GTF.IonPotentialEnergy(WF.ion_positions,WF.ion_charges)  #Ion potential energy
    Eavg=np.average(MC_loop(steps_input)[1]) + Eion 
    return Eavg
    
#define the initial bracket of variable 
"""
(low,high)=(0.5,3.5)   # guess a reasonale range
E_L=optimize.minimize_scalar(Etot,method='Golden',bounds=(low,high))
print E_L.x
"""

#############################################################
# RUN SIMULATIONS
#############################################################

if __name__ == '__main__':
    
    steps_input = steps_default
    
    if len(sys.argv) > 1:
        steps_input = int(sys.argv[1])
    collection_of_positions, E = MC_loop(steps_input)
 
    Eavg=np.average(E)
    print 'Avg Energy: '+str(Eavg)
#############################################################
# PLOT POSITIONS
#############################################################

    x = collection_of_positions[:,0]
    y = collection_of_positions[:,1]
    z = collection_of_positions[:,2]
    
    ### 3D SCATTER PLOT
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x, y, z, marker=".")#, zs)
    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    #plt.show()
   
    # 2D HISTOGRAM
    nbins = 200
    H, xedges, yedges = np.histogram2d(x,y,bins=nbins)
    Hmasked = np.ma.masked_where(H==0,H)
    plt.subplot(2,1,1)
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts of Psi')

    # TODO plot average energy
    # ENERGY TRACE
    plt.subplot(2,1,2)
    plt.plot(E)
    #add a horizontal line of Eavg
    plt.axhline(y=Eavg,xmin=0,xmax=len(E),color='r')
    plt.xlabel('Monte Carlo steps')
    plt.ylabel('Energy')
 
    plt.show()
    


