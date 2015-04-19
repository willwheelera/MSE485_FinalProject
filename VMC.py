import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import GenerateTrialFunctions as GTF
import GenerateStartingFunctions as GSF
import timing

def MC_loop(Psi, N):
  # Psi is the trial wavefunction. This function finds the energy of Psi using Monte Carlo
  # N is the number of electrons
  R = InitializePositions(N)

# TODO
# We probably don't need this function
#def InitializePositions(N):
#  R = numpy.zeros((N,3))
#  return R


def UpdatePosition(R,i,simga): #move the electron at the i'th position
    R_update = R.copy()
    dr = np.random.randn(3)*sigma
    R_update[i,:] = R_update[i,:] + dr
    return(R_update)

#############################################################
# STARTING MAIN LOOP FOR VQMC
#############################################################
sigma = 0.5
steps = 4000
moves_accepted = 0.0

def MC_loop():
    
    sigma = 0.5
    steps = 100000
    moves_accepted = 0.0
    
    e_positions = GSF.InitializeElectrons()
    e_positions_new = e_positions.copy()
    N = len(e_positions)
    collection_of_positions = np.zeros((2*N*steps,3))
    
    Psi = GTF.PsiManyBody(e_positions)
    prob_old = Psi**2
    
    E = np.zeros(steps)
    index = 0

    for t in range(0,steps):
        for i in range(0,len(e_positions)):
            # TODO I don't think we need this line: - Will
            # e_positions_old = e_positions.copy() #generate array of old electron positions
            
            e_positions_new = UpdatePosition(e_positions,i,sigma) #generate array of new electron poisitons
            
            # I don't think we need this: -Will
            # prob_old = GTF.PsiManyBody(e_positions_old)**2 #get modulus^2 of old wave function
            
            Psi_new =  GTF.PsiManyBody(e_positions_new)
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
            #else:
                # I don't think we need this: - Will
                # e_positions = e_positions_old
            collection_of_positions[index:index+2,:] = e_positions
            index += 2
        
        E[t] = GTF.KineticTerm(e_positions)/Psi + GTF.PotentialTerm(e_positions)
        printtime = 5000
        if (t+1)%printtime == 0: print 'Finished step '+str(t+1)

    print('Acceptance Rate:',(moves_accepted/(2.0*t))*100.0)
    
    return collection_of_positions, E


#############################################################
# RUN SIMULATIONS
#############################################################

if __name__ == '__main__':
    
    collection_of_positions, E = MC_loop()

#############################################################
# PLOT POSITIONS
#############################################################

    x = collection_of_positions[:,0]
    y = collection_of_positions[:,1]
    z = collection_of_positions[:,2]
    
    # 3D SCATTER PLOT
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
    #fig2 = plt.figure()
    plt.subplot(2,1,1)
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts of Psi')
    
    # ENERGY TRACE
    plt.subplot(2,1,2)
    plt.plot(E)
    plt.xlabel('Monte Carlo steps')
    plt.ylabel('Energy')

    plt.show()
    


