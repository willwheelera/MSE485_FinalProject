import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import GenerateTrialFunctions as GTF
import GenerateStartingFunctions as GSF
import timing
import sys
#from scipy import optimize

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

def MC_loop(WF, steps=1000, sig=0.5):
    
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
        printtime = 1000
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
#(low,high)=(0.5,3.5)   # guess a reasonale range
#E_L=optimize.minimize_scalar(Etot,method='Golden',bounds=(low,high))
#print E_L.x


# This allows variables to be set via command line arguments
# Arguments must be passed in the form 'varname',value
# args is just sys.argv
def parseArgs(args,x):
    #x = {'numSteps': numSteps, 'separation': separation, 'sigma': sigma}
    #print len(args)
    if len(args) > 1:
      for i in range(1, len(args), 2):
        print args[i], args[i+1]
        x[args[i]] = float(args[i+1])
    print x 
    return int(x['numSteps']), x['separation'], x['sigma']


#############################################################
# RUN SIMULATIONS
#############################################################

if __name__ == '__main__':
    sigma_default = 0.8    # TODO do we need to optimize these to have good acceptance rate?
    
    steps_input = 1000
    separation = 1.0

    x = {'numSteps': steps_input, 'separation': separation, 'sigma': sigma_default}
    steps_input, bond_distance, sigma = parseArgs(sys.argv,x)
    
    WF = GTF.H2Molecule(bond_distance)
    print 'N_electrons:',WF.N_e
    
    #for i in range(1,20):       # loop over different sigma to find minimum
    collection_of_positions, E = MC_loop(WF, steps_input, sigma)
    Eion = GTF.IonPotentialEnergy(WF.ion_positions,WF.ion_charges) 
    Eavg = np.average(E)
    Evar = np.var(E)
    print 'Avg Energy: '+str(Eavg+Eion)
    print 'Var Energy: '+str(Evar)
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
    plt.pcolormesh(xedges,yedges,Hmasked.T) # documentation says it switches axes, need to transpose
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts of Psi')

    #plt.scatter(x,y,c=u'r',s=10)

    # TODO plot average energy
    # ENERGY TRACE
    plt.subplot(2,1,2)
    plt.plot(E)
    #add a horizontal line of Eavg
    plt.axhline(y=Eavg,xmin=0,xmax=len(E),color='r')
    plt.xlabel('Monte Carlo steps')
    plt.ylabel('Energy')
    
    # plot x
    #plt.plot(x[0::4]*4,color='y')
    #plt.axhline(y=np.mean(x[0::4])*4,xmin=0,xmax=len(E),color='g')
    
    #plt.show()
    


