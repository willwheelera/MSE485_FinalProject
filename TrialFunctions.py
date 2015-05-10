import numpy as np
import numpy.linalg as LA
import GenerateStartingFunctions as GSF
from numpy import random

# GenerateTrialFunctions.py
# This contains a function that takes the starting wavefunctions and creates a trial wavefunction
# 1. Compute a Slater determinant
# 2. Multiply a Jastrow factor TODO

KEprefactor  = -GSF.hbar**2 * 0.5/GSF.m_e
q_e2k = GSF.q_e**2 * GSF.k_e

def HydrogenAtom():
    psi_array = np.array([GSF.psi_1s])
    psi_laplacian = np.array([GSF.Lpsi_1s])
    ion_positions = np.array([[0.0,0.0,0.0]])*GSF.a_B
    ion_charges = np.array([1.0])

    wf = WaveFunctionClass()
    wf.setAtomicWavefunctions(psi_array)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setIonPositions(ion_positions)
    wf.setIonCharges(ion_charges)

    #print 'Simulating HydrogenAtom'
    return wf


def H2Molecule(ion_sep, N_e=2):
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
    wf.setNumElectrons(N_e)
    
    #print 'Simulating H2Molecule'
    return wf

def H2OMolecule(bond_length,bond_angle):
    # bond_length is in atomic units of Bohr radius
    xdisp = np.cos(bond_angle)*bond_length * GSF.a_B
    ydisp = np.sin(bond_angle)*bond_length * GSF.a_B
    
    O_atom = GSF.O_Atom(np.array([0,0,0]))
    H_atom1 = GSF.H_Atom(np.array([-xdisp, ydisp, 0]))
    H_atom2 = GSF.H_Atom(np.array([xdisp, ydisp, 0]))
    
    psi_array = np.array([H_atom1.psi_1s, H_atom2.psi_1s, O_atom.psi_1s, O_atom.psi_2s, O_atom.psi_2px, O_atom.psi_2py, O_atom.psi_2pz])
    #psi_laplacian = np.array([GSF.Lpsi_1s, GSF.Lpsi_1s])
    psi_laplacian = []
    
    N_e = 10

    wf = WaveFunctionClass()
    wf.setAtomicWavefunctions(psi_array)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setIonPositions(ion_positions)
    wf.setIonCharges(ion_charges)
    wf.setNumElectrons(N_e)

    #print 'Simulating H2OMolecule'
    return wf

def IonPotentialEnergy(ion_positions,ion_charges):
    V_ion=0.0
    for i in range(0,len(ion_positions)):
       ion_displacements = ion_positions[i+1:] - ion_positions[i]  # only calculate distances to ions not already counted
       ion_distances = np.sqrt(np.sum(ion_displacements*ion_displacements,axis=1))
       #C = np.repeat([ion_charges[i]],len(ion_charges)-i-1,axis=0)
       #Z1Z2 = np.outer(C,ion_charges[i+1:]).diagonal()    #the diagonal of charge array is the Z1*Z2
       Z1Z2= ion_charges[i]*ion_charges[i+1:]
       V_ion += np.sum(1.0*Z1Z2/ion_distances) * q_e2k 
    return V_ion

class WaveFunctionClass:
    # Define the atomic wavefunctions
    psi_array = [] # GSF.getH2Functions()  #generate array of objective basis states
    psi_laplacian = [] # GSF.getH2Laplacians() # get kinetic energy terms of wavefunctions (including hbar^2/2m)
    ion_positions = [] # GSF.ion_positions
    ion_charges = [] # GSF.ion_charges  
    N_ion = len(ion_positions)
    N_e = 1
    N_up = 1
    # Jastrow parameters
    Aee_same = 0.25 # parallel cusp condition, Drummonds et al
    Aee_anti = 0.5 # anti-parallel cusp condition, Drummonds et al
    Bee_same = 1.0 # ?
    Bee_anti = 1.0 # ?
    # Cen = -1*ion_charges # Nucleus cusp condition, Drummonds et al
    Den = 10.0
    # step size for finite difference
    h=0.001

    def setAtomicWavefunctions(self, wfnArray):
        self.psi_array = wfnArray
   
    def setAtomicLaplacians(self, lapArray):
        self.psi_laplacian = lapArray

    def setIonPositions(self, pos):
        self.ion_positions = pos
        self.N_ion = len(pos)
    
    def setIonCharges(self, charges):
        self.ion_charges = charges
        self.Cen = -1*charges

    def setNumElectrons(self, num): # currently implemented in InitializeElectrons
        self.N_e = num
    
    def InitializeElectrons(self,N = N_e):
        #e_positions = self.ion_positions + np.random.randn(self.N,3) * GSF.a_B # generate array of electron positions
        # TODO error catch
        self.N_e = N
        e_positions = self.ion_positions + np.ones((N,3)) * GSF.a_B *(-4) # generate array of electron positions
        return e_positions

    def setNup(self, num):
        self.N_up = num

    # MANY-BODY WAVEFUNCTION
    def PsiManyBody(self, e_positions):
        N_up = self.N_up
        if N_up > 0:
            slater_matrix_up = SlaterMatrix(e_positions[0:N_up], self.ion_positions[0:N_up], self.psi_array[0:N_up])
            slater_det_up = SlaterDeterminant(slater_matrix_up)
        else:
            slater_det_up = 1
        if N_up < self.N_e:
            slater_matrix_down = SlaterMatrix(e_positions[N_up:], self.ion_positions[N_up:], self.psi_array[N_up:])
            slater_det_down = SlaterDeterminant(slater_matrix_down)
        else:
            slater_det_down = 1

        return slater_det_up * slater_det_down * self.Jastrow(e_positions)
    
    def Jastrow(self, e_positions):
        Uen = 0
        Uee = 0
        N_up = self.N_up
        for i in range(len(e_positions)):
            # Compute ion distances from electron i
            ion_disp = self.ion_positions - e_positions[i]
            ion_dist = np.sqrt(np.sum(ion_disp*ion_disp, axis=1))
            #print 'ion_dist',ion_dist
            #print 'Cen',self.Cen
            #print 'numerator',self.Cen*ion_dist
            #print 'denominator',(1+self.Den*ion_dist)
            # update electron-ion energy term
            en_sum = np.sum(self.Cen*ion_dist/(1+self.Den*ion_dist))
            Uen += en_sum
            # Compute electron distances from electron i (only further in the list - count each pair once)
            e_disp = e_positions[i+1:] - e_positions[i]
            e_dist = np.sqrt(np.sum(e_disp*e_disp,axis=1))
            if i < N_up: # if this electron is spin up
                e_same = e_dist[0:N_up-i-1] # electrons [i+1:N_up]
                e_anti = e_dist[N_up-i-1:]
                Uee += np.sum(self.Aee_same*e_same/(1+self.Bee_same*e_same))
                Uee += np.sum(self.Aee_anti*e_anti/(1+self.Bee_anti*e_anti))
            else: # if this electron is spin down
                # all the distances are to other down electrons
                Uee += np.sum(self.Aee_same*e_dist/(1+self.Bee_same*e_dist))
        
        return np.exp(-(Uee + Uen))

    ##########################################
    # LOCAL ENERGY
    def LocalEnergy(self, e_positions, psi_at_rvec):
        # KINETIC TERM
        # We can compute all of the kinetic energy terms given the positions
        # This might be hard to debug...
        # Apparently LA.det will compute determinants of all matrices stacked along dimension 2 at once
        # I am not sure this is any faster... but less for loops :)
        
        useAnalytic = False
        KE = 0
        N = len(e_positions) # 
        
        if useAnalytic:
            deriv_mat = SlaterMatrix(e_positions, self.ion_positions, self.psi_laplacian) # the slater matrix of the laplacians
            
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
        else:
            #Central Finite difference method to get laplacian
            e_posxPlusH = e_positions.copy()
            e_posyPlusH = e_positions.copy()
            e_poszPlusH = e_positions.copy()
            e_posxMinusH = e_positions.copy()
            e_posyMinusH = e_positions.copy()
            e_poszMinusH = e_positions.copy()
            
            psi = self.PsiManyBody
            FDKineticEnergy = 0.0
            for i in range(0,N):
                e_posxPlusH[i,0] += self.h
                e_posyPlusH[i,1] += self.h
                e_poszPlusH[i,2] += self.h
                e_posxMinusH[i,0] += -1.0*self.h
                e_posyMinusH[i,1] += -1.0*self.h
                e_poszMinusH[i,2] += -1.0*self.h
            
            
                FDKineticEnergy += KEprefactor * (-6.0*psi_at_rvec + psi(e_posxPlusH) + psi(e_posyPlusH) + psi(e_poszPlusH) + psi(e_posxMinusH) + psi(e_posyMinusH) + psi(e_poszMinusH))/(self.h*self.h)
            localKineticEnergy = FDKineticEnergy / psi_at_rvec

        
        # POTENTIAL TERM
        V_ion = 0
        V_e = 0
        for i in range(N):
            # electron-ion terms
            ion_displacements = e_positions[i] - self.ion_positions
            ion_distances = np.sqrt(np.sum(ion_displacements*ion_displacements,axis=1))
            V_ion += -np.sum(self.ion_charges/ion_distances) * q_e2k
            
            # electron-electron terms
            e_displacements = e_positions[i] - e_positions[i+1:] # only calculate distances to e- not already counted
            e_distances = np.sqrt(np.sum(e_displacements*e_displacements,axis=1))
            V_e += np.sum(1.0/e_distances) * q_e2k                                                        
            
        #return V_ion + V_e + localKineticEnergy
	return V_ion + V_e + localKineticEnergy



# SLATER DETERMINANT    
def SlaterMatrix(e_positions,ion_positions,fn_array):
    # fn_array has the basis wavefunctions centered at the origin (shifted to the ion_position passed in as argument)
    slater_matrix = np.zeros((len(e_positions),(len(e_positions))))
    for j in range(0, len(fn_array)):
        slater_matrix[j,:] = fn_array[j](e_positions,ion_positions[j,:])  #build slater matrix
    return slater_matrix

def SlaterDeterminant(slater_matrix):
    Nfact = np.math.factorial(len(slater_matrix))
    WF = (1/np.sqrt(Nfact)) * LA.det(slater_matrix)
    return WF

