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
    
testbool = True

def HydrogenAtom():
    H_atom = GSF.Atom(pos=np.array([0,0,0],Z=1.0))
    psi_array = np.array([H_atom.psi_1s])
    psi_laplacian = []
    ion_positions = np.array([H_atom.i_pos])
    ion_charges = np.array([H_atom.Z]) 
    N_e = 1
    
    wf = WaveFunctionClass()
    wf.setUpWavefunctions(psi_array)
    wf.setDownWavefunctions(psi_array)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setIonPositions(ion_positions)
    wf.setIonCharges(ion_charges)
    wf.setNumElectrons(N_e) 
    # Up or down doesn't matter for 1 electron; note the default is 0 in the class
    wf.N_up = 1
    #print 'Simulating HydrogenAtom'
    return wf

def HeliumAtom():
    He_atom = GSF.Atom(pos=np.array([0,0,0],Z=2.0))
    psi_laplacian = [] 
    psi_array = np.array([He_atom.psi_1s])
    ion_positions = np.array([He_atom.i_pos])
    ion_charges = np.array([He_atom.Z])
    N_e = 2
    
    wf = WaveFunctionClass()
    wf.setUpWavefunctions(psi_array)
    wf.setDownWavefunctions(psi_array)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setIonPositions(ion_positions)
    wf.setIonCharges(ion_charges)
    wf.setNumElectrons(N_e)              
    # set 1 up and 1 down for electrons
    wf.N_up = 1
    wf.N_down = 1
    return wf

def LithiumAtom():
    Li_atom = GSF.Atom(pos=np.array([0,0,0]),Z=3.0)
    psi_laplacian = []
    psi_array_up = np.array([ Li_atom.psi_1s])
    psi_array_down = np.array([Li_atom.psi_1s])
    ion_positions = np.array([Li_atom.i_pos])
    ion_charges = np.array([Li_atom.Z])
    N_e = 2

    wf = WaveFunctionClass()
    wf.setUpWavefunctions(psi_array_up)
    wf.setDownWavefunctions(psi_array_down)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setAtomList([Li_atom]) 
    #wf.setIonPositions(ion_positions)
    #wf.setIonCharges(ion_charges)
    #wf.setNumElectrons(N_e)
    # set 1 up and 1 down for electrons
    wf.setNumUp(len(psi_array_up))
    wf.setNumDown(len(psi_array_down))
    return wf

def H2Molecule(ion_sep):
    # ion_sep is in atomic units of Bohr radius 
    ion_positions = np.array([
        [-0.5*ion_sep, 0.0, 0.0],
        [0.5*ion_sep, 0.0, 0.0]]) * GSF.a_B
    H_atom1 = GSF.Atom(pos=np.array(ion_positions[0]),Z=1.0)
    H_atom2 = GSF.Atom(pos=np.array(ion_positions[1]),Z=1.0)
    psi_laplacian = []
    # two options for 2 electrons --> 2(up and down):0 or 1:1  (up: down or up:up)
    # using 1:1 and up for both for now  
    psi_array_up = np.array([H_atom1.psi_1s])
    psi_array_down = np.array([H_atom2.psi_1s])
    
    wf = WaveFunctionClass()
    wf.setUpWavefunctions(psi_array_up)
    wf.setDownWavefunctions(psi_array_down)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setAtomList([H_atom1,H_atom2])
    #wf.setIonPositions(ion_positions)
    #wf.setIonCharges(ion_charges)
    wf.setNumUp(len(psi_array_up))
    wf.setNumDown(len(psi_array_down))
    
    #print 'Simulating H2Molecule'
    return wf

def H3Molecule(ion_sep):
    # ion_sep is in atomic units of Bohr radius 
    ion_positions = np.array([
        [-0.5*ion_sep, 0, 0], 
        [0.5*ion_sep, 0, 0], 
        [0,0.5*ion_sep, 0]]) * GSF.a_B
    H_atom1 = GSF.H_atom(pos=np.array(ion_positions[0]))#,Z=1.0)
    H_atom2 = GSF.H_atom(pos=np.array(ion_positions[1]))#,Z=1.0)
    H_atom3 = GSF.H_atom(pos=np.array(ion_positions[2]))#,Z=1.0)
    psi_laplacian = []
    # two options for 2 electrons --> 2(up and down):0 or 1:1  (up: down or up:up)
    # using 1:1 and up for both for now  
    psi_array_up = np.array([H_atom1.psi_1s,H_atom2.psi_1s])
    psi_array_down = np.array([H_atom3.psi_1s])

    wf = WaveFunctionClass()
    wf.setUpWavefunctions(psi_array_up)
    wf.setDownWavefunctions(psi_array_down)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setAtomList([H_atom1,H_atom2,H_atom3])
    #wf.setIonPositions(ion_positions)
    #wf.setIonCharges(ion_charges)
    wf.setNumUp(len(psi_array_up))
    wf.setNumDown(len(psi_array_down))
    
    #print 'Simulating H2Molecule'
    return wf

def He2Molecule(ion_sep):
    # ion_sep is in atomic units of Bohr radius 
    ion_positions = np.array([
        [-0.5*ion_sep, 0.0, 0.0],
        [0.5*ion_sep, 0.0, 0.0]]) * GSF.a_B
    He_atom1 = GSF.Atom(pos=np.array(ion_positions[0]),Z=2.0)
    He_atom2 = GSF.Atom(pos=np.array(ion_positions[1]),Z=2.0)
    psi_laplacian = []
    # two options for 2 electrons --> 2(up and down):0 or 1:1  (up: down or up:up)
    # using 1:1 and up for both for now  
    psi_array_up = np.array([He_atom1.psi_1s,He_atom2.psi_1s])
    psi_array_down = np.array([He_atom1.psi_1s,He_atom2.psi_1s])
        
    wf = WaveFunctionClass()
    wf.setUpWavefunctions(psi_array_up)
    wf.setDownWavefunctions(psi_array_down)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setAtomList([He_atom1,He_atom2])
    #wf.setIonPositions(ion_positions)
    #wf.setIonCharges(ion_charges)
    wf.setNumUp(len(psi_array_up))
    wf.setNumDown(len(psi_array_down))
        
    #print 'Simulating H2Molecule'
    return wf

def Li2Molecule(ion_sep):
    # ion_sep is in atomic units of Bohr radius 
    ion_positions = np.array([
        [-0.5*ion_sep, 0.0, 0.0],
        [0.5*ion_sep, 0.0, 0.0]]) * GSF.a_B
    Li_atom1 = GSF.Atom(pos=np.array(ion_positions[0]),Z=3.0)
    Li_atom2 = GSF.Atom(pos=np.array(ion_positions[1]),Z=3.0)
    psi_laplacian = []
    # two options for 2 electrons --> 2(up and down):0 or 1:1  (up: down or up:up)
    # using 1:1 and up for both for now  
    psi_array_up = np.array([Li_atom1.psi_1s,Li_atom2.psi_1s,Li_atom1.psi_2s])
    psi_array_down = np.array([Li_atom1.psi_1s,Li_atom2.psi_1s,Li_atom2.psi_2s])
    
    wf = WaveFunctionClass()
    wf.setUpWavefunctions(psi_array_up)
    wf.setDownWavefunctions(psi_array_down)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setAtomList([Li_atom1,Li_atom2])
    #wf.setIonPositions(ion_positions)
    #wf.setIonCharges(ion_charges)
    wf.setNumUp(len(psi_array_up))
    wf.setNumDown(len(psi_array_down))
    
    #print 'Simulating H2Molecule'
    return wf

def HFMolecule(ion_sep):
    # ion_sep is in atomic units of Bohr radius 
    ion_positions = np.array([
        [-0.5*ion_sep, 0.0, 0.0],
        [0.5*ion_sep, 0.0, 0.0]]) * GSF.a_B
    H_atom = GSF.Atom(pos=np.array(ion_positions[0]),Z=1.0)
    F_atom = GSF.Atom(pos=np.array(ion_positions[1]),Z=9.0)
    psi_laplacian = []
    # two options for 2 electrons --> 2(up and down):0 or 1:1  (up: down or up:up)
    # using 1:1 and up for both for now  
    psi_array_up = np.array([H_atom.psi_1s, F_atom.psi_1s,  F_atom.psi_2s, F_atom.psi_2py, F_atom.psi_2pz])
    psi_array_down = np.array([F_atom.psi_1s,  F_atom.psi_2s, F_atom.psi_2px, F_atom.psi_2py, F_atom.psi_2pz])
    
    wf = WaveFunctionClass()
    wf.setUpWavefunctions(psi_array_up)
    wf.setDownWavefunctions(psi_array_down)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setAtomList([H_atom,F_atom])
    #wf.setIonPositions(ion_positions)
    #wf.setIonCharges(ion_charges)
    wf.setNumUp(len(psi_array_up))
    wf.setNumDown(len(psi_array_down))
    
    #print 'Simulating H2Molecule'
    return wf

def H2OMolecule(bond_length,bond_angle=np.pi*2/3):
    # bond_length is in atomic units of Bohr radius
    xdisp = np.cos((np.pi-bond_angle)*0.5)*bond_length * GSF.a_B
    ydisp = np.sin((np.pi-bond_angle)*0.5)*bond_length * GSF.a_B
    
    O_atom = GSF.Atom(pos=np.array([0,0,0]),Z=8.0)
    H_atom1 = GSF.Atom(pos=np.array([-xdisp, ydisp, 0]), Z=1.0)
    H_atom2 = GSF.Atom(pos=np.array([xdisp, ydisp, 0]), Z=1.0)
    
    # for each electron, need to have wavefn, atom position and atomic number
    psi_up = np.array([H_atom1.psi_1s, O_atom.psi_1s, O_atom.psi_2s, O_atom.psi_2py, O_atom.psi_2pz])
    psi_down = np.array([H_atom2.psi_1s, O_atom.psi_1s, O_atom.psi_2s, O_atom.psi_2py, O_atom.psi_2pz])
    #psi_laplacian = np.array([GSF.Lpsi_1s, GSF.Lpsi_1s])
    psi_laplacian = []
    
    ion_positions = np.array([H_atom1.i_pos, H_atom2.i_pos, O_atom.i_pos])
    ion_charges = np.array([H_atom1.Z, H_atom2.Z, O_atom.Z])
    N_e = 10

    wf = WaveFunctionClass()
    wf.setUpWavefunctions(psi_up)
    wf.setDownWavefunctions(psi_down)
    wf.setAtomicLaplacians(psi_laplacian)
    wf.setAtomList([H_atom1, H_atom2, O_atom])
    #wf.setIonPositions(ion_positions)
    #wf.setIonCharges(ion_charges)
    wf.setNumUp(len(psi_up))
    wf.setNumDown(len(psi_down))
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
    # An atomic orbital is assigned to each electron.

    # Define the atomic wavefunctions
    #psi_array = [] # GSF.getH2Functions()  #generate array of objective basis states
    #psi_laplacian = [] # GSF.getH2Laplacians() # get kinetic energy terms of wavefunctions (including hbar^2/2m)
    ion_positions = [] # GSF.ion_positions
    ion_charges = [] # GSF.ion_charges  
    N_ion = len(ion_positions)
    # TODO remove N_e = 1
    N_up = 0
    N_down = 0
    e_positions = np.zeros((1,3)) # use single electron list for now
    #e_pos_up = np.zeros((1,3)) # maybe the up list will be useful later
    #e_pos_down = np.zeros((1,3)) # maybe the down list will be useful later
    # There is a list of orbitals for the up electrons and another for down
    atom_list = []
    psi_up = []
    psi_down = []
    # Jastrow parameters
    Aee_same = 0.25 # parallel cusp condition, Drummonds et al
    Aee_anti = 0.5 # anti-parallel cusp condition, Drummonds et al
    Bee_same = 1.0 # ?
    Bee_anti = 1.0 # ?
    J = 0
    Cen = 0 # -1*ion_charges # Nucleus cusp condition, Drummonds et al
    Den = 10.0
    # Slater Matrix and determinant
    slater_matrix_up = []
    slater_det_up = 1.0
    inverse_SD_up = []                      
    slater_matrix_down = []
    slater_det_down = 1.0
    h=0.001

    def setAtomList(self, atoms):
        self.atom_list = atoms
        self.ion_positions = np.zeros((len(atoms),3))
        self.ion_charges = np.zeros(len(atoms))
        for i in range(len(atoms)):
            self.ion_positions[i] = atoms[i].i_pos
            self.ion_charges[i] = atoms[i].Z
        self.Cen = -1 * self.ion_charges

    def setUpWavefunctions(self, wfnArray):
        self.psi_up = wfnArray
        self.N_up = len(wfnArray)

    def setDownWavefunctions(self, wfnArray):
        self.psi_down = wfnArray
        self.N_down = len(wfnArray)

    def setNumUp(self, num): # should not be necessary
        self.N_up = num 

    def setNumDown(self, num): # should not be necessary
        self.N_down = num

    def setAtomicLaplacians(self, lapArray): 
        self.psi_laplacian = lapArray

    def setIonPositions(self, pos): 
        self.ion_positions = pos
        self.N_ion = len(pos)
    
    def setIonCharges(self, charges): 
        self.ion_charges = charges
        self.Cen = -1*charges
   
    def psiDiff(self, fns, rvec):
        out_list = np.zeros((2,len(fns))) # 0-old, 1-new ; cols are wfns
        for i in range(len(fns)):
            out_list[:,i] = fns[i](rvec)
        return out_list[1]-out_list[0]
     
    def InitializeElectrons(self):
        if self.N_up == 0 and self.N_down == 0:
            print 'Error: no electrons to initialize'
            return []
        else:
            # generate array of electron positions, normally distributed from the origin with Bohr radius
            n = self.N_up+self.N_down
            self.e_positions = np.random.randn(n,3) * GSF.a_B # generate array of electron positions
            print 'init e_pos',self.e_positions
            # Store displacements and distances
            self.e_disp = np.zeros((n,n,3)) # store the displacements in a 3D matrix to make indexing easier
            self.e_dist = np.zeros((n,n)) # the electron matrices should only be upper diagonal
            self.atom_disp = np.zeros((n,len(self.atom_list),3))
            self.atom_dist = np.zeros((n,len(self.atom_list)))
            index = 0
            for i in range(n):
                self.e_disp[i,i+1:] = self.e_positions[i] - self.e_positions[i+1:]
                self.e_dist[i,i+1:] = np.sqrt(np.sum(self.e_disp[i,i+1:]**2,1))
                self.atom_disp[i] = self.e_positions[i] - self.ion_positions
                self.atom_dist[i,:] = np.sqrt(np.sum(self.atom_disp[i,:]**2,1))
 	      #Once the e_position is initialize, the slater matrix and its deteriminant and inverse are all initialized. 
        self.slater_matrix_up = SlaterMatrix(self.e_positions[0:self.N_up],self.psi_up)
        self.slater_matrix_down = SlaterMatrix(self.e_positions[self.N_up:],self.psi_down)
        print 'slater_matrix',self.slater_matrix_up
        if self.N_up>0: 
            self.inverse_SD_up = LA.inv(self.slater_matrix_up)
            self.slater_det_up = LA.det(self.slater_matrix_up)			
        if self.N_down>0: 
            self.inverse_SD_down = LA.inv(self.slater_matrix_down) 
            self.slater_det_down = LA.det(self.slater_matrix_down)
        self.J = self.Jastrow()
        print 'slater_inv',self.inverse_SD_up
        return self.e_positions

    #def setNup(self, num):
    #    self.N_up = num

    def UpdatePosition(self, i, dr): 
        # function to update position of one electron and the corresponding distances
       	rnew = self.e_positions[i] + dr
        
        if testbool==False:
            # update the inverse of determinant matrix
            u = np.zeros(self.N_up+self.N_down)
            u[i]=1.0    # u = [0...1...0] ith electron
            if i < self.N_up:    # if the electron i to be updated is spin up 
                u = np.zeros(self.N_up)
		u[i]=1.0
		v = self.psiDiff(self.psi_up, np.array([self.e_positions[i], rnew]))  # v^T for rank one update method, it is simply the different of psi(r_old) and psi(r_new)
                ratio = 1.0 + np.dot(v,np.dot(self.inverse_SD_up,u))
	              # A_inv_new = A_inv - (A_inv*u*v^T*A_inv)/ratio
                self.inverse_SD_up += -1*np.outer(np.dot(self.inverse_SD_up,u),np.dot(v,self.inverse_SD_up.T))/ratio
                #print i,'dr',dr,'v',v,'ratio',ratio
                self.slater_det_up *= ratio
                #print 'SlaterInverse',self.inverse_SD_up, '          ratio',ratio,'cond',np.linalg.cond(self.inverse_SD_up)
            else: # if electron i is spin down
                u = np.zeros(self.N_down)
                u[i-self.N_up]=1.0
                v = self.psiDiff(self.psi_down, np.array([self.e_positions[i], rnew]))
                ratio = 1.0 + np.dot(v,np.dot(self.inverse_SD_down,u))
                self.inverse_SD_down += -1*np.outer(np.dot(self.inverse_SD_down,u),np.dot(v,self.inverse_SD_down.T))/ratio
                self.slater_det_down *= ratio
	      
        ## TODO test code
        if testbool:
            if i < self.N_up:
                slater_mat = SlaterMatrix(self.e_positions[:self.N_up],self.psi_up)
            else:
                slater_mat = SlaterMatrix(self.e_positions[self.N_up:],self.psi_down)
            old_SD = SlaterDeterminant(slater_mat)

        ## end test code

        #TODO Ji_before = self.Jastrow_i(i)
        self.e_positions[i] = rnew
        self.e_disp[i,i+1:] = rnew - self.e_positions[i+1:]
        self.e_dist[i,i+1:] = np.sqrt(np.sum(self.e_disp[i,i+1:]*self.e_disp[i,i+1:],1))
        self.e_disp[:i,i] = self.e_positions[:i] - rnew # displacements of earlier electrons
        self.e_dist[:i,i] = np.sqrt(np.sum(self.e_disp[:i,i]*self.e_disp[:i,i],1)) # distances of earlier electrons
        self.atom_disp[i,:] = rnew - self.ion_positions
        self.atom_dist[i,:] = np.sqrt(np.sum(self.atom_disp[i,:]*self.atom_disp[i,:],1))
        
        ## TODO test code
        if testbool:
            if i < self.N_up:
                slater_mat = SlaterMatrix(self.e_positions[:self.N_up],self.psi_up)
                new_SD = SlaterDeterminant(slater_mat)
                ratio = new_SD / old_SD
                self.slater_det_up = new_SD
            else:
                slater_mat = SlaterMatrix(self.e_positions[self.N_up:],self.psi_down)
                new_SD = SlaterDeterminant(slater_mat)
                ratio = new_SD / old_SD
                self.slater_det_down = new_SD
        #print 'ratio',ratio, 'SD',new_SD
        ## end test code
        
        # DEBUG test - PASSED
        #test_disps = np.zeros((len(self.e_positions),len(self.e_positions),3))
        #test_distances = np.zeros((len(self.e_positions),len(self.e_positions)))
        #for i in range(len(self.e_positions)):
        #    test_disps[i,i+1:] = self.e_positions[i] - self.e_positions[i+1:]
        #    test_disps[:i,i] = self.e_positions[:i] - self.e_positions[i]
        #    test_distances[i] = np.sqrt(np.sum(test_disps[i]*test_disps[i],1))
        #print 'distance check:',test_distances
            
        #TODO Ji_after = self.Jastrow_i(i)
        #deltaJ = Ji_after - Ji_before
        #self.J += deltaJ
        oldJ = self.J.copy()
        newJ = self.Jastrow()
        deltaJ = newJ-oldJ
        #print 'Jastrows match', deltaJ
        self.J = newJ.copy()
        return ratio * np.exp(-deltaJ)

    # MANY-BODY WAVEFUNCTION
    def PsiManyBody(self):
        """
        N_up = self.N_up
        if N_up > 0:
            slater_matrix_up = SlaterMatrix(self.e_positions[0:N_up], self.psi_up)
            slater_det_up = SlaterDeterminant(slater_matrix_up)
        else:
            slater_det_up = 1
        if N_down > 0:
            slater_matrix_down = SlaterMatrix(self.e_positions[N_up:], self.psi_down)
            slater_det_down = SlaterDeterminant(slater_matrix_down)
        else:
            slater_det_down = 1
        """
        self.J = self.Jastrow()
        return self.slater_det_up * self.slater_det_down * np.exp(-self.J)

    def QuickPsi(self, i, dr):
        rnew = self.e_positions[i] + dr
        # This should return an approximate value for psi,
        # where ONLY electron i is moved a SMALL amount dr
        # This meets finite difference halfway with partial analysis to save time
        # TODO write the function
        #u = np.zeros(self.N_up+self.N_down)                                                                                                 
        #u[i]=1.0    # u = [0...1...0] ith electron
        if i < self.N_up:    # if the electron i to be updated is spin up 
            u = np.zeros(self.N_up)
	    u[i]=1.0
	    v = self.psiDiff(self.psi_up, np.array([self.e_positions[i], rnew]))  # v^T for rank one update method, it is simply the different of psi(r_
            ratio = 1.0 + np.dot(v,np.dot(self.inverse_SD_up,u))
        else: # if electron i is spin down
            u = np.zeros(self.N_down)
	    u[i-self.N_up]=1.0
	    v = self.psiDiff(self.psi_down, np.array([self.e_positions[i], rnew]))
            ratio = 1.0 + np.dot(v,np.dot(self.inverse_SD_down,u))
        #print i,'psiratio',ratio,'   v',v
        return ratio * np.exp(-self.JastrowDiff(i,dr))

    def Jastrow(self):
        Uen = 0
        Uee = 0
        N_up = self.N_up
        
        Uen = np.sum(self.Cen * self.atom_dist / (1+self.Den*self.atom_dist))
        Uee_up_same = np.sum(self.Aee_same * self.e_dist[:N_up,:N_up] / (1+self.Bee_same * self.e_dist[:N_up,:N_up]))
        Uee_down_same = np.sum(self.Aee_same * self.e_dist[N_up:,N_up:] / (1+self.Bee_same * self.e_dist[N_up:,N_up:]))
        Uee_anti = np.sum(self.Aee_anti * self.e_dist[:N_up,N_up:] / (1+self.Bee_anti * self.e_dist[:N_up,N_up:]))
        Uee = Uee_up_same + Uee_down_same + Uee_anti
        """
        for i in range(len(e_positions)):
            # Compute ion distances from electron i
            # update electron-ion energy term
            
            en_sum = np.sum(self.Cen*self.atom_dist[i]/(1+self.Den*self.atom_dist[i]))
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
        """
        return -(Uee+Uen) #np.exp(-(Uee + Uen))

    def Jastrow_i(self,i):
        # compute Jastrow terms with electron i
        # update electron-ion energy term
        N_up = self.N_up
        Uen = np.sum(self.Cen*self.atom_dist[i]/(1+self.Den*self.atom_dist[i]))
        # Compute electron distances from electron i (only further in the list - count each p    air once)
        Uee = 0
        if i < N_up: # if this electron is spin up
            e_same = np.hstack((self.e_dist[:i,i], self.e_dist[i,i+1:N_up])) # electrons [i+1:N_up]
            e_anti = self.e_dist[i,N_up:]
            Uee += np.sum(self.Aee_same*e_same/(1+self.Bee_same*e_same))
            Uee += np.sum(self.Aee_anti*e_anti/(1+self.Bee_anti*e_anti))
        else: # if this electron is spin down
            # all the distances are to other down electrons
            Uee += np.sum(self.Aee_same*self.e_dist[i,i+1:]/(1+self.Bee_same*self.e_dist[i,i+1:]))

        return Uen + Uee

    def JastrowDiff(self,i,dr):
        # approximates the difference in Jastrow factor for incrementing electron i by vector dr
        da = np.sum(self.atom_disp[i]*dr,1)/self.atom_dist[i]
        a_new = self.atom_dist[i] + da
        Uen = np.sum(self.Cen*a_new/(1+self.Den*a_new))
        # Compute electron distances from electron i (only further in the list - count each p        air once)
        Uee = 0
        N_up = self.N_up
        if i < N_up: # if this electron is spin up
            e_same1 = self.e_dist[:i,i] # for earler electrons
            e_same2 = self.e_dist[i,i+1:N_up] # for later electrons [i+1:N_up]
            e_anti = self.e_dist[i,N_up:]
            # update: r_new = r_old + (disp_old * dr_vec)/r_old
            e_same_new1 = e_same1 + np.sum(self.e_disp[:i,i]*dr,1)/e_same1
            e_same_new2 = e_same2 + np.sum(self.e_disp[i,i+1:N_up]*dr,1)/e_same2
            e_anti_new = e_anti + np.sum(self.e_disp[i,N_up:]*dr,1)/e_anti

            Uee += np.sum(self.Aee_same*e_same_new1/(1+self.Bee_same*e_same_new1))
            Uee += np.sum(self.Aee_same*e_same_new2/(1+self.Bee_same*e_same_new2))
            Uee += np.sum(self.Aee_anti*e_anti_new/(1+self.Bee_anti*e_anti_new))
        else: # if this electron is spin down
            # all the distances are to other down electrons
            Uee += np.sum(self.Aee_same*self.e_dist[i,i+1:]/(1+self.Bee_same*self.e_dist[i,i+1:]))
        Jdiff = Uee + Uen - self.Jastrow_i(i)
        return Jdiff


    ##########################################
    # LOCAL ENERGY
    def LocalEnergy(self, psi_at_rvec):
        # KINETIC TERM
        # We can compute all of the kinetic energy terms given the positions
        # This might be hard to debug...
        # Apparently LA.det will compute determinants of all matrices stacked along dimension 2 at once
        # I am not sure this is any faster... but less for loops :)
        
        useHopper = True
        KE = 0
        N = len(self.e_positions) # 
        
        if useHopper: # THIS WILL NOT WORK AT ALL until laplacian matrix and lap Jastrow are also updated
            KE = 0 # need an indented line
            FDKineticEnergy = 0.0
            e_plusx = self.h * np.array([1,0,0])
            e_plusy = self.h * np.array([0,1,0])
            e_plusz = self.h * np.array([0,0,1])
            e_minusx = -1.0*self.h * np.array([1,0,0])
            e_minusy = -1.0*self.h * np.array([0,1,0])
            e_minusz = -1.0*self.h * np.array([0,0,1])
            for i in range(0,N):
                psi_sum = 0.0
                ratio = self.UpdatePosition(i,e_plusx)
                psi_sum += ratio
                #print 'ratio(FDhop)',ratio
                ratio *= self.UpdatePosition(i,e_plusy-e_plusx)
                psi_sum += ratio
                #print 'ratio(FDhop)',ratio
                ratio *= self.UpdatePosition(i,e_plusz-e_plusy)
                psi_sum += ratio
                #print 'ratio(FDhop)',ratio
                ratio *= self.UpdatePosition(i,e_minusx-e_plusz)
                psi_sum += ratio
                #print 'ratio(FDhop)',ratio
                ratio *= self.UpdatePosition(i,e_minusy-e_minusx)
                psi_sum += ratio
                #print 'ratio(FDhop)',ratio
                ratio *= self.UpdatePosition(i,e_minusz-e_minusy)
                psi_sum += ratio
                #print 'ratio(FDhop)',ratio
                ratio *= self.UpdatePosition(i,-e_minusz)
                #print 'ratio(FDhop)',ratio
                #print 'psi_sum',psi_sum
                FDKineticEnergy += KEprefactor * (-6.0 + psi_sum ) /(self.h*self.h)
            localKineticEnergy = FDKineticEnergy
            """
                ratio *= UpdatePosition(i,e_plusy-e_plusx)
                psi_sum += ratio
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
            """
        else:
            #Central Finite difference method to get laplacian
            #  e_posxPlusH = e_positions.copy()
            #  e_posyPlusH = e_positions.copy()
            #  e_poszPlusH = e_positions.copy()
            #  e_posxMinusH = e_positions.copy()
            #  e_posyMinusH = e_positions.copy()
            #  e_poszMinusH = e_positions.copy()
            psi = self.QuickPsi
            FDKineticEnergy = 0.0
            for i in range(0,N):
                #e_posxPlusH[i,0] += self.h
                #e_posyPlusH[i,1] += self.h
                #e_poszPlusH[i,2] += self.h
                #e_posxMinusH[i,0] += -1.0*self.h
                #e_posyMinusH[i,1] += -1.0*self.h
                #e_poszMinusH[i,2] += -1.0*self.h
                e_plusx = self.h * np.array([1,0,0])
                e_plusy = self.h * np.array([0,1,0])
                e_plusz = self.h * np.array([0,0,1])
                e_minusx = -1.0*self.h * np.array([1,0,0])
                e_minusy = -1.0*self.h * np.array([0,1,0])
                e_minusz = -1.0*self.h * np.array([0,0,1])
            
            # TODO This won't work until QuickPsi is working - maybe better to write DiffPsi for just differences instead
                #print 'psiratio e+dx',i,psi(i,e_plusx)
                psi_sum = psi(i,e_plusx) + psi(i,e_plusy) + psi(i,e_plusz) + psi(i,e_minusx) + psi(i,e_minusy) + psi(i,e_minusz)
                #print 'psisum',psi_sum 
                FDKineticEnergy += KEprefactor * (-6.0 + psi_sum ) /(self.h*self.h)
            localKineticEnergy = FDKineticEnergy

        
        # POTENTIAL TERM
        V_ion = 0
        V_e = 0
         
        V_ion = -np.sum(self.ion_charges * np.sum(1.0/self.atom_dist,axis=0)) * q_e2k
        V_e = np.sum(1.0/self.e_dist[self.e_dist!=0]) * q_e2k
        """
        for i in range(N):
            # electron-ion terms
            V_ion += -np.sum(self.ion_charges/self.atom_dist) * q_e2k
            
            # electron-electron terms
            e_displacements = e_positions[i] - e_positions[i+1:] # only calculate distances to e- not already counted
            e_distances = np.sqrt(np.sum(e_displacements*e_displacements,axis=1))
            V_e += np.sum(1.0/e_distances) * q_e2k                                                        
        """
        #print 'KE',localKineticEnergy
        return V_ion + V_e + localKineticEnergy



# SLATER DETERMINANT - is this still necessary?
def SlaterMatrix(e_positions,fn_array):
    # fn_array has the basis wavefunctions centered at the origin (shifted to the ion_position passed in as argument)
    slater_matrix = np.zeros((len(e_positions),(len(e_positions))))
    for j in range(0, len(fn_array)):
        slater_matrix[j,:] = fn_array[j](e_positions)  #build slater matrix
    return slater_matrix

def SlaterDeterminant(slater_matrix):
    Nfact = np.math.factorial(len(slater_matrix))
    SD = (1/np.sqrt(Nfact)) * LA.det(slater_matrix)
    return SD



