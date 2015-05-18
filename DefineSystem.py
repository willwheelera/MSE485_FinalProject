import numpy as np
import GenerateStartingFunctions as GSF


def HydrogenAtom(wf):
    H_atom = GSF.Atom(pos=np.array([0,0,0],Z=1.0))
    psi_array = np.array([H_atom.psi_1s])
    psi_laplacian = []
    ion_positions = np.array([H_atom.i_pos])
    ion_charges = np.array([H_atom.Z]) 
    N_e = 1
    
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

def HeliumAtom(wf):
    He_atom = GSF.Atom(pos=np.array([0,0,0],Z=2.0))
    psi_laplacian = [] 
    psi_array = np.array([He_atom.psi_1s])
    ion_positions = np.array([He_atom.i_pos])
    ion_charges = np.array([He_atom.Z])
    N_e = 2
    
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

def LithiumAtom(wf):
    Li_atom = GSF.Atom(pos=np.array([0,0,0]),Z=3.0)
    psi_laplacian = []
    psi_array_up = np.array([ Li_atom.psi_1s])
    psi_array_down = np.array([Li_atom.psi_1s])
    ion_positions = np.array([Li_atom.i_pos])
    ion_charges = np.array([Li_atom.Z])
    N_e = 2

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

def H2Molecule(wf,ion_sep):
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

def H3Molecule(wf,ion_sep):
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

def He2Molecule(wf,ion_sep):
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

def Li2Molecule(wf,ion_sep):
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

def HFMolecule(wf,ion_sep):
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

def H2OMolecule(wf,bond_length,bond_angle=np.pi*2/3):
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
