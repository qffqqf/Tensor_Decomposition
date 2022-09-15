import numpy as np
import matplotlib.pyplot as plt

# Model definition
nDOF = 10
nFreq = 1000
k = 1000
m = 1
damping = 0.01
nDOF_check = 2
freq_array = np.linspace(0,5,nFreq)
K = 2*k*np.diag(np.ones(nDOF)) - k*np.diag(np.ones(nDOF-1),-1) - k*np.diag(np.ones(nDOF-1),1)
M = m*np.eye(nDOF)
K = K*(1+damping*1j)
F = np.zeros(nDOF)
F[5] = 1
Jk = np.eye(nFreq)
Jm = -np.diag((2*np.pi*freq_array)**2)
Jf = np.ones(nFreq)
jm = -(2*np.pi*freq_array)**2


# Solve the FOM
U_FOM = np.zeros([nDOF, nFreq])*1j
for iFreq in range(nFreq):
    omega = 2*np.pi*freq_array[iFreq]
    U_FOM[:,iFreq] = np.linalg.solve(K - omega**2*M, F)
u_FOM = U_FOM[nDOF_check,:]

# Hyper parameters
nSamp = 1
nFP = 3
fp_tol = 1e-2
# Greedy search of basis
Basis = np.zeros([nDOF, nSamp])*1j
Rat_func = np.zeros([nFreq, nSamp])*1j
for iSamp in range(nSamp):
    Phi = np.zeros(nDOF, dtype=complex)
    Hfunc = np.ones(nFreq, dtype=complex)
    Hfunc[0] = 1e3
    Phaux = np.zeros(nFreq, dtype=complex)
    Hfaux = Hfunc.copy()
    for iFP in range(nFP):
        print(iFP)
        # Solve for a new base
        F_int = F.copy()*0
        for jSamp in range(iSamp):
            F_int = F_int + (Hfaux.conj() @ Jk @ Rat_func[:,jSamp])* \
                            (K @ Basis[:,jSamp]) + \
                            (Hfaux.conj() @ Jm @ Rat_func[:,jSamp])* \
                            (M @ Basis[:,jSamp]) 
        Phi_old = Phi.copy()
        A = (Hfaux.conj() @ Jk @ Hfunc)*K + \
            (Hfaux.conj() @ Jm @ Hfunc)*M
        B = (Hfaux.conj() @ Jf)*F - F_int
        Phi = np.linalg.solve(A, B)
        Phi = Phi/np.linalg.norm(Phi)
        #print(Phi)
        # Solve for an auxilary new base
        A = (Hfunc.conj() @ Jk.conj().T @ Hfaux)*K.conj().T + \
            (Hfunc.conj() @ Jm.conj().T @ Hfaux)*M.conj().T
        B = (Hfunc.conj() @ Jf)*F
        Phaux = np.linalg.solve(A, B)
        Phaux = Phaux/np.linalg.norm(Phaux)
        #print(Phaux)
        # Solve for freq response
        G_int = Jf.copy()*0
        for jSamp in range(iSamp):
            G_int = G_int + (Phaux.conj() @ K @ Basis[:,jSamp])* \
                            (Jk @ Rat_func[:,jSamp]) + \
                            (Phaux.conj() @ M @ Basis[:,jSamp])* \
                            (Jm @ Rat_func[:,jSamp]) 
        H_old = Hfunc.copy()
        A = (Phaux.conj() @ K @ Phi)*Jk + \
            (Phaux.conj() @ M @ Phi)*Jm
        B = (Phaux.conj() @ F)*Jf - G_int
        Hfunc = np.linalg.solve(A, B)
        #print(Hfunc)
        # Solve for an auxilary response
        A = (Phi.conj() @ K.conj().T @ Phaux)*Jk.conj().T + \
            (Phi.conj() @ M.conj().T @ Phaux)*Jm.conj().T
        B = (Phi.conj() @ F)*Jf
        Hfaux = np.linalg.solve(A, B)
        #print(Hfaux)
        diff_fp = np.linalg.norm(Phi - Phi_old)/np.linalg.norm(Phi) + \
                  np.linalg.norm(Hfunc - H_old)/np.linalg.norm(Hfunc)
        print(diff_fp)
        if diff_fp < fp_tol:
            print("#########################################################")
            break
    Basis[:,iSamp] = Phi
    Rat_func[:,iSamp] = Hfunc
    

U_ROM = Basis @ Rat_func.T
u_ROM = U_ROM[nDOF_check,:]
# Plot FRF
fig = plt.figure(1)
plt.semilogy(freq_array, np.abs(u_FOM))
plt.semilogy(freq_array, np.abs(u_ROM))
#plt.semilogy(freq_array, np.abs(Rat_func[:,1]),'*')
#plt.semilogy(freq_array, np.abs(Rat_func[:,2]),'--')
plt.show()