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
nSamp = 3
nFP = 50
fp_tol = 1e-3
# Greedy search of basis
Basis = np.zeros([nDOF, nSamp])*1j
Rat_func = np.zeros([nFreq, nSamp])*1j
for iSamp in range(nSamp):
    Phi = np.zeros(nDOF)*1j
    Hfunc = np.ones(nFreq, dtype=complex)
    Hfunc[0] = 1e3
    Hfaux = np.ones(nFreq, dtype=complex)
    for iFP in range(nFP):
        print(iFP)
        if iSamp == 0:
            # Solve for a new base
            Phi_old = Phi.copy()
            A = (Hfunc.conj() @ Jk @ Hfunc)*K + \
                (Hfunc.conj() @ Jm @ Hfunc)*M
            B = (Hfunc.conj() @ Jf)*F
            Phi = np.linalg.solve(A, B)
            Phi = Phi/np.linalg.norm(Phi)
            # Solve for freq response
            H_old = Hfunc.copy()
            A = (Phi.conj() @ K @ Phi)*Jk + \
                (Phi.conj() @ M @ Phi)*Jm
            B = (Phi.conj() @ F)*Jf
            Hfunc = np.linalg.solve(A, B)
        else:
            # Solve for a new base
            F_int = F.copy()*0
            for jSamp in range(iSamp):
                F_int = F_int + (Hfunc.conj() @ Jk @ np.diag(Rat_func[:,jSamp]) @ Hfaux)* \
                                (K @ Basis[:,jSamp]) + \
                                (Hfunc.conj() @ Jm @ np.diag(Rat_func[:,jSamp]) @ Hfaux)* \
                                (M @ Basis[:,jSamp]) 
            Phi_old = Phi.copy()
            A = (Hfunc.conj() @ Jk @ Hfunc)*K + \
                (Hfunc.conj() @ Jm @ Hfunc)*M
            B = (Hfunc.conj() @ Jf)*F - F_int
            Phi = np.linalg.solve(A, B)
            Phi = Phi/np.linalg.norm(Phi)
            # Solve for freq response
            H_old = Hfunc.copy()
            for iFreq in range(nFreq):
                D_red = np.zeros([2,2])*1j
                F_red = np.zeros(2)*1j
                D_red[0,0] = (Phi.conj() @ K @ Phi) + jm[iFreq]* (Phi.conj() @ M @ Phi)
                F_red[0] = Phi.conj() @ F
                for jSamp in range(iSamp):
                    for kSamp in range(iSamp):
                        D_red[1,1] = D_red[1,1] + (Basis[:,jSamp].conj() @ K @ Basis[:,kSamp])* Rat_func[iFreq,jSamp].conj()*Rat_func[iFreq,kSamp] + \
                                                  jm[iFreq]* (Basis[:,jSamp].conj() @ M @ Basis[:,kSamp])* Rat_func[iFreq,jSamp].conj()*Rat_func[iFreq,kSamp]
                    D_red[0,1] = D_red[0,1] + (Phi.conj() @ K @ Basis[:,jSamp])* Rat_func[iFreq,jSamp] + \
                                              jm[iFreq]* (Phi.conj() @ M @ Basis[:,jSamp])* Rat_func[iFreq,jSamp]
                    D_red[1,0] = D_red[1,0] + (Basis[:,jSamp].conj() @ K @ Phi)* Rat_func[iFreq,jSamp].conj() + \
                                              jm[iFreq]* (Basis[:,jSamp].conj() @ M @ Phi)* Rat_func[iFreq,jSamp].conj()
                    F_red[1] = F_red[1] + (Basis[:,jSamp].conj() @ F)* Rat_func[iFreq,jSamp].conj()
                U_red = np.linalg.solve(D_red, F_red)
                Hfunc[iFreq] = U_red[0]
                Hfaux[iFreq] = U_red[1]
        diff_fp = np.linalg.norm(Phi - Phi_old)/np.linalg.norm(Phi) + \
                  np.linalg.norm(Hfunc - H_old)/np.linalg.norm(Hfunc)
        if diff_fp < fp_tol:
            print("#########################################################")
            print(np.linalg.norm(Phi))
            print(np.linalg.norm(Hfunc))
            break
    Basis[:,iSamp] = Phi
    Rat_func[:,0:iSamp] = np.diag(Hfaux) @ Rat_func[:,0:iSamp]
    Rat_func[:,iSamp] = Hfunc
    

U_ROM = Basis @ Rat_func.T
u_ROM = U_ROM[nDOF_check,:]
# Plot FRF
fig = plt.figure(1)
plt.semilogy(freq_array, np.abs(u_FOM))
plt.semilogy(freq_array, np.abs(u_ROM))
plt.semilogy(freq_array, np.abs(Rat_func[:,0]),'*')
plt.semilogy(freq_array, np.abs(Rat_func[:,1]),'*')
plt.semilogy(freq_array, np.abs(Rat_func[:,2]),'--')
plt.show()