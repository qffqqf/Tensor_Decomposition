import numpy as np
import matplotlib.pyplot as plt

# Model definition
nDOF = 5
nFreq = 1000
k = 1000
m = 1
damping = 0.01
nDOF_check = 4
freq_array = np.linspace(-20,20,nFreq)
K = 2*k*np.diag(np.ones(nDOF)) - k*np.diag(np.ones(nDOF-1),-1) - k*np.diag(np.ones(nDOF-1),1)
M = m*np.eye(nDOF)
K = K*(1+damping*1j)
F = np.zeros(nDOF)
F[3] = 1
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
nFP = 50
fp_tol = 1e-2
# Greedy search of basis
Basis = np.zeros([nDOF, nSamp])*1j
Fintk = np.zeros([nDOF, nSamp])*1j
Fintm = np.zeros([nDOF, nSamp])*1j
Rat_func = np.zeros([nFreq, nSamp])*1j
for iSamp in range(nSamp):
    Phi = np.zeros(nDOF, dtype=complex)
    Hfunc = np.ones(nFreq, dtype=complex)
    Hfunc[0] = 1e3
    k_eig = 1
    m_eig = 1
    for iFP in range(nFP):
        print(iFP)
        # Solve for a new base
        F_int = F.copy()*0
        for jSamp in range(iSamp):
            F_int = F_int + (Hfunc.conj() @ Jk @ Rat_func[:,jSamp])* \
                            (K @ Basis[:,jSamp]) + \
                            (Hfunc.conj() @ Jm @ Rat_func[:,jSamp])* \
                            (M @ Basis[:,jSamp]) 
        Phi_old = Phi.copy()
        A = (Hfunc.conj() @ Jk @ Hfunc)*K + \
            (Hfunc.conj() @ Jm @ Hfunc)*M
        omega_eig2 = -(Hfunc.conj() @ Jm @ Hfunc)/(Hfunc.conj() @ Jk @ Hfunc)
        omega_eig2_ = k_eig/m_eig
        print('freq = ', np.sqrt(omega_eig2)/2/np.pi)
        print('omega_eig2 = ', omega_eig2)
        print('omega_eig2_ = ', omega_eig2_)
        A = K - omega_eig2_*M
        B = ((Hfunc.conj() @ Jf)*F - F_int)/(Hfunc.conj() @ Jk @ Hfunc)
        Phi = np.linalg.solve(A, B)
        Phi = Phi/np.linalg.norm(Phi)        
        # Solve for freq response
        H_old = Hfunc.copy()
        Eintk = Phi.conj() @ Fintk
        Eintm = Phi.conj() @ Fintm
        k_eig = (Phi.conj() @ K @ Phi)
        m_eig = (Phi.conj() @ M @ Phi)
        for iFreq in range(nFreq):
            omega = 2*np.pi*freq_array[iFreq]
            G_int = Eintk @ Rat_func[iFreq,:].T - omega**2*Eintm @ Rat_func[iFreq,:].T
            Hfunc[iFreq] = ((Phi.conj() @ F) - G_int)/(k_eig - omega**2*m_eig)
        diff_fp = np.linalg.norm(Phi - Phi_old)/np.linalg.norm(Phi) + \
                  np.linalg.norm(Hfunc - H_old)/np.linalg.norm(Hfunc)
        print(diff_fp)
        if diff_fp < fp_tol:
            print("#########################################################")
            break
    Basis[:,iSamp] = Phi
    Fintk[:,iSamp] = K @ Phi
    Fintm[:,iSamp] = M @ Phi
    Rat_func[:,iSamp] = Hfunc
    

U_ROM = Basis @ Rat_func.T
u_ROM = U_ROM[nDOF_check,:]
# Plot FRF
fig = plt.figure(1)
plt.semilogy(freq_array, np.abs(u_FOM))
plt.semilogy(freq_array, np.abs(u_ROM))
fig = plt.figure(2)
plt.semilogy(freq_array, np.abs(u_ROM-u_FOM)/np.abs(u_FOM))
plt.show()