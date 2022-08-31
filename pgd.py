import numpy as np
import matplotlib.pyplot as plt

# Model definition
nDOF = 40
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
F[10] = 1
Jk = np.eye(nFreq)
Jm = -np.diag((2*np.pi*freq_array)**2)
Jf = np.ones(nFreq)

# Solve the FOM
U_FOM = np.zeros([nDOF, nFreq])*1j
for iFreq in range(nFreq):
    omega = 2*np.pi*freq_array[iFreq]
    U_FOM[:,iFreq] = np.linalg.solve(K - omega**2*M, F)
u_FOM = U_FOM[nDOF_check,:]

# Hyper parameters
nSamp = 1
nFP = 10
fp_tol = 1e-2
# Greedy search of basis
Basis = np.zeros([nDOF, nSamp])*1j
Rat_func = np.zeros([nFreq, nSamp])*1j
for iSamp in range(nSamp):
    Phi = np.zeros(nDOF)*1j
    Hfunc = np.ones(nFreq)
    Hfunc[0] = 1e3
    for iFP in range(nFP):
        print(iFP)
        # Solve for a new base
        Phi_old = Phi.copy()
        A = np.dot(Hfunc.conj(), np.matmul(Jk, Hfunc))*K + \
            np.dot(Hfunc.conj(), np.matmul(Jm, Hfunc))*M
        B = np.dot(Hfunc.conj(), Jf)*F
        Phi = np.linalg.solve(A, B)
        # Solve for freq response
        H_old = Hfunc.copy()
        A = np.dot(Phi.conj(), np.matmul(K, Phi))*Jk + \
            np.dot(Phi.conj(), np.matmul(M, Phi))*Jm
        B = np.dot(Phi.conj(), F)*Jf
        Hfunc = np.linalg.solve(A, B)
        diff_fp = np.linalg.norm(Phi - Phi_old)/np.linalg.norm(Phi) + \
                np.linalg.norm(Hfunc - H_old)/np.linalg.norm(Hfunc)
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
plt.show()