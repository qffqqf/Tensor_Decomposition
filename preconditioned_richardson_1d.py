import numpy as np
from scipy.linalg import lu
import matplotlib.pyplot as plt

# Model definition
nDOF = 40
nFreq = 100
k = 1000
m = 1
damping = 0.03
nDOF_check = 2
freq_array = np.linspace(0,0.4,nFreq)
K = 2*k*np.diag(np.ones(nDOF)) - k*np.diag(np.ones(nDOF-1),-1) - k*np.diag(np.ones(nDOF-1),1)
M = m*np.eye(nDOF) 
K = K*(1+damping*1j)
F = np.zeros([nDOF,1])
F[0] = 1

# Solve the FOM
U_FOM = np.zeros([nDOF, nFreq])*1j
for iFreq in range(nFreq):
    omega = 2*np.pi*freq_array[iFreq]
    U_FOM[:,iFreq] = np.linalg.solve(K - omega**2*M, F[:,0])
u_FOM = U_FOM[nDOF_check,:]

# Hyper parameters
alpha = np.mean(freq_array)*2*np.pi
nFP = 1000

# Richardson process
U_ROM = np.zeros([nDOF, nFreq])*1j
Resi0 = np.zeros([nDOF, nFreq])*1j
D_alpha = K - alpha**2*M
D2_alpha = np.matmul(D_alpha.conj().T, D_alpha)

for iFreq in range(nFreq):
    omega = 2*np.pi*freq_array[iFreq]
    D = K - omega**2*M
    Resi0[:,iFreq] = np.matmul(D.conj().T, F[:,0])
resi0 = np.linalg.norm(Resi0)

Resi = Resi0.copy()
for iFP in range(nFP):
    print(iFP)
    U_ROM = U_ROM + np.linalg.solve(D2_alpha, Resi)
    Phi, sigma, H_basis = np.linalg.svd(U_ROM, full_matrices=True)
    sigma = sigma[sigma>1e-3]
    Sigma = np.diag(sigma)
    nTrunc = len(Sigma[:,0])
    print(np.linalg.norm(np.matmul(np.matmul(Phi[:,0:nTrunc], Sigma), H_basis[0:nTrunc,:])-U_ROM))
    U_ROM = np.matmul(np.matmul(Phi[:,0:nTrunc], Sigma), H_basis[0:nTrunc,:])
    for iFreq in range(nFreq):
        omega = 2*np.pi*freq_array[iFreq]
        D = K - omega**2*M
        D2 = np.matmul(D.conj().T, D)
        Resi[:,iFreq] = Resi0[:,iFreq] - np.matmul(D2, U_ROM[:,iFreq])
    resi = np.linalg.norm(Resi)
    print(resi/resi0)




u_ROM = U_ROM[nDOF_check,:]
# Plot FRF
fig = plt.figure(1)
plt.semilogy(freq_array, np.abs(u_FOM))
plt.semilogy(freq_array, np.abs(u_ROM))

fig = plt.figure(2)
plt.semilogy(freq_array, np.abs(u_FOM-u_ROM)/np.abs(u_FOM))
plt.show()