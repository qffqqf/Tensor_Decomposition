import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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


# Solve the FOM
U_FOM = np.zeros([nDOF, nFreq])*1j
for iFreq in range(nFreq):
    omega = 2*np.pi*freq_array[iFreq]
    U_FOM[:,iFreq] = np.linalg.solve(K - omega**2*M, F)
u_FOM = U_FOM[nDOF_check,:]

# Hyper parameters
nSamp = 3
nFP = 50
fp_tol = 1e-7
# Greedy search of basis
Basis = np.zeros([nDOF, nSamp])*1j
Fintk = np.zeros([nDOF, nSamp])*1j
Fintm = np.zeros([nDOF, nSamp])*1j
Ratpole = np.zeros(nFreq)*1j
Ratscal = np.zeros(nFreq)*1j
Rat_func = np.zeros([nFreq, nSamp])*1j
for iSamp in range(nSamp):
    print('iSamp = ', iSamp)
    Phi = np.zeros(nDOF, dtype=complex)
    scalar = 1
    shift = 1e-5*(1+1j)
    for iFP in range(nFP):
        print('iFP = ', iFP)
        # Solve for a new base
        F_int = F.copy()*0
        for jSamp in range(iSamp):
            F_int = F_int + Ratscal[jSamp]/(np.conjugate(shift)**2 - Ratpole[jSamp]**2)* \
                            Fintk[:,jSamp] - \
                            Ratscal[jSamp]*np.conjugate(shift)**2/(np.conjugate(shift)**2 - Ratpole[jSamp]**2)* \
                            Fintm[:,jSamp]
        shift_old = deepcopy(shift)
        A = K - np.conjugate(shift)**2*M
        B = (np.conjugate(shift)**2 - shift**2)/scalar*(F - F_int)
        Phi = np.linalg.solve(A, B)
        Phi = Phi/np.linalg.norm(Phi)        
        # Solve for freq response
        k_eig = (Phi.conj() @ K @ Phi)
        m_eig = (Phi.conj() @ M @ Phi)
        f_eig = (Phi.conj() @ F)
        shift = np.sqrt(k_eig/m_eig)
        scalar = -f_eig/m_eig
        diff_fp = np.linalg.norm(shift - shift_old)/np.linalg.norm(shift)
        print('diff = ', diff_fp)
        print('shift = ', shift)
        print('scalar = ', scalar)
        if diff_fp < fp_tol:
            print("#########################################################")
            break
    Eintk = Phi.conj() @ Fintk
    Eintm = Phi.conj() @ Fintm
    Hfunc = np.ones(nFreq, dtype=complex)
    for iFreq in range(nFreq):
        omega = 2*np.pi*freq_array[iFreq]
        G_int = Eintk @ Rat_func[iFreq,:].T - omega**2*Eintm @ Rat_func[iFreq,:].T
        Hfunc[iFreq] = (f_eig - G_int)/(k_eig - omega**2*m_eig)
    Basis[:,iSamp] = Phi
    Fintk[:,iSamp] = K @ Phi
    Fintm[:,iSamp] = M @ Phi
    Ratpole[iSamp] = shift
    Ratscal[iSamp] = scalar
    Rat_func[:,iSamp] = Hfunc
    if iSamp > 0:
        for iter in range(5):
            for jSamp in range(iSamp+1):
                Eintk = Basis[:,jSamp].conj() @ Fintk
                Eintm = Basis[:,jSamp].conj() @ Fintm
                f_eig = (Basis[:,jSamp].conj() @ F)
                index = list(range(0,jSamp)) + list(range(jSamp+1,iSamp+1))
                print('jSamp = ', jSamp)
                print('index = ', index)
                for iFreq in range(nFreq):
                    omega = 2*np.pi*freq_array[iFreq]
                    G_int = Eintk[index] @ Rat_func[iFreq,index].T - omega**2*Eintm[index] @ Rat_func[iFreq,index].T
                    k_eig = Eintk[jSamp]
                    m_eig = Eintm[jSamp]
                    Hfunc[iFreq] = (f_eig - G_int)/(k_eig - omega**2*m_eig)
                Rat_func[:,jSamp] = Hfunc
    '''  
    for iter in range(5):
        for jSamp in range(iSamp+1):
            Eintkk = Fintk[:,jSamp].conj() @ Fintk
            Eintkm = Fintk[:,jSamp].conj() @ Fintm
            Eintmk = Fintm[:,jSamp].conj() @ Fintk
            Eintmm = Fintm[:,jSamp].conj() @ Fintm
            Ekf = Fintk[:,jSamp].conj() @ F
            Emf = Fintm[:,jSamp].conj() @ F
            index = list(range(0,jSamp)) + list(range(jSamp+1,iSamp+1))
            print('jSamp = ', jSamp)
            print('index = ', index)
            for iFreq in range(nFreq):
                omega = 2*np.pi*freq_array[iFreq]
                G_int = Eintkk[index] @ Rat_func[iFreq,index].T - omega**2*(Eintkm[index] + Eintmk[index]) @ Rat_func[iFreq,index].T \
                      + omega**4*Eintmm[index] @ Rat_func[iFreq,index].T
                kk_eig = Eintkk[jSamp]
                km_eig = Eintkm[jSamp] + Eintmk[jSamp]
                mm_eig = Eintmm[jSamp]
                Hfunc[iFreq] = (Ekf - omega**2*Emf - G_int)/(kk_eig - omega**2*km_eig + omega**4*mm_eig)
            Rat_func[:,jSamp] = Hfunc
    '''      
    
    
U_ROM = Basis @ Rat_func.T
u_ROM = U_ROM[nDOF_check,:]
# Plot FRF
fig = plt.figure(1)
plt.semilogy(freq_array, np.abs(u_FOM))
plt.semilogy(freq_array, np.abs(u_ROM))
#plt.semilogy(freq_array, np.abs(Rat_func[:,0]), '*')
#plt.semilogy(freq_array, np.abs(Rat_func[:,1]), 'o')
#plt.semilogy(freq_array, np.abs(Rat_func[:,2]), '^')
fig = plt.figure(2)
plt.semilogy(freq_array, np.abs(u_ROM-u_FOM)/np.abs(u_FOM))
plt.show()