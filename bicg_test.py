import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicgstab

# Model definition
nDOF = 1000
nFreq = 100
k = 10000
m = 1
damping = 0.03
nDOF_check = 2
freq = 0.4
K = 2*k*np.diag(np.ones(nDOF)) - k*np.diag(np.ones(nDOF-1),-1) - k*np.diag(np.ones(nDOF-1),1)
M = m*np.eye(nDOF) 
K = K*(1+damping*1j)
D = K - (2*np.pi*freq)**2 *M
F = np.zeros(nDOF)
F[0] = 1
U_FOM = np.linalg.solve(D, F)

D = csc_matrix(D)
U_test, exit_code = bicgstab(D, F, tol=1e-15, maxiter=30000)
print(exit_code)
#print(U_test)
#print(U_FOM)
print(np.linalg.norm(U_test-U_FOM)/np.linalg.norm(U_FOM))