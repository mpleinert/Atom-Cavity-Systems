# GET PHOTON DISTRIBUTION
#
# Jaynes/Tavis-Cummings model for N=2 two-level atoms coupled to a cavity considering m modes
#
# one atom is fixed at an antinode of the cavity field implying g_1=g
# while the second atom is scanned along the cavity axis implying g_2=g*cos(2*Pi*Delta_z/lambda_C)=g*cos(phi_z)
#
# Maximum coupling constant: g
#
# Coherent pumping: eta * (S^+ + S^-)
#
# Possible dissipative processes:
#   - cavity decay; kappa
#   - thermal photons; n_th
#   - incoherent pumping; gamma_12
#   - spontaneous emission; gamma_21
#   - detuning from cavity; delta = w_c - w_L
#   - detuning from atom; Delta = = w_a - w_L
#   - dephasing; Gamma
#

import sys
import scipy as sp
import numpy as np
from qutip import *

# in this version only support fur N=2 due to phase limits

# --- parameters
N = int(sys.argv[1])                #number of atoms
m = int(sys.argv[2]) # >= 2 !       #maximum number of excitations(photons) considered

kappa = 1.0                         #cavity-decay constant
# -> everything will be normalized w.r.t. kappa

n_th = float(sys.argv[3])           #thermal photons

g_max = float(sys.argv[4])          #atom-cavity coupling

gamma_21 = float(sys.argv[5])       #rate of spontaneous emission
gamma_12 = float(sys.argv[6])       #rate of incoherent pumping

#w_L = 1.0                          #driving laser frequency

delta = float(sys.argv[7])          #detuning from cavity = w_c - w_L
#w_c = w_L + delta                  #cavity frequency

Delta = float(sys.argv[8])          #detuning from atom = w_a - w_L
#w_a = w_L + Delta                  #atom frequency

Gamma = float(sys.argv[9])          #decay rate due to dephasing

eta = float(sys.argv[10])           #amplitude/rate of coherent pumping

phiz = float(sys.argv[11])*np.pi    #phase between the two atoms due to different coupling


# --- cavity mode operator
# - annihilation operator
a = destroy(m)
for k in range(1,N+1):
    a = tensor(a,identity(2))


# --- atom operators
# - sigma_z for one atom (j needs to be smaller than N)
def sz(j):
    dummy = identity(m)
    for k in range(1,j):
        dummy = tensor(dummy,identity(2))
    dummy = tensor(dummy,sigmaz())
    for k in range(j+1,N+1):
        dummy = tensor(dummy,identity(2))
    return dummy

# - Sigma_z collective
Sz = 0
for k in range(1,N+1):
    Sz = Sz + sz(k)

# - sigma_minus for one atom (j needs to be smaller than N)
def sm(j):
    dummy = identity(m)
    for k in range(1,j):
        dummy = tensor(dummy,identity(2))
    dummy = tensor(dummy,destroy(2))
    for k in range(j+1,N+1):
        dummy = tensor(dummy,identity(2))
    return dummy

# - Sigma_minus collective
Sm = 0
for k in range(1,N+1):
    Sm = Sm + sm(k)

# - sigma_p * sigma_m collective
#SpSm = 0
#for k in range(1,N+1):
#    SpSm = SpSm + sm(k).dag()*sm(k)


# --- collapse operators / loss terms
c_ops = []
c_ops.append(np.sqrt((1+n_th) * kappa) * a)         # cavity decay
c_ops.append(np.sqrt(n_th * kappa) * a.dag())

for l in range(1,N+1):
    c_ops.append(np.sqrt(gamma_21) * sm(l))         # spontaneous emission
    c_ops.append(np.sqrt(Gamma) * sz(l))            # dephasing
    c_ops.append(np.sqrt(gamma_12) * sm(l).dag())   # incoherent pumping

# atom-cavity-coupling strength for each atom
g = np.zeros(N)
g[0] = g_max    # first atom maximally coupled
if (N > 1):
    g[1] = g_max * np.cos(phiz)     #second atom's coupling depends on relative phase


# --- Hamiltonian

H = Delta * Sz + delta * a.dag()*a + eta * (Sm.dag() + Sm)
# = H_{free atoms}  +  H_{cavity}  +  H_{coherent pumping}

# add H_{Jaynes/Tavis-Cummings interaction}
for l in range(len(g)):         # g: 0 -> N-1 whereas sm: 1 -> N !!
    H = H + g[l] * (sm(l+1).dag() * a + sm(l+1) * a.dag())

# --- Calculate steady-state solution
psi = steadystate(H, c_ops, method='eigen')     #method can be direct, eigen, power, ...

aa = expect(a.dag()*a, psi)             #first moment of cavity field = mean photon number
aaaa = expect(a.dag()*a.dag()*a*a, psi) #second moment (normally ordered) of cavity field
g2a = aaaa / (aa * aa)      # normalized Glauber's second-order intensity correlation function

# --- negative binomial distribution (nbd) - s and p parameter
s = (aa*aa) / (aaaa - aa*aa)
p = aa / (aaaa + aa - aa*aa)

save = np.array([g2a, aa, s, p])

# --- Saving system's distribution and related distributions
data = np.zeros((m,4))
# - system, coherent, thermal, nbd

for n in range(0,m):
    data[n] = [psi.ptrace(0)[n,n], (aa**n) * np.exp(-aa) / np.math.factorial(n), (aa / (aa + 1) )**n / (aa + 1), ( sp.special.gamma(s+n) / (np.math.factorial(n) * sp.special.gamma(s)) ) * (p**s) * ((1-p)**n)]

# --- Fidelity

fidelity = np.zeros(3)

for n in range(0,m):
    fidelity[0] = fidelity[0] + np.sqrt( psi.ptrace(0)[n,n] * (aa**n) * np.exp(-aa) / np.math.factorial(n) )
    fidelity[1] = fidelity[1] + np.sqrt( psi.ptrace(0)[n,n] * (aa / (aa + 1) )**n / (aa + 1) )
    fidelity[2] = fidelity[2] + np.sqrt( psi.ptrace(0)[n,n] * ( sp.special.gamma(s+n) / (np.math.factorial(n) * sp.special.gamma(s)) ) * (p**s) * ((1-p)**n) )


# --- Writing to file
np.savetxt("photon_distribution.dat", data, delimiter=', ', newline='\n')
np.savetxt("g2a-aa-n-p.dat", save, delimiter=' ', newline=' ', fmt='%.5f') #fmt for specifying the valid digits
np.savetxt("fidelity.dat", fidelity, delimiter=' ', newline=' ', fmt='%.4f')
