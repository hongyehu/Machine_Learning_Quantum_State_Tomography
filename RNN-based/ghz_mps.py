import numpy as np
from ncon import ncon

class GHZ():
    def __init__(self, POVM='Tetra', MPS = 'GHZ', Number_qubits=4,Nmax=10000):
        self.N = Number_qubits;

        # POVMs and other operators
        # Pauli matrices
        self.I = np.array([[1, 0],[0, 1]]);
        self.X = np.array([[0, 1],[1, 0]]);    self.s1 = self.X;
        self.Z = np.array([[1, 0],[0, -1]]);   self.s3 = self.Z;
        self.Y = np.array([[0, -1j],[1j, 0]]); self.s2 = self.Y;
        self.Nmax = Nmax

        # Which POVM 
        if POVM=='4Pauli':
            self.K = 4;
            self.M = np.zeros((self.K,2,2),dtype=complex);
            self.M[0,:,:] = 1.0/3.0*np.array([[1, 0],[0, 0]])
            self.M[1,:,:] = 1.0/6.0*np.array([[1, 1],[1, 1]])
            self.M[2,:,:] = 1.0/6.0*np.array([[1, -1j],[1j, 1]])
            self.M[3,:,:] = 1.0/3.0*(np.array([[0, 0],[0, 1]]) + \
                                     0.5*np.array([[1, -1],[-1, 1]]) \
                                   + 0.5*np.array([[1, 1j],[-1j, 1]]) )
        elif POVM=='Tetra':
            self.K=4;

            self.M=np.zeros((self.K,2,2),dtype=complex);

            self.v1=np.array([0, 0, 1.0]);
            self.M[0,:,:]=1.0/4.0*( self.I + self.v1[0]*self.s1+self.v1[1]*self.s2+self.v1[2]*self.s3);

            self.v2=np.array([2.0*np.sqrt(2.0)/3.0, 0.0, -1.0/3.0 ]);
            self.M[1,:,:]=1.0/4.0*( self.I + self.v2[0]*self.s1+self.v2[1]*self.s2+self.v2[2]*self.s3);

            self.v3=np.array([-np.sqrt(2.0)/3.0 ,np.sqrt(2.0/3.0), -1.0/3.0 ]);
            self.M[2,:,:]=1.0/4.0*( self.I + self.v3[0]*self.s1+self.v3[1]*self.s2+self.v3[2]*self.s3);

            self.v4=np.array([-np.sqrt(2.0)/3.0, -np.sqrt(2.0/3.0), -1.0/3.0 ]);
            self.M[3,:,:]=1.0/4.0*( self.I + self.v4[0]*self.s1+self.v4[1]*self.s2+self.v4[2]*self.s3);
        elif POVM=='Trine':
            self.K=3;
            self.M=np.zeros((self.K,2,2),dtype=complex);
            phi0=0.0
            for k in range(self.K):
                phi =  phi0+ (k)*2*np.pi/3.0
                self.M[k,:,:]=0.5*( self.I + np.cos(phi)*self.Z + np.sin(phi)*self.X)*2/3.0
        else:
            raise NotImplementedError('POVM not implemented')
        if MPS=="GHZ":
            # Copy tensors used to construct GHZ as an MPS. The procedure below should work for any other MPS 
            cc = np.zeros((2,2)); # corner
            cc[0,0] = 2**(-1.0/(2*self.N));
            cc[1,1] = 2**(-1.0/(2*self.N));
            cb = np.zeros((2,2,2)); # bulk
            cb[0,0,0] = 2**(-1.0/(2*self.N));
            cb[1,1,1] = 2**(-1.0/(2*self.N));
            self.MPS = []
            self.MPS.append(cc)
            for i in range(self.N-2):
                self.MPS.append(cb)
            self.MPS.append(cc)
        else:
            raise NotImplementedError('MPS not implemented')
        self.state = self.state_init()
    def state_init(self):
        v = ([-1,1],)
        for i in range(2, self.N):
            v = v + ([i-1,-i,i],)
        v = v + ([-self.N, self.N-1],)
        return ncon(self.MPS, v)
    def prob(self, a):
        '''
        a: one-dim np-array containing POVM result: [N]
        '''
        AA = [self.state]
        for i in range(self.N):
            AA.append(self.M[a[i],:,:])
        AA.append(self.state)
        v = ([i for i in range(1,self.N+1)],)
        for i in range(1,self.N+1):
            v = v + ([i,i+self.N],)
        v = v + ([i+self.N for i in range(1,self.N+1)],)
        return ncon(AA,v).real
    def batch_prob(self, a):
        # a:[bs, N]
        probs = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            probs[i] = self.prob(a[i,:])
        return probs