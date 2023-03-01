import numpy as np
from utils.r_eval import compute_R_diff

def projM2R(m):
    u, _, v = np.linalg.svd(m)
    R = u@v
    R = R.T
    # R = np.sqrt(3)* R / np.sqrt(np.sum(np.square(R))+1e-6) #SVD guaranteed
    if np.linalg.det(R)<0:
        R = R[[1,0,2]]
    return R

def eigenrs(locws, Rs, N, normalized = False):
    # normalized: if we use normalized Laplacian matrix of not
    L = np.eye(N*3)
    # construct D
    Ddiag = np.zeros([N])
    for pid in range(locws.shape[0]):
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        Ddiag[i] += w
    if not normalized:
        for i in range(N):
            xs, xe, ys, ye = i*3, i*3+3, i*3, i*3+3
            L[xs:xe, ys:ye] = np.eye(3)*Ddiag[i]
    # construct -A
    for pid in range(locws.shape[0]):
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        xs, xe, ys, ye = i*3, i*3+3, j*3, j*3+3
        if normalized:
            L[xs:xe, ys:ye] = -w*Rs[pid]/np.sqrt(Ddiag[i]*Ddiag[j]+1e-8)
        else:
            L[xs:xe, ys:ye] = -w*Rs[pid]
    # Rs
    w, v = np.linalg.eig(L)
    w, v = w.real.astype(np.float32), v.real.astype(np.float32)
    # small to large
    args = np.argsort(w)
    w = w[args[0:3]]
    v = v[:,args[0:3]]
    Rpre = []
    for i in range(N):
        m = v[i*3:(i+1)*3,:]
        r = projM2R(m)
        Rpre.append(r[None,:,:])
    return np.concatenate(Rpre, axis=0)

def leastsquare(locws, ts, pcrs, N):
    '''
    base :  RiPi+ti = RjPj+tj
            Pi = RijPj+tij
            --> Ri(RijPj+tij)+ti = RjPj+tj
            --> RiRij = Rj   Ritij+ti = tj
            --> -ti + tj = Ritij (3 formular)
    least square:
        B: (p*3)*(N*3)
        P: (p*3)*(p*3)
        L: Ritij--> (p*3)        
    '''
    tpre = np.zeros([N,3])
    p = locws.shape[0]
    B = np.zeros([p*3, N*3])
    P = np.eye(p*3)
    L = np.zeros([p*3])
    # get B and P
    for pid in range(locws.shape[0]):
        # the pid-th pair
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        P[pid*3:(pid+1)*3] *= w
        B[pid*3:(pid+1)*3, i*3:(i+1)*3] = -np.eye(3)
        B[pid*3:(pid+1)*3, j*3:(j+1)*3] = np.eye(3)
    # get L
    for pid in range(locws.shape[0]):
        # the pid-th pair
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        L[pid*3:(pid+1)*3] = pcrs[i]@ts[pid] - (tpre[j]-tpre[i])
    # delta
    deltat = np.linalg.pinv(B.T@P@B)@(B.T@P@L)
    tpre += deltat.reshape(N,3)
    
    # final T
    Tpre = []
    for i in range(N):
        r = pcrs[i]
        t = tpre[i]
        T = np.eye(4)
        T[0:3,0:3] = r
        T[0:3,-1] = t
        Tpre.append(T[None,:,:])
    Tpre = np.concatenate(Tpre, axis=0)
    return Tpre 

def LaplacianT(locws, Rs, ts, N, calt = False):
    '''
    Generate the pc transformation to global coor from the pair transformations(Tpair graph)
    locws: p*3 [(i,j,w)]
    Rs: p*3*3
    ts: p*3
    N: the number of point cloud
    '''
    # for Rpres
    Rpre = eigenrs(locws, Rs, N)
    # for tpres --> Ts
    if calt:
        Tpre = leastsquare(locws, ts, Rpre, N)
    else:
        Tpre = np.eye(4)[None,:,:].repeat(N,axis=0)
        for i in range(N):
            Tpre[i,0:3,0:3]=Rpre[i]
    return Tpre
 
def error_reweight(iterid, locws, Ts, Tpre, all_iters = 50):
    for pid in range(locws.shape[0]):
        i, j = int(locws[pid,0]), int(locws[pid,1])
        Ti = Tpre[i]
        Tj = Tpre[j]
        Tijfit = np.linalg.inv(Ti)@Tj
        Tijpre = Ts[i,j]      
        rdiff = compute_R_diff(Tijfit[0:3,0:3], Tijpre[0:3,0:3])
        # re-weighting
        rdiff = 2*rdiff*(iterid+1)/np.sum(np.arange(all_iters)+1)
        rdiff = np.exp(-rdiff)
        locws[pid,-1] *= rdiff
    return locws

def keep_symmetry(locws,N):
    mat = np.zeros([N,N])
    for pid in range(locws.shape[0]):
        i, j, w = int(locws[pid,0]), int(locws[pid,1]), locws[pid,2]
        if mat[j,i]>0:
            mat[i,j] = min(w,mat[i,j])
            mat[j,i] = min(w,mat[j,i])
        else:
            mat[i,j] = w
            mat[j,i] = w
    # check
    for pid in range(locws.shape[0]):
        locws[pid,2] = mat[int(locws[pid,0]),int(locws[pid,1])]
    return locws
    
def pair2globalT_cycle(W_edges, Ts, iters):
    # Input\W_edges: N*N
    # Input\Ts:      N*N*4*4
    # Input\iters:   100
    # Intermediate:
    #   locws --> p*[i,j,w] (None zero w)
    #   Rs --> p*[3*3]
    # Output\Tpre:   N*4*4
    # Output\locs:  pair indexes and pair transformations
    N = W_edges.shape[0]
    locws = []
    Rs = []
    ts = []
    for i in range(Ts.shape[0]):
        for j in range(Ts.shape[1]):
            if W_edges[i,j]>0:
                locws.append(np.array([[i,j,W_edges[i,j]]]))
                Rs.append(Ts[i,j,0:3,0:3][None,:,:])
                ts.append(Ts[i,j,0:3,-1][None,:])
    Rs = np.concatenate(Rs, axis=0)
    ts = np.concatenate(ts, axis=0)
    locws = np.concatenate(locws, axis=0)
    # cycle re-weighting
    Tpre = LaplacianT(locws, Rs, ts, N)
    for i in range(iters):
        locws = error_reweight(i, locws, Ts, Tpre, all_iters = iters)  
        locws = keep_symmetry(locws,N) 
        Tpre = LaplacianT(locws, Rs, ts, N, calt=False)
    Tpre = LaplacianT(locws, Rs, ts, N, calt=True)
    return Tpre, locws