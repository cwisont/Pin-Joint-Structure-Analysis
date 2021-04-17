import math

import numpy as np
from scipy import linalg as linalg
# from Structures_Class import *


#Define some basic equations
def subtract(X,Y):
    '''Return difference of two vectors'''
    return [x-y for x,y in zip(X,Y)]

def mult(a,X):
    '''Return a constant 'a' multiplied into a vector'''
    return [a*x for x in X]

def norm(X):
    '''Return norm of a vector'''
    return sum([x**2 for x in X])**(0.5)

def dot(X,Y):
    '''Return dot product of two vectors'''
    return sum([x*y for x,y in zip(X,Y)])

def cross(X,Y):
    '''Return cross product of two vectors'''
    if len(X) == 2 and len(Y) == 2:
        return [0.0,0.0,X[0]*Y[1]-Y[0]*X[1]]
    else:
        return [X[1]*Y[2]-Y[1]*X[2], Y[0]*X[2]-X[0]*Y[2], X[0]*Y[1]-Y[0]*X[1]]

def vProduct(x,Y):
    '''Return the product of a vector and a matrix'''
    return [dot(x,y) for y in Y]


def rref(B, tol=1e-8, debug=False):
  '''
  This finds the reduced row echelon form of a matrix (B) as well as the pivot positions
  '''
  A = B.copy()
  rows, cols = A.shape
  r = 0
  pivots_pos = []
  row_exchanges = np.arange(rows)
  for c in range(cols):
    # if debug: print ("Now at row", r, "and col", c, "with matrix:")
    # print(A)

    # Find the pivot row:
    pivot = np.argmax (np.abs (A[r:rows,c])) + r
    m = np.abs(A[pivot, c])
    # if debug: print("Found pivot", m, "in row", pivot)
    if m <= tol:
    # Skip column c, making sure the approximately zero terms are
    # actually zero.
      A[r:rows, c] = np.zeros(rows-r)
      # if debug: print( "All elements at and below (", r, ",", c, ") are zero.. moving on..")
    else:
      # keep track of bound variables
      pivots_pos.append((r,c))

      if pivot != r:
        # Swap current row and pivot row
        A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
        row_exchanges[[pivot,r]] = row_exchanges[[r,pivot]]

        # if debug: print( "Swap row", r, "with row", pivot, "Now:")
        # print(A)

      # Normalize pivot row
      A[r, c:cols] = A[r, c:cols] / A[r, c];

      # Eliminate the current column
      v = A[r, c:cols]
      # Above (before row r):
      if r > 0:
        ridx_above = np.arange(r)
        A[ridx_above, c:cols] = A[ridx_above, c:cols] - np.outer(v, A[ridx_above, c]).T
        # if debug: print( "Elimination above performed:")
        # print(A)
      # Below (after row r):
      if r < rows-1:
        ridx_below = np.arange(r+1,rows)
        A[ridx_below, c:cols] = A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
        # if debug: print( "Elimination below performed:")
        # print(A)
      r += 1
    # Check if done
    if r == rows:
      break;
  return (A, pivots_pos, row_exchanges)


def staticIndetermine(A,e_,F,tp):
    '''
    This algorithm comes from Pellegrino, S. (1990). Analysis of prestressed mechanisms. International Journal of Solids and Structures
    Calculates the self stress tensions in a structure due to an arbitrary load or member elongation
    A is the equilibrium matrix of the structure as a numPy array (see EQTR3)
    e_ is the applied elongations of each member in the structure in units of length as a numPy array (number of bars x 1)
    F is the matrix of axial flexibilities the structure, where the structure stiffness (Len/(E*A)) is on the diagonal as a numPy array
    tp is the applied load to the structure as a numPy array (number of nodes*3 - degrees of confinement).
    '''
    s = np.shape(linalg.null_space(A))[1] #Find number of self stressed mechanisms in A
    B = np.transpose(A)
    t = np.zeros(np.shape(e_));

    if s != 0:
        SS = linalg.null_space(A) #Self stress states
        alpha = linalg.inv(SS.T.dot(F).dot(SS)).dot(-SS.T.dot(e_+F.dot(tp))) #Equation 7
        t = np.dot(SS,alpha) #member tensions
    return t


def kinematicIndetermine(A,V,C,Len,t,e,l):
    '''
    This algorithm comes from Pellegrino, S. (1990). Analysis of prestressed mechanisms. International Journal of Solids and Structures

    Calculates the deformation of a structure (dx,dy,dz) x number of vertices due to applied loads and elongations, when subjected to a prestress. Also calculates the increase in tension in each structural member and the the mechanism load that is applied.
    A is the equilibrium matrix of the structure as a numPy array (see EQTR3)
    V is the matrix of the nodal coordinates and their constraints
    C is the conectivity of the structure
    Len is the length of the structural members
    e is the applied elongations of each member in the structure in units of length as a numPy array (number of bars x 1)
    t is the tension in each member of the structure (the self-stress)
    l is the applied load to the structure as a numPy array (number of nodes*3 - degrees of confinement).
    '''

    b = len(C) #number of bars
    j = len(V) #number of joints
    k = sum(sum(V[:,3:])) #constraints
    SS = linalg.null_space(A) #Self stress states
    s = np.shape(linalg.null_space(A))[1] #Find number of self stressed mechanisms in A
    r = np.linalg.matrix_rank(A)
    A,D,m = EQTR3(V,C)
    # need to find d for each node for each mechanism
    c = 0; d = np.zeros([m,3*len(V)])
    #Bar numbers and coordinates for mechanisms
    DC = [(x,i) for x in range(j) for i in range(3) if V[x,i+3]==0]
    #Determine G matrix ('geometric loads' or 'product forces')
    G = np.zeros(np.shape(D))
    for x in range(m):
        for i,dc in enumerate(DC):
            n0 = dc[0] #node connected to bar k
            ks = [y for y in range(b) if n0 in C[y,1:]] #bars connected to node n0
            g = []
            for k in ks:
                n1 = C[k,1] if C[k,2]==n0 else C[k,2] #other node connected to bar k
                if (n1,dc[1]) in DC:
                    jj = DC.index((n1,dc[1]))
                    g += [(D[i,x]-D[jj,x])*t[k]/Len[k]] # (eq. 15)
                else:
                    g += [D[i,x]*t[k]/Len[k]] #connected to a grounded node
            G[i,x] = sum(g)

    #Find Ar
    R,p,z = rref(A)
    p = np.asarray(p)
    Ar = A[:,p[:,0]] #matrix of nonredundant bars
    q = np.setdiff1d(range(len(t)),p[:,0])
    Ap = np.concatenate((Ar,G), axis = 1)

    dtrB = linalg.pinv(Ap).dot(l) #Equation 16
    beta = dtrB[r:]
    dtr = np.zeros((len(t),1))
    dtr[p[:,0]] = dtrB[:r]
    # dtr = np.insert(dtrB[:r],q,0)
    # dtr = np.expand_dims(dtr, axis=1)


    der = e[p[:,0]]
    Bp = Ap.T
    dep = np.concatenate((der,np.zeros((len(Bp)-len(der),np.shape(der)[1]))),axis=0)

    di = linalg.pinv(Bp).dot(dep) #equation 24
    dii = D.dot(beta) #Equation 11
    dlii = G.dot(beta) #Equation 12

    return di,dii,G,dlii,dtr

def nonLinearCorrect(A,V,C,F,dii,tp,dlii):
    '''
    This algorithm calculates the correction factor for type 4 structure (mechanism and self stress) under an applied load or member elongation.
    A is the equilibrium matrix of the structure as a numPy array (see EQTR3)
    V is the matrix of the nodal coordinates and their constraints
    C is the connectivity of the structure
    Len is the length of the structural members
    F is the matrix of axial flexibilities the structure, where the structure stiffness (Len/(E*A)) is on the diagonal as a numPy array
    dii is the mechanism displacement (calculated from kinematicIndetermine) of the structure
    tp is the tension in each member of the structure (the self-stress)
    dlii is the mechanism load on the structure (calculated from kinematicIndetermine)
    This algorithm comes from Pellegrino, S. (1990). Analysis of prestressed mechanisms. International Journal of Solids and Structures

    '''
    # Reformat dii to make the list comprehension nice
    dii_r = np.zeros((len(V), 3))
    count= 0
    n = len(C)
    # Add zero disp to constrained nodes
    for j,v in enumerate(V):
        for k in range(3):
            if v[k+3]!=1:
                dii_r[j, k] = dii[count]
                count += 1
            else:
                dii_r[j, k] = 0
    SS = linalg.null_space(A)
#      #   # "Undesired elongations": Elongations caused by the "inextensional" deformation
    Len = [np.linalg.norm((V[C[i, 1], 0:3]-V[C[i, 2], 0:3])) for i in range(n)]
    ec = [np.asarray([np.linalg.norm((V[C[k, 1], 0:3]+dii_r[C[k, 1], 0:3])-(V[C[k, 2], 0:3]+dii_r[C[k, 2], 0:3])) for k in range(n)])- Len] #The elongations in the members caused by the "inextensional" load
    ec = np.asarray(ec).T #Convert to convenient type and form
    alpha = (SS.T.dot(ec))/(SS.T.dot(F).dot(SS))
    tn = SS.dot(alpha) #Increase in prestress due to the inextensional deformation
    ec=ec-F.dot(SS).dot(alpha) #Equation 25
    # ec = ec[p[:,0]]
    dc,_,_,_,_ = kinematicIndetermine(A,V,C,Len,tp,-ec,dlii*0)

#      #   #Compute epsilon
    P = (2*SS.T.dot(F).dot(SS).dot(alpha**2))/(dlii.T.dot(dii))
    if math.isnan(P) or P<1e-5:
        e1 = [1.0+0j,1.0+0j]
        epsilon = 1
    else:
        e1 = np.roots([P,0,1,-1]) #equation 28
        epsilon = e1[e1.imag == 0].real  #The imaginary component should be zero
    dii = epsilon*dii+(epsilon**2*dc) #equation 26
    t=np.squeeze(epsilon**2*tn) #Apply correction to tensions
    dii_c = dii
    # di_save +=di
    # dii_save +=dii
    return t,dii_c

#EQTR
def EQTR3 (NODES,ELEM):

    '''The function EQTR returns the equilibrium matrix for any 3D truss
    NODES is list of the nodal coordinates of each vertex of the truss and the constraints on each
    NODES should be an nx6 array where the last 3 positions are either 1 or 0 indicating if there is a kinematic constraint in that direction
    ELEM is an array containing the connectivity of the truss and the type of each member
    ELEM stores the type of member in column 1 and the vertex numbers in columns 2 and 3
    '''
    for i in [1]:

        import numpy as np
        import math as m
        np.set_printoptions(threshold=np.inf)

        [nnode, var] = np.shape(NODES)

        if var != 6:
            raise Exception ('Error: NODE matrix of incorrect size')

        [nelem,lelem] = np.shape(ELEM)

        #Add remove constrained vertices and determine the degrees of freedom
        #ROWNO(i,j)=0 => displ. of node i in dir. j is constrained, hence no contribution to EQUIL
        #ROWNO(i,j)=n => contribution of d.o.f. j of node i to be stored in row no. n of EQUIL
        ROWNO = np.zeros([nnode,3])
        icount = 0
        for i in range(0,nnode):
            for j in range(0,3):
                if (NODES[i,j+3] == 0):
                    icount+=1
                    ROWNO[i,j] = icount
                else:
                    ROWNO[i,j] = 0


        ndof = icount+1
        EQUIL = np.zeros([ndof-1,ELEM.shape[0]])
        icol = -1

        for ielem in range(0,nelem):


            if (ELEM[ielem,0] == 1): #If the member is a pin jointed bar

                icol+=1
                C = np.concatenate((NODES[ELEM[ielem,1],0:3],NODES[ELEM[ielem,2],0:3]))
                len = m.sqrt((C[0]-C[3])**2+(C[1]-C[4])**2+(C[2]-C[5])**2)
                x = (C[0]-C[3])/len
                y = (C[1]-C[4])/len
                z = (C[2]-C[5])/len
                EQ1=[x,y,z,-x,-y,-z];
                ii = -1
                for i in range(1,3):
                    for j in range(0,3):
                        ii+=1
                        irow = int(ROWNO[ELEM[ielem,i],j])
                        if (irow > 0):
                            EQUIL[irow-1,icol] = EQ1[ii]


            elif (ELEM[ielem,0] == 2):  #member is a constant tension cable

                icol +=1
                for j in range(1,lelem-1):
                    if (ELEM[ielem,j+1] != -1):
                        C = np.concatenate((NODES[ELEM[ielem,j],0:3],NODES[ELEM[ielem,j+1],0:3]))
                        len = m.sqrt((C[0]-C[3])**2+(C[1]-C[4])**2+(C[2]-C[5])**2)
                        x = (C[0]-C[3])/len
                        y = (C[1]-C[4])/len
                        z = (C[2]-C[5])/len
                        EQ1=[x,y,z,-x,-y,-z];
                        ii = -1
                        for i in range(j,j+2):
                            for jj in range(0,3):
                                ii+=1;
                                irow=int(ROWNO[ELEM[ielem,i],jj])
                                if (irow > 0):
                                    EQUIL[irow-1,icol]+=EQ1[ii];

            else:
                raise Exception('Error in array ELEM, Unknown element type')
    # # print(EQUIL)

   ##Find and remove rigid body mechanisms
    #Make C matrix of kinematic constraints
    D = linalg.null_space(np.transpose(EQUIL)) #A basis for the set of inextensional mechanisms
    C = []
    k = NODES[:,3:6]
    # k = np.reshape(k,(np.shape(k)[0]*3,1))
    (X,Y) = np.where(k==1)
    V = NODES[:,0:3]
    for i,x in enumerate(X):
        CX = [[1,0,0,0,V[x,2],-V[x,1]],[0,1,0,-V[x,2],0,V[x,0]],[0,0,1,V[x,1],-V[x,0],0]] #Constraint equations for x,y,and z kinematic constraints
        C += [CX[Y[i]][:]]

    C = np.asarray(C)
    if C.size == 0:
        D = linalg.null_space(EQUIL.T)
        m = np.shape(D)[1]
        return EQUIL,D,m

    nr = linalg.null_space(C)

    if nr.size==0 or C.size ==0: #If fully constrained
        D = linalg.null_space(EQUIL.T)
        m = np.shape(D)[1]
        return EQUIL,D,m
    U=[]
    for v in V:
        U += [nr[:3]+np.cross(nr[3:].T,v).T] #rigid body mechanism in terms of coordinates
    U = np.asarray(U)
    U = np.reshape(U,(np.shape(U)[0]*np.shape(U)[1],np.shape(U)[2]))
    U = np.delete(U,3*X+Y,axis = 0) #remove directions stopped by constraints
    # U = np.reshape(U,(np.shape(U)[0],1))

    rb = 6 - np.linalg.matrix_rank(C) #Total number of rigid body mechanisms
    m = np.linalg.matrix_rank(linalg.null_space(np.transpose(EQUIL))) #Find number of inextensional mechanisms in EQUIL
    B = linalg.null_space(C)
    im = m - rb #The total number of non rigid body mechanisms

 #     A = linalg.null_space(np.transpose(EQUIL)) #Computes a basis for the system mechanisms
    Delta = np.zeros((np.shape(D)[0]))
    B = np.zeros((np.shape(D)))
    for i in range(m):
        for j in range(rb): #Orthogonalize rigid body mechanisms?
            Delta += D[:,i].dot(U[:,j])/(np.linalg.norm(U[:,j]))**2 *U[:,j]
        B[:,i] = D[:,i]-Delta
        Delta = np.zeros((np.shape(D)[0]))

    [non, p, re] = rref(B)
    if p==[]: D = np.zeros((np.shape(D)[0], 1))
    else:
        p = np.asarray(p)
        D = D[:,p[:,0]]

    return EQUIL,D,im




