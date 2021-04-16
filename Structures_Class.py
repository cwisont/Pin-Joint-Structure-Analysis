
from math import *
import numpy as np

def AmendolaPreset(preStrain = 0.036,Q = 0.048e2):
    '''
    This function is a preset structure of Amendola's rigid top simplex simulation in Amendola, A., Carpentieri, G., de Oliveira, M., Skelton, R. E., & Fraternali, F. (2014). Experimental investigation of the softening-stiffening response of tensegrity prisms under compressive loading. Composite Structures, 117(1), 234–243. https://doi.org/10.1016/j.compstruct.2014.06.022
    Returns the structure class
    '''
    # A Amendola
    sc = 0 #length scale (mm)
    l0 = 0.03408 #side tendon length (for simplex)
    H = 0.02229
    # b0 = sqrt(s0**2+(2*l0**2/sqrt(3)))
    # l0 = sqrt((H**2-s0**2)/(1/3*(sqrt(3)-2)))
    D1 = l0
    # V,C,T,e_per,l = simplex(H,D1,1.125)
    # T = T*0.0014
    E1 = 112.3e6 #Bar stiffness
    E2 = 5.48e9 #Tendon stiffness
    E3 = 10000e9 #Simulated rigid top
    E = np.concatenate((E1*np.ones((3,1)),E2*np.ones((3,1)),E3*np.ones((6,1))),axis = 0)
    LW = 1200 #Linewidth for printing (should be just T is T>1)
    T =[0.0015,0.0001]
    ctrlE = True
    Struct = simplex(H,D1,T,E1 = E,E2 = E,shrink = -preStrain,LW = LW)
    return sc,Q,Struct


def FraternaliPreset(preStrain = 0.036,Q = 0.18e3/3):
    '''
    This function is a preset structure of Fraternali's simplex simulation in Fraternali, F., Carpentieri, G., & Amendola, A. (2015). On the mechanical modeling of the extreme softening/stiffening response of axially loaded tensegrity prisms. Journal of the Mechanics and Physics of Solids. https://doi.org/10.1016/j.jmps.2014.10.010
    Returns the structure class
    Q is the load applied to each node in newton, the total load is 3*Q
    preStress is the approximate delta in strain applied to the bars and tendons
    '''
    # Fraternali
    sc = 0 #length scale (um)
    s0 = 0.080 #side tendon length (for simplex)
    l0 = 0.132 #Base tendon length (for simplex)
    H = sqrt(s0**2+1/3*(sqrt(3)-2)*l0**2)
    b0 = sqrt(s0**2+(2*l0**2/sqrt(3)))
    D1 = l0
    # V,C,T,e_per,l = simplex(H,D1,1.125)
    # T = T*0.0014
    E1 = 203.3e9 #Bar stiffness
    E2 = 5.48e9 #Tendon stiffness
    E = np.concatenate((E1*np.ones((3,1)),E2*np.ones((9,1))),axis = 0)
    LW = 1000 #Linewidth for printing (should be just T is T>1)
    T =[0.00341,0.000378]
    Struct = simplex(H,D1,T,E1 = E, E2 = E,shrink =-preStrain,LW = LW)
    ctrlE = True #Takes stiffness from above not from class
    return sc,Q,Struct

class simplex:
    '''This class defines the various parameters in a simplex structure.
    The outputs are:
                V: Set of nodal coordinates (nx6) with the columns 1-3 representing the nodes and columns 4-6 representing the confinement of the structure
                C: Set of connectivities of the structure (bx3) where column 1 represents whether the member is a bar (1) or a tendon (2)
                T: Thickness of each member
                e_: The prescribed elongation percent of each member
                l: the loads applied to the structure
    This class requires inputs of the geometric dimensions (Height H and width w), the bar and tendon thicknesses, as well as the tendon radius (p). They take option arguments of the pre and post pyrolysis stiffnesses, defined shrinkage (not the experimental shrinkage), and an option to confine the structure for just the loading step.
    '''
    def __init__(self,H,w,T,E1 = 2.7e9, E2 = 36e9,shrink = None, postConfine = False,LW = 5):
        self.name = 'simplex'
        self.height = H
        self.width = w
        self.E1 = E1
        self.E2 = E2
        self.T1 = T
        self.LW = LW

        #Shrinkage
        if shrink == None:
            #Add global shrinkage onto initial geometry
            SHRNK = [-(0.7391434 + 0.1007027*exp(-0.08831393*t*2)) for t in T] #Experimental equation

            H = H*(1+SHRNK[0]) #Adjust height to minimize applied shrinkage
            w = w*(1+SHRNK[0])
            e_ = [0,SHRNK[1]-SHRNK[0]] #Only apply differential shrinkage
            e_trans = list(map(lambda t: -(0.0003*(2*t)**2-0.0054*(2*t)+0.8251),T))
            T = [t*(1+e) for (t,e) in zip(T,e_trans)] #Adjust to post pyrolysis shrinkage
        else:
            e_ = [-shrink,0]
            e_trans = [0 for t in T]

        self.T2 = T
        self.e_trans = e_trans

        #w is the length of the base tendons
        R = w*sin(pi/6)/sin(2*pi/3) #radius of the triangle vertices
        n = 3 #Number of bars
        thetal = -2*pi/3 #Angle between nodes
        thetah = -(pi/2-pi/n) #twist angle

        #Vertices
        V = [(R, 0, -H/2, 0, 0, 1),(R*cos(thetal), R*sin(thetal), -H/2, 0, 0, 1),(R*cos(2*thetal), R*sin(2*thetal),-H/2, 0, 0, 1), (R*cos(thetah), R*sin(thetah), H/2, 0, 0, 0),(R*cos(thetal+thetah), R*sin(thetal+thetah), H/2, 0, 0, 0),(R*cos(2*thetal+thetah), R*sin(2*thetal+thetah), H/2, 0, 0, 0)]
        self.V = V

        #Connectivity
        self.C = [(1, 0, 4),(1, 1, 5),(1, 2, 3),(2, 0, 3),(2, 1, 4),(2, 2, 5),(2, 0, 1),(2, 1, 2),(2, 2, 0),(2, 3, 4),(2, 4, 5),(2, 5, 3)]

        # Loads
        self.l = [3,4,5] #Coordinates of V that the loads are applied to
        self.e_ = e_

        # self.LW = 1

        if postConfine:
            self.postConfine = [(0., 1., 1.),
                                (1., 1., 1.),
                                (0., 0., 1.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.)]
        else:
            self.postConfine = [(0., 0., 1.),
                                (0., 0., 1.),
                                (0., 0., 1.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.)]


class icosahedron:
    '''This class defines the various parameters in an icosahedron structure.
    The outputs are:
                V: Set of nodal coordinates (nx6) with the columns 1-3 representing the nodes and columns 4-6 representing the confinement of the structure
                C: Set of connectivities of the structure (bx3) where column 1 represents whether the member is a bar (1) or a tendon (2)
                T: Thickness of each member
                e_: The prescribed elongation percent of each member
                l: the loads applied to the structure
    This class requires inputs of the geometric dimensions (Height H and width w), the bar and tendon thicknesses, as well as the tendon radius (p). They take option arguments of the pre and post pyrolysis stiffnesses, defined shrinkage (not the experimental shrinkage), and an option to confine the structure for just the loading step.
    '''
    def __init__(self,H,w,T,E1 = 2.7e9, E2 = 36e9,shrink = None, postConfine = False,LW = 5):
        self.name = 'icos'
        self.height = H
        self.width = w
        self.E1 = E1
        self.E2 = E2
        self.T1 = T
        self.LW = LW

        #Shrinkage
        if shrink == None:
            #Add global shrinkage onto initial geometry
            SHRNK = [-(0.7391434 + 0.1007027*exp(-0.08831393*t*2)) for t in T] #Experimental equation
            H = H*(1+SHRNK[0]) #Adjust height to minimize applied shrinkage
            e_ = [0,SHRNK[1]-SHRNK[0]] #Only apply differential shrinkage
            e_trans = list(map(lambda t: -(0.0003*(2*t)**2-0.0054*(2*t)+0.8251),T))
            T = [t*(1+e) for (t,e) in zip(T,e_trans)] #Adjust to post pyrolysis shrinkage
        else:
            e_ = [-shrink,0]
            e_trans = [0 for t in T]

        self.T2 = T
        self.e_trans = e_trans

        #Vertices
        h = H
        V = [(h/2, 0, -h/4, 0,0,0),
            (h/2, 0, h/4,0,0,1),
            (-h/2, 0, -h/4, 0,0,0),
            (-h/2, 0, h/4, 0,0,0),
            (h/4, h/2, 0, 0,0,0),
            (h/4, -h/2, 0, 0,0,1),
            (-h/4, -h/2, 0, 0,0,0),
            (-h/4, h/2, 0, 0,0,0), #top 21
            (0, h/4, h/2, 0,0,0),
            (0, h/4, -h/2, 0,0,0), #top 27
            (0, -h/4, -h/2, 0,0,0),
            (0, -h/4, h/2, 0,0,1)]

        Vert = [tuple(v[:3]) for v in V]
        J = subtract(Vert[2],Vert[9])
        K = subtract(Vert[7],Vert[9])
        Tn = mult(1/norm(cross(J,K)),cross(J,K))
        N = mult(1/norm(J),J)
        R = mult(1/norm(cross(Tn,N)),cross(Tn,N))

        B = [N]+[R]+[Tn] #Basis matrix

        for i,v in enumerate(V): #Rotate icos to desired position
            V[i] = tuple(vProduct(v[:3],B)+[v[3]]+[v[4]]+[v[5]])
        self.V = V

        #Connectivity
        self.C = [(1, 1, 3),(1, 5, 4),(1, 11, 10),(1, 0, 2),(1, 6, 7),(1, 8, 9),(2, 1, 11),(2, 1, 5),(2, 5, 11),(2, 5, 0),(2, 5, 10),(2, 11, 3),(2, 11, 6),(2, 1, 4),(2, 1, 8),(2, 4, 0),(2, 0, 9),(2, 0, 10),(2, 2, 6),(2, 2, 7),(2, 2, 10),(2, 2, 9),(2, 3, 6),(2, 3, 7),(2, 3, 8),(2, 4, 8),(2, 4, 9),(2, 6, 10),(2, 7, 8),(2, 7, 9)]

        #loads
        self.l = [2,7,9] #The nodes that the load is applied on
        self.e_ = e_

        self.LW = LW


        if postConfine:
            self.postConfine = [(0., 0., 0.),
                                (1., 1., 1.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (1., 0., 1.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 1.)]
        else:
            self.postConfine = [(0., 0., 0.),
                                (0., 0., 1.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 1.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 1.)]



class tetrakaidecahedron:
    '''
    This class defines the various parameters in a tetrakaidecahedron structure.
    The outputs are:
        V: Set of nodal coordinates (nx6) with the columns 1-3 representing the nodes and columns 4-6 representing the confinement of the structure
        C: Set of connectivities of the structure (bx3) where column 1 represents whether the member is a bar (1) or a tendon (2)
        T: Thickness of each member
        e_: The prescribed elongation percent of each member
        l: the loads applied to the structure
    This class requires inputs of the geometric dimensions (Height H and width w), the bar and tendon thicknesses, as well as the tendon radius (p). They take option arguments of the pre and post pyrolysis stiffnesses, defined shrinkage (not the experimental shrinkage), and an option to confine the structure for just the loading step.
    '''
    def __init__(self,H,w,T,E1 = 2.7e9, E2 = 36e9,shrink = None, postConfine = False,LW = 1):
        self.name = 'tetkai'
        self.height = H
        self.width = w
        self.E1 = E1
        self.E2 = E2
        self.T1 = T
        self.LW = 1

        #Shrinkage
        if shrink == None:
            #Add global shrinkage onto initial geometry
            SHRNK = [-(0.7391434 + 0.1007027*exp(-0.08831393*t*2)) for t in T] #Experimental equation
            H = H*(1+SHRNK[0]) #Adjust height to minimize applied shrinkage
            w  = w*(1+SHRNK[0])
            e_ = [0,SHRNK[1]-SHRNK[0]] #Only apply differential shrinkage
            e_trans = list(map(lambda t: -(0.0003*(2*t)**2-0.0054*(2*t)+0.8251),T))
            T = [t*(1+e) for (t,e) in zip(T,e_trans)] #Adjust to post pyrolysis shrinkage
        else:
            e_ = [-shrink,0]
            e_trans = [0 for t in T]

        self.T2 = T
        self.e_trans = e_trans

        h = H
        phi = 0.3287663062125372
        if round(h/w,2) != 1.5:
            print('H:W must be 3:2 for tetkai\nIf you need a different config, you must run form finding to find new phi and replace in code')
        # Nodes
        #Define nodes of the cell

        V = [(-w/2*sin(phi/2),-w/2*cos(phi/2),-h/2,1,1,1),
            (-w/2*cos(phi/2),w/2*sin(phi/2),-h/2,0,0,1),
            (w/2*sin(phi/2),w/2*cos(phi/2),-h/2,0,0,1),
            (w/2*cos(phi/2),-w/2*sin(phi/2),-h/2,0,0,1),
            (w/2*sin(phi/2),-w/2*cos(phi/2),h/2,0,0,0),
            (-w/2*cos(phi/2),-w/2*sin(phi/2),h/2,0,0,0),
            (-w/2*sin(phi/2),w/2*cos(phi/2),h/2,0,0,0),
            (w/2*cos(phi/2),w/2*sin(phi/2),h/2,0,0,0),
            (-h/2,-w/2*cos(phi/2),w/2*sin(phi/2),0,0,0),
            (-h/2, w/2*sin(phi/2),w/2*cos(phi/2),0,0,0),
            (-h/2,w/2*cos(phi/2),-w/2*sin(phi/2),0,0,0),
            (-h/2, -w/2*sin(phi/2),-w/2*cos(phi/2),0,0,0),
            (h/2,-w/2*cos(phi/2),-w/2*sin(phi/2),0,0,0),
            (h/2, -w/2*sin(phi/2),w/2*cos(phi/2),0,0,0),
            (h/2,w/2*cos(phi/2),w/2*sin(phi/2),0,0,0),
            (h/2, w/2*sin(phi/2),-w/2*cos(phi/2),0,0,0),
            (-w/2*sin(phi/2),-h/2,w/2*cos(phi/2),0,0,0),
            (-w/2*cos(phi/2),-h/2,-w/2*sin(phi/2),0,0,0),
            (w/2*sin(phi/2),-h/2,-w/2*cos(phi/2),0,0,0),
            (w/2*cos(phi/2),-h/2,w/2*sin(phi/2),0,0,0),
            (w/2*sin(phi/2),h/2,w/2*cos(phi/2),0,0,0),
            (-w/2*cos(phi/2),h/2,w/2*sin(phi/2),0,0,0),
            (-w/2*sin(phi/2),h/2,-w/2*cos(phi/2),0,0,0),
            (w/2*cos(phi/2),h/2,-w/2*sin(phi/2),0,0,0)]

        # Members
        C =[(1,0, 14),(1,1, 19),(1,2, 8),(1,3, 21),(1,9, 18),(1,11, 20),(1,13, 22),(1,15, 16),(1,4, 10),(1,5, 23),(1,6, 12),(1,7, 17),(2,0, 1),(2,0, 3),(2,1, 2),(2,2, 3),(2,14, 15),(2,15, 3),(2,19, 18),(2,18, 0),(2,8, 11),(2,11, 1),(2,21, 22),(2,22, 2),(2,8, 17),(2,17, 18),(2,19, 12),(2,12, 15),(2,14, 23),(2,23, 22),(2,21,10),(2,10, 11),(2,10, 9),(2,9, 8),(2,17, 16),(2,16, 19),(2,12, 13),(2,13, 14),(2,23, 20),(2,20, 21),(2,4, 16),(2,5, 9),(2,6, 20),(2,7, 13),(2,7, 4),(2,6, 7),(2,4,5),(2,5,6)]

        self.V = V
        self.C = C

        # Loads applied to the structure
        self.l = [4,5,6,7] #The nodes that the load is applied on
        self.e_ = e_

        self.LW = 1


        if postConfine:
            self.postConfine = [(1., 1., 1.),
                                (0., 0., 1.),
                                (0., 0., 1.),
                                (0., 0., 1.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.)]
        else:
            self.postConfine = [(0., 0., 1.),
                                (0., 0., 1.),
                                (0., 0., 1.),
                                (0., 0., 1.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.),
                                (0., 0., 0.)]


class hangNet:

    def __init__(self):
        self.sc = -3
        self.Q = 100 #N
        self.LW = 10
        self.name = 'hanging Net'
        self.height = H
        self.p = p
        self.width = w
        self.E1 = 202e9
        self.E2 = 202e9

        V = [[-961.,-305,155,1,1,1],
            [-961,305,155,1,1,1],
            [-305,-961,-146.,1,1,1],
            [-305,-305,0,0,0,0],
            [-305,305,0,0,0,0],
            [-305,961,-146,1,1,1],
            [305,-961,-146,1,1,1],
            [305,-305,0,0,0,0],
            [305,305,0,0,0,0],
            [305,961,-146,1,1,1],
            [961,-305,155,1,1,1],
            [961,305,155,1,1,1]]

        C = [[1,0,3],
            [1,3,7],
            [1,10,7],
            [1,1,4],
            [1,4,8],
            [1,11,8],
            [1,2,3],
            [1,3,4],
            [1,5,4],
            [1,6,7],
            [1,7,8],
            [1,9,8]]

        D = 0.42
        # Member thickness
        T = [D/2,D/2]

        e_trans = [0,0]

        A,_,_ = EQTR3(np.asarray(V),np.asarray(C)); #Create the Equilibrium matrix
        SS = linalg.null_space(A)
        ss = SS/np.linalg.norm(SS)
        e_ = np.squeeze(-0.01*ss)
        e_[[(1),(4),(7),(10)]] = -0.25

        # l = [2,5,8,11] = -Q
        l = [2,11]

        self.V = V
        self.C = C
        self.T1 = T
        self.e_trans = e_trans
        self.T2 = T
        self.e_ = e_
        self.l = l
        self.postConfine = False


# # 3 bar m = 2 (Pellegrino example)
# def MemberNet():
    # sc= -3;
    # Q = 0.005;
    # E = 2e9
    # LW = 1
    # V,C,T,e_per,l = MemberNet()
    # dims = 2
#     # Initial Nodal Positions
#     V = [[0.,0,0,1,1,1],
#         [160.,0,0,0,0,1],
#         [320.,0,0,0,0,1],
#         [480.,0,0,1,1,1]])
#
#     # Connectivity
#     C = [[1,0,1],
#         [1,1,2],
#         [1,2,3]]
#
#     # Member thickness
#     T = [[3],
#         [3],
#         [3]]
#
#     # Elongation percent
#     e_ = [[0],
#         [-0.01],
#         [0]]
#
#     # Define load on the system
#     l = [1,2] #loaded nodes
#
#     self.V = V
#     self.C = C
#     self.T = T
#     self.e_ = e_
#     self.l = l
#     # return (V,C,T,e_,l)


def hedron4(w):


    # w = 40

    # Initial Nodal Positions
    V = [[-w/2, 0, 0, 0, 0, 0],
                  [0, -w/2, 0, 0, 0, 0],
                  [w/2, 0, 0, 0, 0, 0],
                  [0, w/2, 0, 0, 0, 0],
                  [0, 0, w/2, 0, 0, 0],
                  [0, 0, -w/2, 0, 0, 0]]


    # Connectivity
    C = [[1, 0, 4],
                  [1, 1, 4],
                  [1, 2, 4],
                  [1, 3, 4],
                  [1, 0, 1],
                  [1, 1, 2],
                  [1, 2, 3],
                  [1, 3, 0],
                  [1, 0, 5],
                  [1, 1, 5],
                  [1, 2, 5],
                  [1, 3, 5],
                  [1, 4, 5]]

    # Member thickness
    T = [[3],
                [3],
                [3],
                [3],
                [3],
                [3],
                [3],
                [3],
                [3],
                [3],
                [3],
                [3],
                [0.75]]

    gamma = .605
    delta = 0.01
    #Shrinkage percent per bar
    e_ = [[-gamma],
                [-gamma],
                [-gamma],
                [-gamma],
                [-gamma],
                [-gamma],
                [-gamma],
                [-gamma],
                [-gamma],
                [-gamma],
                [-gamma],
                [-gamma],
                [-(gamma+delta)]]

    l = []
    return (V,C,T,e_,l)

class StackedSimplex:
    def __init__(self,H,w,T,n = 3,m = 2,E1 = 2.7e9, E2 = 36e9,shrink = None, postConfine = False,LW = 1, theta2 = 0.12739665775465892):
        self.name = 'StackedSimplex'
        self.height = H
        self.width = w
        self.E1 = E1
        self.E2 = E2
        self.T1 = T
        self.LW = 1

        #Shrinkage
        if shrink == None:
            #Add global shrinkage onto initial geometry
            SHRNK = [-(0.7391434 + 0.1007027*exp(-0.08831393*t*2)) for t in T] #Experimental equation
            H = H*(1+SHRNK[0]) #Adjust height to minimize applied shrinkage
            w  = w*(1+SHRNK[0])
            e_ = [0,SHRNK[1]-SHRNK[0]] #Only apply differential shrinkage
            e_trans = list(map(lambda t: -(0.0003*(2*t)**2-0.0054*(2*t)+0.8251),T))
            T = [t*(1+e) for (t,e) in zip(T,e_trans)] #Adjust to post pyrolysis shrinkage
        else:
            e_ = [-shrink,0]
            e_trans = [0 for t in T]

        self.T2 = T
        self.e_trans = e_trans

        h = H

        handed = -1
        r = w*sin(pi/6)/sin(2*pi/3) #radius of the triangle vertices
        theta = handed*(pi/2 - pi/n) #the twist angle
        phi = handed*2*pi/n #the angle between vertices
        delta = (pi/2 - pi/n) #the twist angle
        v = n*3 #number of vertices
        angle = delta*180/pi
        zeta = 2*pi/n #the angle between vertices
        gamma =  (zeta/2-delta)*(1+2*(n%2)) # the angle of rotation of each prism to the last
        R = np.array([[cos(gamma), -sin(gamma), 0],
                    [sin(gamma), cos(gamma), 0],
                    [0,          0,          1]])
        D = np.zeros((3*n,3));
        D[:,2] = 0.7*h;

        b = n #number of struts
        t = 5*n #number of tendons

        Q = np.zeros((v,3)) #number of vertices by 3 coordinates
        W = np.zeros((2*v,3))
        #from polar coordinates
        count = n+1
        for k in range(len(Q[:,1])+1):
            if k <= n:
                    Q[k-1,:] = [r*cos((k-1)*phi), r*sin((k-1)*phi), 0]

            if k > n:
                Q[count-1,:] = [r*cos((count-1)*phi/2+theta), r*sin((count-1)*phi/2+theta), h-0.3*h*(count%2)]
                count = count+1
        W[:v,:] = Q

        handed = 1
        phi = handed*2*pi/n #the angle between vertices

        count = n+1
        for k in range(7): #Second cell
            if k <= n:
                    Q[k-1,:] = [r*cos((k-1)*phi), r*sin((k-1)*phi), 0]

            if k > n:
                Q[count-1,:] = [r*cos((count-1)*phi+theta2), r*sin((count-1)*phi+theta2), h]
                count = count+1

        Q = Q+D #stacks cell
        Q = Q.dot(np.linalg.matrix_power(R,int(-(handed+1)/2))) #rotates the cell
        W[v:2*v,:] = Q
        W = np.delete(W,slice(15,18), 0)
        #Bars
        B = np.array([[0,3],[1,5],[2,7],
                    [8,9],[6,10],[4,11]])
        B = np.concatenate((np.ones((len(B),1)),B),axis = 1)

        #Tendons
        Tend = np.array([[0,5],[1,7],[2,3],#vertical 1
                    [0,4],[1,6],[2,8],#vertical 2
                    [0,1],[1,2],[2,0],#horizontal 1
                    [3,4],[4,5],[5,6],#horizontal 2
                    [6,7],[7,8],[8,3],#Vertical 3
                    [4,9],[8,10],[6,11],
                    [9,10],[10,11],[11,9],
                    [5,11],[7,10],[3,9]])
        Tend = np.concatenate((2*np.ones((len(Tend),1)),Tend),axis = 1)


        C = np.concatenate((B,Tend),axis = 0)
        V = W[:]

        #Remove duplicate rows
        dups = []
        for i,v in enumerate(V[:,:3]):
            W = np.delete(V,range(i),axis = 0)[:]
            for j,w in enumerate(W[:,:3]):
                if (abs(v-w)<0.01).all() and j+i!=i:
                    dups+=[(i,j+i)]
        dups = np.array(dups)
        V = np.delete(V,dups[:,1],axis = 0)

        e_ = [-(0.1391434 + 0.1007027*exp(-0.08831393*t*2)) for t in T]
        e_ = np.reshape(e_,(len(T),1))

        V = np.concatenate((V,np.zeros((len(V),3))),axis = 1)
        # V[0,3:] = [1,1,1] #kinematic constraints
        V[0:3,3:] = [0,0,1] #kinematic constraints

        l = [11,10,9]

        self.V = V
        self.C = C
        self.e_trans = e_trans
        self.T2 = T
        self.e_ = e_
        self.l = l
        self.postConfine = False

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



    [non,p,re] = rref(B)
    if p==[]: D = np.zeros((np.shape(D)[0],1))
    else:
        p = np.asarray(p)
        D = D[:,p[:,0]]

    return EQUIL,D,im




