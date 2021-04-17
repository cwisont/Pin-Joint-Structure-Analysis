
'''This program returns the predicted nodal displacements and member tensions given the geometry, loads, and elongations of a pin jointed structure.
This code is based on the work done by Sergio Pellegrino available at http://dx.doi.org/10.1016/0020-7683(90)90082-7

This program depends on,Indeterminacy.py, and Structures_Class.py, which must be run BEFORE running this program.
The structure definitions are contained in the structure classes in Structures_Class.py. Each class in here should have all the required information in the correct form to run in this program. The class contains the member elongation and points of loading, which can only be modified from that program.

The program can sweep through a list of structures given to it and output the responses to a csv file in the desired directory. That directory is controlled in the "Control file directory" section of the code (just below).

The information displayed can also be controlled in the "Initial Geometry and Mechanical Properties" section of the code by changing the following
    #Show different plots
    showShrink = 0 #Shows pre and post pyrolysis overlayed 3d models
    showLoad = 0 #Shows load vs disp response
    showStress = 1 #Shows stress in members throughout loading as well as load vs disp
    showShape = 0 #Shows initial and post loading shape in 3d plot
    showIter = 0 #Shows the delta t for each iteration through scheme
Other properties can be changed in this section such as the number of points on the load curve (N = 25 currently) and the material yield stress. Note there is no post yielding behavior defined here, the yield stress is only used through colors to show how close each member is to yielding.

'''

## Import modules
import numpy as np
from math import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path
import csv
from scipy import linalg as linalg
from scipy.optimize import curve_fit

from Indeterminacy import *
from Structures_Class import *

## Control file directory

cwd = "G:\\Shared drives\\Meza Research Group Shared Drive\\Group - Tensegrities\\Simulation Data\\"
# # cwd += 'TestData\\'
os.chdir(cwd)

## Initial Geometry and Mechanical Properties

plt.close('all')
dims = 3
#Show different plots
showShrink = 0 #Show before and after pyrolysis shape
showLoad = 0 #Plots structure load vs displacement
showStress = 1 #Plots member stress to structure displacement
showShape = 0 #Plots before and after loading shapes
showIter = 0 #shows dt in members upon iteration
calcStiff = 0 #calculates the stiffness of each structure
printResults = 0 #Prints loads and nodal deformations
N = 25 #Number of points of evaluation
saveResult = 0 #Saves the load data to a csv file at location of current path

# Set up the parameters to run sweeps
sc = -6 # length scale (um) sc is a scaling factor for the structure so that structures can easily be made at the um, mm, or m sizes
Q = (600e-6)/3
H = 45
wid = 45
# P = [1.125,1.25,1.35,1.5,1.65,1.8,1.9,2.0]
P = [1.75]
E_ = 36e9
E_var = [36e9,56e9,76e9,96e9,150e9]
SLOPE = []
for p in P:
    Stiffness = [] #for plotting stiffnesses of sweeps
    NAME = []
    for E_ in E_var:

        T = [2.25, p]
        # Struct = hangNet();Q = Struct.Q;sc = Struct.sc

        # sc,Q,Struct = FraternaliPreset(preStrain = 0.072,Q = 80)

        Struct = icosahedron(H,H,T,E1 = E_, E2 = E_,shrink = 0.05,postConfine = True)

        # theta2 = OneDFormFind('simplex',H,wid)
        # Struct = StackedSimplex(H,wid,T,E1 = 36e9,E2 = 36e9,n = 3,m = 3)

        # Struct = simplex(H,wid,T,E1 = E_, E2 = E_,postConfine = True)

        # Struct = tetrakaidecahedron(H,wid,T,E1 = 36e9, E2 = 36e9,postConfine = True)

        ctrlE = False # Takes stiffness from specification or from class (false = take from class)
        LW = Struct.LW
        Height = str(int(H))
        Width = str(int(wid))
        TD = '225_'+str(int(T[1]*100))
        jobName = Struct.name+'_HW'+Height+'_'+Width+'_TD'+TD+'_E'+str(int(E_*10e-10))+'_ALGEBRAIC'

        ## Properties

        #Convert to arrays
        V = np.asarray(Struct.V)
        C = np.asarray(Struct.C)
        C = C.astype(int)
        T = np.asarray(Struct.T2)
        load = Struct.l
        if Struct.postConfine:
            confine = [0 if i == 0 else 1 for v in Struct.postConfine for i in v]
        else:
            confine = [0 if i == 0 else 1 for v in V[:,3:] for i in v]
        l2 = [0]*len(Struct.V)*3
        for indx in load: l2[indx*3+2] = -1
        l2 = [[ld] for i,ld in enumerate(l2) if confine[i]!= 1]
        l2 = np.asarray(l2)
        e_trans = np.asarray([[Struct.e_trans[0] if c[0] == 1 else Struct.e_trans[1]] for c in Struct.C])


        #initial properties
        confine = [0 if i == 0 else 1 for v in V[:,3:] for i in v]
        l = [0]*len(Struct.V)*3
        for indx in load: l[indx*3+2] = -1
        l = [[ld] for i,ld in enumerate(l) if confine[i]!= 1]
        l = np.asarray(l)
        e_per = np.asarray([[Struct.e_[0] if c[0] == 1 else Struct.e_[1]] for c in Struct.C])
        T = np.asarray([[Struct.T2[0] if c[0] == 1 else Struct.T2[1]] for c in Struct.C])

        #post shrinkage properties
        T_post = np.asarray([[Struct.T2[0] if c[0] == 1 else Struct.T2[1]] for c in Struct.C])
        if Struct.postConfine:
            K0 = np.asarray(Struct.postConfine)
        else:
            K0 = V[:,3:]
        #
        V = np.squeeze(V)

        C = np.reshape(C,(len(C),3))
        T = np.reshape(T,(len(T),1))
        T_post = np.reshape(T_post,(len(T_post),1))
        e_per = np.reshape(e_per,(len(e_per),1))

        ## Set up
        szUnits = {-6:'um',-3:'mm',0:'m',3:'km'}
        fcUnits = {-6:'uN',-3:'mN',0:'N',3:'kN'}

        #stiffness (Pa)
        if ctrlE:
            E1 = E
            E_post = E
        else:
            E_post = Struct.E2
            E = Struct.E1

        #Yield stress (MPa)
        YD = 2000

        #Geometric properties
        #Convert nodes to correct length scale
        V[:,:3]= V[:,:3]*(10**(sc))
        V = np.asarray(V)

        n = len(T) #number of members

        CS = pi*(T_post*10**(sc))**2 # meters sq--Cross sectional area
        Len = [np.linalg.norm((V[C[i,1],0:3]-V[C[i,2],0:3])) for i in range(n)]
        # L1 = Len[0],Len[3],Len[6]
        e_ = [[Len[i]*e_per[i][0]]for i in range(len(e_per))] #Convert percent shrinkage to element elongations in um
        e_ = np.asarray(e_)

        #Calculation of the member axial flexibilities
        F = np.diag(np.diag(Len/CS/E)) # Diagonal matrix (L/EA)

        #Create the Equilibrium matrix
        #Gives Inextensional mechanisms, D and number of mechanisms, m
        A,D,m = EQTR3(V,C);
        Ainv = linalg.pinv(A)
        [h,w] = np.shape(A) #Find dimensions of A
        B = A.T #Calculate compatability matrix
        Binv = linalg.pinv(B)
        k = sum(sum(V[:,3:])) #number of constraints

        s = np.shape(linalg.null_space(A))[1] #Find number of self stressed states in A

        # Type 1 assemblies: s = 0, m = 0
        if not s and not m:
            Type = 1

        # Type 2 assemblies
        if not s and m:
            Type = 2

        # Type 3 assemblies
        if s and not m:
            Type = 3
            # if abs(np.linalg.norm(SS)

        # Type 4 assemblies: s>1 and m>1
        if s and m:
            Type = 4


        # # #############################################
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # # ax = plt.axes()
        # S = [np.concatenate([V[C[i,1],range(3)],V[C[i,2],range(3)]]) for i in range(len(C))]


        #     for i in range(len(S)):
        #     ax.plot3D([S[i][0],S[i][3]],[S[i][1],S[i][4]],[S[i][2],S[i][5]],'b',linewidth=(T[i]*LW))
        # ax.grid(False)
        # ax.axis('off')

    #   plt.show()
        # #############################################

        ## Initial State
        tp = np.zeros((len(e_),1)) #No initial load
        t = staticIndetermine(A,e_,F,tp) #Tensions from self stress
        edot = e_ + np.dot(F,t)
        di,dii,G,_,_ = kinematicIndetermine(A,V,C,Len,t,edot,l*0) #Displacements from mechanisms from tensions and elongations
        d = di+dii #total displacement from load and elongations (di is compatible elongations, dii is from mechanisms)

        Vd = np.zeros(np.shape(V[:,0:3]))
        count = 0
        #Apply displacements to original nodal coordinates
        for j,v in enumerate(V):
            for i in range(3):
                if v[i+3]!=1:
                    Vd[j,i] = V[j,i]+d[count]
                    count+=1
                else:
                    Vd[j,i] = V[j,i]


        W = np.concatenate((Vd,K0),axis = 1) #Update the vertex positions
        l = l2

        #For some reason the tetrakai structure loses its self-stress state following the prestress step
        if Struct.name != 'tetkai' and Struct.name != 'StackedSimplex':
            A,D,m = EQTR3(W,C)
        else:
            W = V

        #Update structure properties
        CS = pi*(T_post*10**(sc))**2 # meters sq--Cross sectional area
        E = E_post
        Len = [np.linalg.norm((W[C[i,1],0:3]-W[C[i,2],0:3])) for i in range(n)]

        #Calculation of the member axial flexibilities
        F = np.diag(np.diag(Len/(CS*E))) # Diagonal matrix (L/EA)

        #determine prestress magnitude
        #Initial stress
        stress = t/CS*10**-6

        ## Additional load
        #set prestress
        t0 = t[:]
        #Loads
        l = l*Q #Scale the loads on the structure
        SS = linalg.null_space(A) #Self stress states

        ep = e_*0 #For zero elongation
        dl = l/N
        stresses = np.zeros((N+1,len(t))) #Track progression of stresses in members for plotting
        load = np.zeros((N+1,1)) #track applied load
        strains = np.zeros((N+1,1)) #Track progression of global strain for plotting (correspond to the height)
        elong= np.zeros((N+1,1)) #Track progression of global compression for plotting (correspond to the height)
        t = t*np.ones((len(t),N+1)) #Track progression of member tensions
        dummy = [] #used for plotting the change in t for each cycle through the interative scheme
        if Type == 4: #Only use iteration on s>1 and m>1 structures
            if showIter:
                fig = plt.figure()
                ax = plt.axes()

        #Iteration Scheme
        d = np.zeros((len(D),N+1))
        dt = 0
        for i in range(1,N+1):
            load[i] = sum(dl*i) #-Q/N*i #Assumes loads are all in the same direction
            tp = t0[:]
            for ii in range(40):
                _,_,_,_,dtr = kinematicIndetermine(A,W,C,Len,tp,ep,dl*(i)) #Change in member tensions from applied load
                dts = staticIndetermine(A,ep,F,dtr) #Change in member tensions from increase in self stress
                dt=dts+dtr #total change in member tensions

                if Type != 4: tp = t0+dt; break #Only iterate for type 4 structures

                close = abs(np.linalg.norm(t0+dt)/np.linalg.norm(tp)-1)
                dummy.append(np.linalg.norm(dt))
                if close<0.0001: #Within .01% (convergence)
                    # print('go \n'+str(np.linalg.norm(dt)/np.linalg.norm(tp)))
                    break
                else:
                    # print('loop\n'+str(close))
                    tp = t0+dt

            t[:,i]=np.squeeze(t0+dt)[:]
            d_e = ep + np.dot(F,dt)
            di,dii,_,dlii,_ = kinematicIndetermine(A,W,C,Len,tp,d_e,dl*(i))

            if Type == 4:
                if showIter:
                    plt.plot(range(len(dummy)),dummy) #plots the norm of dt
                tn,diic = nonLinearCorrect(A,W,C,F,dii,tp,dlii)
                # t0 = t00+np.expand_dims(tn,axis = 1)
                t[:,i]+=tn #Apply correction to tensions
                d[:,i] = np.squeeze(di+diic)
            else:
                d[:,i] = np.squeeze(di+dii)

            count = 0
            dr = np.zeros((len(W),3))
            for j,v in enumerate(W):
                for k in range(3):
                    if v[k+3]!=1:
                        dr[j,k] = d[count,i]
                        count+=1
                    else:
                        dr[j,k] = 0
        #
            #Store stresses and strains
            stresses[i] = np.squeeze(np.asarray([(t[j,i]/CS[j]*10**-6) for j in range(len(CS))]))
            # strains[i] = [(np.amax(dr[:,2])-np.amin(dr[:,2]))/(np.amax(W[:,2])-np.amin(W[:,2]))] #Plots strain
            elong[i] = [((np.amax(dr[:,2])-np.amin(dr[:,2])))] #Plots vertical displacement

        #Plot the convergence of dt
        if Type == 4:
            if showIter:
                plt.xlabel('cycle')
                plt.ylabel('change in t')
                plt.show()

        ## Display Shrinkage Results

        # H_post = H*(1+min(e_per))*10**(sc)
        H_post = H*10**-6
        if showShrink:
            if Type == 4 or Type == 2: #Check that structure had initial deformation
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                # ax = plt.axes()
                S = [np.concatenate([V[C[i,1],range(3)],V[C[i,2],range(3)]]) for i in range(len(C))]
                Sp = [np.concatenate([Vd[C[i,1],range(3)],Vd[C[i,2],range(3)]]) for i in range(len(C))]

                for i in range(len(S)):
                    ax.plot3D([S[i][0],S[i][3]],[S[i][1],S[i][4]],[S[i][2],S[i][5]],'b',linewidth=(T[i]*LW))
                    # ax.plot3D([Sp[i][0],Sp[i][3]],[Sp[i][1],Sp[i][4]],[Sp[i][2],Sp[i][5]],'k',linewidth=(T_post[i]*LW))
                    ax.plot3D([Sp[i][0],Sp[i][3]],[Sp[i][1],Sp[i][4]],[Sp[i][2],Sp[i][5]],linewidth=(T[i]*LW),color=plt.cm.RdYlBu_r(np.squeeze(stress[i])/YD/2+0.5))
                # ax.scatter3D(V[[2,7,9],0],V[[2,7,9],1],V[[2,7,9],2],marker = 'o')
                ax.grid(False)
                #Set limis of axis to view scale structure
                ax.set_xlim3d((-H_post/2,H_post/2))
                ax.set_ylim3d((-H_post/2,H_post/2))
                ax.set_zlim3d((np.amin(W[:,2]),np.amin(W[:,2])+H_post))
                ax.axis('off')

                plt.show()

        ## Calculation of final geometry
        Wd = np.zeros(np.shape(W[:,0:3]))
        stresses[0,:] = np.squeeze(stress)
        count = 0
        count1 = 0
        disp = d[:,-1]
        tension = t
        #Apply displacements to original nodal coordinates
        for j,v in enumerate(W):
            for i in range(3):
                if v[i+3]!=1:
                    Wd[j,i] = W[j,i]+disp[count1]
                else:
                    Wd[j,i] = W[j,i]
                    disp = np.insert(disp,count1,0) #reformat d to include displacements of constrained nodes
                count1+=1
        d = disp*10**(-sc) #convert to proper length scale
        t = t[:,-1]*10**(-sc) #convert to proper length scale
        load = load*10**(-sc)
        elong = elong*10**(-sc)
        d = np.round(d,4)
        t = np.round(t,4)
        d = np.reshape(d,(int(len(d)/3),3))

        # load = load*10**-6
        # stress = stress*10**-6
        # stresses = stresses*10**-6

        if s:
            SS = linalg.null_space(A) #Self stress states
            # gamma = int(linalg.norm(t/SS))
        stresses = np.round(stresses,4)

        ## Display of Loading Results

        if printResults:
            print('The tensions in the members are:\n' + str(t)+' '+str(fcUnits[sc])+'\n') #the tensions in the bars in N
            print('The displacements of the free nodes are:\n'+ str(d)+' '+str(szUnits[sc])+'\n') #the displacements of the free nodes (x,y,z)

        #If there was a load applied to the structure, plot those results
        if sum(l)!=0 and dims !=2:
            # Plotting
            if showShape:
                fig = plt.figure()
                ax = plt.axes(projection='3d')

                #Make structure of before and after applied load in an nx6 matrix corresponding to the coordinates of each bar (xi,yi,zi,xj,yj,zj)
                S = [np.concatenate([W[C[i,1],range(3)],W[C[i,2],range(3)]]) for i in range(len(C))]
                Sp = [np.concatenate([Wd[C[i,1],range(3)],Wd[C[i,2],range(3)]]) for i in range(len(C))]

                #Plot before and after structure, with a color map representing the stress in the members, note the thickness of the members is NOT to scale, but simply represents the different sizes

                for i in range(len(S)):
                    ax.plot3D([Sp[i][0],Sp[i][3]],[Sp[i][1],Sp[i][4]],[Sp[i][2],Sp[i][5]],linewidth=(T[i]*LW),color=plt.cm.RdYlBu_r(stresses[-1,i]/YD/0.3+0.5))
                    # ax.plot3D([S[i][0],S[i][3]],[S[i][1],S[i][4]],[S[i][2],S[i][5]],'k',linewidth=(T[i]*LW), alpha = 0.25)

                ax.axis('off')
                #Set limis of axis to view scale structure
                ax.set_xlim3d((-H_post/2,H_post/2))
                ax.set_ylim3d((-H_post/2,H_post/2))
                ax.set_zlim3d((np.amin(W[:,2])-H_post/2,np.amin(W[:,2])+H_post/2))

                ax.grid(False)
                plt.show()

                fig, ax = plt.subplots(figsize=(1, 6))
                fig.subplots_adjust(right=0.3)

                cmap = plt.cm.RdYlBu_r
                normlz = mpl.colors.Normalize(vmin=-1, vmax=1)
                mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                            norm=normlz,
                                            orientation='vertical')
                plt.show()

            if showStress:
                #Find members to plot stresses for (only include members that are unique)
                member_stress = np.array([])
                unique = np.array([])
                for i in range(len(C)):
                    if stresses[-1,i] not in member_stress:
                        member_stress = np.append(member_stress,[stresses[-1,i]])
                        unique = np.append(unique,[i])


                fig, (ax1,ax2) = plt.subplots(2,1,sharex = True)
                ax1.plot(elong,-1*load)

                ax1.set_ylabel('Load '+str(fcUnits[sc]))

                ax1.set_title('Structure load response')
                ax1.grid(True)
                plt.show()

                #Plot all of the members with different end stresses
                iterate = 0
                for i in unique:
                    i = int(i)
                    ax2.plot(elong, stresses[:,i], color=plt.cm.YlOrRd(3*abs(member_stress[iterate]/np.linalg.norm(stresses[-1,:]))))
                    iterate+=1

                #Plot tensions vs loads
                # iterate = 0
                # for i in unique: #Plot all of the members with different stresses
                #     i = int(i)
                #     ax2.plot(-1*load,tension[i,:],color=plt.cm.YlOrRd(3*abs(member_stress[iterate]/np.linalg.norm(stresses[-1,:]))))
                #     iterate+=1

                #Plot stresses vs strains
                # ax2.plot(strains*10**2,stresses[:,3],color=plt.cm.YlOrRd(3*abs(stresses[-1,3]/np.linalg.norm(stresses[-1,:]))))
                # ax2.plot(strains*10**2,stresses[:,6],color=plt.cm.YlOrRd(3*abs(stresses[-1,6]/np.linalg.norm(stresses[-1,:]))))

                # ax2.set_xlabel('Load '+str(fcUnits[sc]))
                ax2.set_ylabel('Stress (MPa)')
                ax2.set_xlabel('Elongation '+str(szUnits[sc]))

                # ax2.set_xlabel('Applied Load')
                # ax2.legend(['Bars','Vertical Tendon','Horizontal Tendon'])
                ax2.set_title('Member stresses')
                ax2.grid(True)
                plt.show()

            if showLoad:
                fig = plt.figure()
                ax = plt.axes()
                # iterate = 0
                # for i in unique: #Plot all of the members with different stresses
                #     i = int(i)
                #     ax.plot(-1*load,tension[i,:],color=plt.cm.YlOrRd(3*abs(member_stress[iterate]/np.linalg.norm(stresses[-1,:]))))
                #     iterate+=1
                ax.plot(elong,-1*load,'r')
                ax.set_ylabel('Load '+str(fcUnits[sc]))
                ax.set_xlabel('Elongation '+str(szUnits[sc]))
                # ax.set_ylabel('Load (uN)')
                # ax.set_xlabel('Elongation (um)')
                # plt.xlim(0,50)
                # plt.ylim(-20,200)
                ax.grid(True)
                plt.show()


        #   for i in unique:
            #     i = int(i)
            #     ax.plot(-1*load*10**-3,stresses[:,i],'r')
            # ax.set_ylabel('Stress (MPa)') #+str(fcUnits[sc]))
            # ax.set_xlabel('Load (N)')#+str(fcUnits[sc]))
            # # plt.xlim(0,50)
            # # plt.ylim(-20,200)
            # ax.grid(True)
            # plt.show()

        if dims == 2:
            fig = plt.figure()
            ax = plt.axes()
            #Make structure of before and after applied load in an nx6 matrix corresponding to the coordinates of each bar (xi,yi,zi,xj,yj,zj)
            S = [np.concatenate([W[C[i,1],range(2)],W[C[i,2],range(2)]]) for i in range(len(C))]
            Sp = [np.concatenate([Wd[C[i,1],range(2)],Wd[C[i,2],range(2)]]) for i in range(len(C))]


            for i in range(len(S)):
                ax.plot([Sp[i][0],Sp[i][2]],[Sp[i][1],Sp[i][3]],linewidth=(T[i]*LW),color=plt.cm.YlOrRd(3*abs(stresses[-1,i]/np.linalg.norm(stresses[-1,:]))))
                ax.plot([S[i][0],S[i][2]],[S[i][1],S[i][3]],'k',linewidth=(T[i]*LW), alpha = 0.25)
                plt.xlim(0,0.5)
                plt.ylim(-0.25,0.25)
            ax.axis('off')
            plt.show()

            fig = plt.figure()
            ax = plt.axes()
            for i in range(len(e_)):
                ax.plot(-1*load*10**-3, stresses[:,i],'r')
            plt.xlabel('Load (N)')
            plt.ylabel('Stress (Mpa)')
            ax.grid(True)
            plt.show()

        if saveResult:
            with open(jobName+'.csv', 'w') as file:
                jobCSV = csv.writer(file)
                jobCSV.writerow(['Frame','Displacement','Load','PrestressB','PrestressT'])
                jobCSV.writerows([[i,-1*float(elong[i]),-1*float(load[i]),float(min(stress*10**6)),float(max(stress*10**6))] for i in range(N)])
            print('Results Saved')
        # After leaving the above block of code, the file is closed

    ##  Calculate Stiffness
        if calcStiff:

            # fig = plt.figure() #new figure
            # ax = plt.axes()

            disp = np.asarray([float(el) for el in elong])
            load = np.asarray([-1*float(lo) for lo in load])
            start = 0
            end = len(disp)
            preStress = [round(float(min(stress))), round(float(max(stress)))]
            #Structure properties depend on the size of the structure given in the name
            #This needed to get the stiffness
            xplot = np.arange(100,320)

            #specify bounds of the line fit
            i = np.where(disp>0.40)[0][0];j=end
            def stiff(x,m,b):
                return m*x+b #Linear stiffness

            #Fit line to the data within the bounds
            xdata = disp[i:j]; ydata = load[i:j]
            # xdata = strain[i:j]; ydata = stress[i:j]
            popt, pcov = curve_fit(stiff, xdata, ydata)

            #Calculate the stiffness
            # E = (popt[0]*10**3)*H/A*10**6 #N/m^2 (convert uN/um to N/m to N/m^2)
            # E = int(E*10**-6) #MPa

            #Calculate load vs displacement slope
            E = round(popt[0],2) #N/m

            Stiffness += [E]
            name = jobName


            # ax.plot(disp[i:],load[i:], label = name+'\nE: '+ str(E) +' N/m, ')# +'PS: '+str(preStress) +'MPa')
            # plt.title('Icosahedron 45 Analytical Load Response')
            # ax.set_xlabel('Displacement (um)', fontsize = 11)
            # ax.set_ylabel('Load (uN)',fontsize = 11)

       #      #Print results for each data series and plot the data along with the stiffness results
            # print(jobName+' stiffness')
            # print(str(E)+' N/m\n'+'Prestress: '+str(preStress)+' MPa\n')
            # ax.set_xlim([0,max(disp)+1])
            # plt.legend(['Analytical','Numerical'],loc = "upper right")
            # plt.legend(loc = "upper left",fontsize=11)


            # plt.savefig(name+'.png', bbox_inches='tight')

    if calcStiff:
        # plt.show()
        # fig = plt.figure() #new figure
        # ax = plt.axes()
        E = [E*10**-9 for E in E_var]
        popt, pcov = curve_fit(stiff, E, Stiffness)
        Slope = round(popt[0],2)
        SLOPE += [Slope]
        print('2.25_'+str(p)+': '+ str(Slope)+' (N/m)/GPa')
        # ax.plot(E,Stiffness)
        # plt.title(name + ' Stiffness')
        # ax.set_xlabel('Material Stiffness (GPa)', fontsize = 11)
        # ax.set_ylabel('Structure Stiffness (N/m)',fontsize = 11)
        # plt.show()

if calcStiff:
    fig = plt.figure() #new figure
    ax = plt.axes()
    ax.plot(P,SLOPE)
    plt.title('Change in Structure Stiffness With Material Stiffness')
    ax.set_xlabel('Tendon Diameter (um)', fontsize = 11)
    ax.set_ylabel('Structure Stiffness to Material Stiffness ((N/m)/GPa)',fontsize = 10)
    plt.show()