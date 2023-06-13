import numpy as num
import matplotlib.pyplot as plt

#Constants
class FEM:
    def __init__(self):
        self.ax, self.ay, self.az, self.nx, self.ny, self.nz = self.inputData()
        self.initData()
        print('Processing!')
        self.generateMesh()
        self.generateDeformation()
                
    def inputData(self):
        print('To continue please input requred data:')
        print('Input ax')
        ax = int(input())
        print('Input ay')
        ay = int(input())
        print('Input az')
        az = int(input())
        print('Input nx')
        nx = int(input())
        print('Input ny')
        ny = int(input())
        print('Input nz')
        nz = int(input())
        print('This is the received data : ax, ay, az = ' + str(ax) +', '+ str(ay) +', '+ str(az) + ' and nx, ny, nz = '+ str(nx) +', '+ str(ny) +', '+ str(nz))
        return ax, ay, az, nx, ny, nz 
    
    def initData(self):
        self.nqp = 4 * self.nx * self.ny * self.nz + 3 * self.ny * self.nz + 3 * self.nx * self.nz + 2 * self.nz + 3 * self.nx * self.ny + 2 * self.ny + 2 * self.nx + 1
        self.nel = self.nx * self.ny * self.nz
        self.xStep, self.yStep, self.zStep = self.ax/self.nx, self.ay/self.ny, self.az/self.nz
        self.xHalf, self.yHalf, self.zHalf = self.xStep/2, self.yStep/2, self.zStep/2

        self.AKT = num.zeros((self.nqp, 3))
        self.NT = num.zeros((self.nel, 20))

        self.ZP = num.zeros((self.nx * self.ny, 2)) #pressure array
        self.ZU = num.zeros((self.nx * self.ny, 2)) #fixing array

        self.DFIABG = num.zeros((27, 3, 20)) #Gauss cude
        self.DPSITE = num.zeros((9, 2, 8)) #Gauss vektor

        self.MG = num.zeros((3 * self.nqp, 3 * self.nqp)) #K
        self.F = num.zeros(3 * self.nqp)

        self.U = num.zeros((3 * self.nqp))

        self.Pn = 50
        self.E = 200 #модуль Юнга
        self.nu = 0.31 #коефіцієнт Пуасона
        self.lambda_ = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu)) #коефіцієнт Ляме
        
        self.faces = {
            1: [0, 1, 4, 5, 8, 12, 13, 16],
            2: [1, 2, 5, 6, 9, 13, 14, 17],
            3: [2, 3, 6, 7, 10, 14, 15, 18],
            4: [0, 3, 4, 7, 11, 12, 15, 19],
            5: [0, 1, 2, 3, 8, 9, 10, 11],
            6: [4, 5, 6, 7, 16, 17, 18, 19]
        }
        
    def generateMesh(self):
        self.getAKT()
        print(self.AKT)
        self.getNT()
        print(self.NT)
        self.getZU()
        self.getZP()
        self.rgb = 'blue'
        self.generateGraphView(self.AKT, self.rgb)
    
    def getAKT(self):
        iter_X, iter_Y, iter_Z = 0,0,0
        y, z = 0, 0
        iterator = 0
        while iterator != self.nqp:
            if iter_X <= self.ax:
                if z % 2 == 0:
                    if y % 2 == 0:
                        self.AKT[iterator,0], self.AKT[iterator,1], self.AKT[iterator,2] = iter_X, iter_Y, iter_Z
                        iter_X += self.xHalf
                    else:
                        self.AKT[iterator,0], self.AKT[iterator,1], self.AKT[iterator,2] = iter_X, iter_Y, iter_Z
                        iter_X += self.xStep
                else:
                    self.AKT[iterator,0], self.AKT[iterator,1], self.AKT[iterator,2] = iter_X, iter_Y, iter_Z
                    iter_X += self.xStep
                    if iter_X > self.ax:
                        iter_X = 0
                        iter_Y += self.yStep
                        if iter_Y > self.ay:
                            z+=1
                            y+=1
                            iter_Y = 0
                            iter_Z += self.zHalf
                iterator +=1
            else:
                iter_X = 0
                y+=1
                if iter_Y < self.ay:
                    iter_Y += self.yHalf
                elif iter_Z <= self.az:
                    z+=1
                    iter_Y = 0
                    iter_Z += self.zHalf
                else:
                        print('Z is out')

        return self.AKT
    
    def getNT(self):
        x, y, z = 0,0,0
        modAKT = num.array([list(round(val, 5) for val in row) for row in self.AKT])
        for i in range(self.nel):
            try:
                # corner items of the floor
                self.NT[i, 0] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                x+=self.xStep
                self.NT[i, 1] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                y+=self.yStep
                self.NT[i, 2] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                x -= self.xStep
                self.NT[i, 3] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]

                # return to start position
                y -= self.yStep

                #corner items of ceiling
                z += self.zStep
                self.NT[i, 4] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                x+=self.xStep
                self.NT[i, 5] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                y+=self.yStep
                self.NT[i, 6] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                x -= self.xStep
                self.NT[i, 7] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]

                # return to start position
                y -= self.yStep
                z -= self.zStep

                # half items of the floor
                x += self.xHalf
                self.NT[i, 8] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                x += self.xHalf
                y += self.yHalf
                self.NT[i, 9] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                y += self.yHalf
                x -= self.xHalf
                self.NT[i, 10] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0] 
                x-= self.xHalf
                y -= self.yHalf
                self.NT[i, 11] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]

                # return to start position
                y -= self.yHalf

                # middle of the faces
                z += self.zHalf
                self.NT[i, 12] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0] 
                x+=self.xStep
                self.NT[i, 13] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0] 
                y+=self.yStep
                self.NT[i, 14] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                x -= self.xStep
                self.NT[i, 15] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]

                # return to start position
                y -= self.yStep
                z -= self.zHalf

                # half items of the ceiling
                z += self.zStep
                x += self.xHalf
                self.NT[i, 16] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                x += self.xHalf
                y += self.yHalf
                self.NT[i, 17] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0] 
                y += self.yHalf
                x -= self.xHalf
                self.NT[i, 18] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]
                x-= self.xHalf
                y -= self.yHalf
                self.NT[i, 19] = num.where((num.round(modAKT, 5) == num.round(num.array([x,y,z]), 5)).all(axis=1))[0][0]

                #return to start position
                y -= self.yHalf
                z -= self.zStep
            except IndexError:
                print(x, y, z, i)
            else:
                if(num.round(x + self.xStep,5) >= self.ax):
                    if(num.round(y + self.yStep, 5) >= self.ay):
                        if(num.round(z + self.zStep,5) >= self.az):
                            break
                        else:
                            x = 0
                            y = 0
                            z += self.zStep
                    else:
                        x = 0
                        y += self.yStep
                else:
                    x += self.xStep

        self.NT = self.NT.astype(int)
        
    def getZU(self):
        for i in range(self.ZU.shape[0]):
            self.ZU[i] = num.array([i, 5])

        return self.ZU
    
    #ceiling array
    def getZP(self):
        j = self.nx * self.ny * (self.nz - 1) + 1
        for i in range(self.ZP.shape[0]):
            self.ZP[i] = num.array([j - 1, 5])
            j+=1

        return self.ZP
    
    def getAKT_NT(self, akt):
        matrix = [[[0, 0, 0] for _ in range(20)] for _ in range(self.nel)]
        for j in range(self.nel):
            for i in range(20):
                index = self.NT[j][i]
                if index < akt.shape[0]:
                    matrix[j][i] = akt[index]
        
        return matrix
    
    def generateDeformation(self):
        self.getMG()
        self.getF()
        self.calculateEquations()
        modifiedAKT = self.AKT + self.U
        self.generateGraphView(modifiedAKT, self.rgb)
    
    def getMG(self):

        self.getDFIABG()
        matrix = self.getAKT_NT(self.AKT)
        
        c = [5/9, 8/9, 5/9]
        for se, nt in zip(matrix,self.NT):
            MGE = num.zeros((60,60))
            func = self.getDFIXYZ(se)
            for i in range(20):
                for j in range(20):
                    iter_guas = 0
                    for m in c:
                        for n in c:
                            for k in c:
                                dfixyz = func[0][iter_guas]
                                det = func[1][iter_guas]
                                a11 = (self.lambda_ * (1 - self.nu) * (dfixyz[i,0] * dfixyz[j,0]) + self.mu *((dfixyz[i,1] * dfixyz[j,1])+(dfixyz[i,2]*dfixyz[j,2])))
                                MGE[i,j] += m * n * k * a11 * det

                                a12 = self.lambda_ * self.nu * (dfixyz[i,0] * dfixyz[j,1]) + self.mu * (dfixyz[i,1] * dfixyz[j,0])
                                MGE[i,j+20] += m * n * k * a12 * det
                                # MGE[i+20,j] += m * n * k * a12 * det

                                a13 = self.lambda_ * self.nu * (dfixyz[i,0] * dfixyz[j,2]) + self.mu * (dfixyz[i,2] * dfixyz[j,0])
                                MGE[i,j+40] += m * n * k * a13 * det
                                # MGE[i+40,j] += m * n * k * a13 * det

                                a22 = (self.lambda_ * (1 - self.nu) * (dfixyz[i,1] * dfixyz[j,1]) + self.mu *((dfixyz[i,0] * dfixyz[j,0])+(dfixyz[i,2]*dfixyz[j,2])))
                                MGE[20+i,20+j] += m * n * k * a22 * det

                                a23 = self.lambda_ * self.nu * (dfixyz[i,1] * dfixyz[j,2]) + self.mu * (dfixyz[i,2] * dfixyz[j,1])
                                MGE[20+i,40+j] += m * n * k * a23 * det
                                # MGE[40+i,20+j] += m * n * k * a23 * det

                                a33 = (self.lambda_  * (1 - self.nu) * (dfixyz[i,2] * dfixyz[j,2]) + self.mu *((dfixyz[i,0] * dfixyz[j,0])+(dfixyz[i,1]*dfixyz[j,1]))) 
                                MGE[40+i,40+j] += m * n * k * a33 * det

                                iter_guas+=1
            
            # num.savetxt("mge{}.csv".format(iter), MGE, delimiter=',')

            MGE = self.getMirroredMatrix(MGE)

            # create MG from MGE
            for i in range(60):
                row = self.getIndex(nt, i)
                for j in range(60):
                    column = self.getIndex(nt,j)

                    self.MG[row, column] += MGE[i,j]

        # pin ZU points in MG
        for i, zu in enumerate(self.ZU):
            number, face = int(zu[0]), int(zu[1])
            face_points = self.faces.get(face)
            NTpoints = self.NT[number, face_points].astype(int)

            for point in NTpoints:
                point *= 3
                row, col = point, point
                for i in range(3):
                    self.MG[row, col] = num.power(10,54)
                    row +=1
                    col +=1

        return self.MG   
    
    #Derivatives of approximating functions in local coordinates
    def getDFIABG(self):
        AiBiGi = num.array([[-1, 1, 1, -1, -1, 1, 1, -1, 0, 1, 0, -1, -1, 1, 1, -1, 0, 1, 0, -1],
                            [1, 1, -1, -1, 1, 1, -1, -1, 1, 0, -1, 0, 1, 1, -1, -1, 1, 0, -1, 0],
                            [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]])
        
        AiBiGi = num.array([list(round(val) for val in row) for row in zip(*AiBiGi)])
        print(AiBiGi)

        ABG = {-num.sqrt(0.6), 0, num.sqrt(0.6)}
        counter = 0
        for g in ABG:
            for b in ABG:
                for a in ABG:
                    for i in range(0,8):
                        a_i = AiBiGi[i,0]
                        b_i = AiBiGi[i,1]
                        g_i = AiBiGi[i,2]
                        dAlpha = 0.125 * (1 + b*b_i)*(1 + g*g_i)*(a_i*(2 * a * a_i + b * b_i + g * g_i - 1))
                        dBeta = 0.125 * (1 + a*a_i)*(1 + g*g_i)*(b_i*(a * a_i + 2 * b * b_i + g * g_i - 1))
                        dGamma = 0.125 * (1 + b*b_i)*(1 + a*a_i)*(g_i*(a * a_i + b * b_i + 2 * g * g_i - 1))
                        self.DFIABG[counter,0,i] = dAlpha
                        self.DFIABG[counter,1,i] = dBeta
                        self.DFIABG[counter,2,i] = dGamma
                    for i in range(8,20):
                        a_i = AiBiGi[i,0]
                        b_i = AiBiGi[i,1]
                        g_i = AiBiGi[i,2]
                        dAlpha = 0.25 * (1 + b*b_i)*(1 + g*g_i)*( -a_i ** 3 * b_i ** 2 * g ** 2 - a_i ** 3 * b ** 2 * g_i ** 2 - 3* a**2 * a_i * b_i ** 2 * g_i ** 2 + a_i - 2 * a * b_i ** 2 * g_i ** 2)
                        dBeta = 0.25 * (1 + a*a_i)*(1 + g*g_i)*( -a_i ** 2 * b_i ** 3 * g ** 2 - a ** 2 * b_i ** 3 * g_i ** 2 - 3 * a_i**2 * b ** 2 * b_i * g_i ** 2 + b_i - 2 * a_i ** 2 * b * g_i ** 2)
                        dGamma = 0.25 * (1 + b*b_i)*(1 + a*a_i)*( -a_i ** 2 * b ** 2 * g_i ** 3 - a ** 2 * b_i ** 2 * g_i ** 3 - 3 * a_i**2 * b_i ** 2 * g**2 * g_i + g_i - 2 * a_i**2 * b_i**2 * g)
                        self.DFIABG[counter,0,i] = dAlpha
                        self.DFIABG[counter,1,i] = dBeta
                        self.DFIABG[counter,2,i] = dGamma
                    counter +=1
        print(self.DFIABG)
        return self.DFIABG
    
    #Jacobian of the transformation from local coordinate system to global coordinate system
    def getDFIXYZ(self,se):
        DJ = num.zeros(27)
        Jacobean = num.zeros((27, 3, 3))
        DFIXYZ = num.zeros((27,20,3))
        se = num.array(se)

        for j, GaussNode in enumerate(self.DFIABG):
            for i,node in enumerate(se):
                #dx,dy,dz /dalpha
                Jacobean[j,0,0] += node[0] * GaussNode[0, i]
                Jacobean[j,0,1] += node[1] * GaussNode[0, i]
                Jacobean[j,0,2] += node[2] * GaussNode[0, i]
                
                #dx,dy,dz /dbetta
                Jacobean[j,1,0] += node[0] * GaussNode[1, i]
                Jacobean[j,1,1] += node[1] * GaussNode[1, i]
                Jacobean[j,1,2] += node[2] * GaussNode[1, i]

                #dx,dy,dz /dgama
                Jacobean[j,2,0] += node[0] * GaussNode[2, i]
                Jacobean[j,2,1] += node[1] * GaussNode[2, i]
                Jacobean[j,2,2] += node[2] * GaussNode[2, i]
            
            for i in range(20):
                DFIXYZ[j,i] = num.linalg.solve(Jacobean[j], [GaussNode[0,i], GaussNode[1,i], GaussNode[2,i]])

            DJ[j] = (Jacobean[j,0,0] * Jacobean[j,1,1] * Jacobean[j,2,2] +
            Jacobean[j,1,0] * Jacobean[j,2,1] * Jacobean[j,0,2] +
            Jacobean[j,2,0] * Jacobean[j,0,1] * Jacobean[j,1,2] -
            Jacobean[j,2,0] * Jacobean[j,1,1] * Jacobean[j,0,2] -
            Jacobean[j,1,0] * Jacobean[j,0,1] * Jacobean[j,2,2] -
            Jacobean[j,0,0] * Jacobean[j,2,1] * Jacobean[j,1,2])

        return DFIXYZ, DJ
    
    def getMirroredMatrix(self, MGE):
        # Get the submatrices
        a11 = MGE[:20, :20]
        a12 = MGE[:20, 20:40]
        a13 = MGE[:20, 40:]

        a21 = MGE[20:40, :20]
        a22 = MGE[20:40, 20:40]
        a23 = MGE[20:40, 40:]

        a31 = MGE[40:, :20]
        a32 = MGE[40:, 20:40]
        a33 = MGE[40:, 40:]
        
        # Mirror the specified submatrices
        a21_mirror = num.transpose(a12)
        a31_mirror = num.transpose(a13)
        a32_mirror = num.transpose(a23)
        
        # Create a new matrix with the mirrored submatrices
        new_matrix = num.zeros((60,60))

        new_matrix[:20, :20] = a11
        new_matrix[:20, 20:40] = a12
        new_matrix[:20, 40:] = a13

        new_matrix[20:40, :20] = a21_mirror
        new_matrix[20:40, 20:40] = a22
        new_matrix[20:40, 40:60] = a23

        new_matrix[40:, :20] = a31_mirror
        new_matrix[40:, 20:40] = a32_mirror
        new_matrix[40:, 40:] = a33
        
        return new_matrix
    
    def getIndex(self, nt, index):
        return 3 * nt[index%20] + (index//20)
    
    def getF(self):
        self.getDPSITE()
        self.getZP()
        EtaiTaui = num.array([
            [-1, 1, 1, -1, 0, 1, 0, -1],
            [-1, -1, 1, 1, -1, 0, 1, 0]])
        modEtaiTaui = num.array([list(round(val) for val in row) for row in zip(*EtaiTaui)])

        EtaTau = {-num.sqrt(0.6), 0, num.sqrt(0.6)}
        c = [5/9, 8/9, 5/9]

        for i,zp in enumerate(self.ZP):
            FE = num.zeros(60)
            XYZ_ZP = num.zeros((8,3))

            # get coordinates of loaded points from zp
            number, face = int(zp[0]), int(zp[1])
            face_points = self.faces.get(face)
            NTpoints = self.NT[number, face_points].astype(int)
            XYZ_ZP = [self.AKT[i] for i in NTpoints]

            DPSIXYZ = self.getDPSIXYZ(XYZ_ZP)
            points_iter = 0
            for k in range(20):
                if k in face_points:
                    counter = 0
                    for m,tau in zip(c,EtaTau):
                        for n,eta in zip(c, EtaTau):
                            eta_i = modEtaiTaui[points_iter,0]
                            tau_i = modEtaiTaui[points_iter,1]
                            if points_iter in range(4):
                                phi_k = 0.25 * (1 + eta * eta_i)* (1 + tau * tau_i) * (eta * eta_i + tau * tau_i - 1)
                            if (points_iter == 4 )or (points_iter == 6):
                                phi_k = 0.5 * (1 - num.power(eta,2)) * (1 + tau * tau_i)
                            if (points_iter == 5 )or (points_iter == 7):
                                phi_k = 0.5 * (1 - num.power(tau,2)) * (1 + eta * eta_i) 
                            
                            FE[k] += m * n * self.Pn * (DPSIXYZ[counter,1,0] * DPSIXYZ[counter,2,1] - DPSIXYZ[counter,2,0] * DPSIXYZ[counter,1,1]) * phi_k
                            FE[20 + k] += m * n * self.Pn * (DPSIXYZ[counter,2,0] * DPSIXYZ[counter,0,1] - DPSIXYZ[counter,0,0] * DPSIXYZ[counter,2,1]) * phi_k
                            FE[40 + k] += m * n * self.Pn * (DPSIXYZ[counter,0,0] * DPSIXYZ[counter,1,1] - DPSIXYZ[counter,1,0] * DPSIXYZ[counter,0,1]) * phi_k
                            counter +=1
                    points_iter += 1

            for item in range(60):
                row = self.getIndex(self.NT[number], item)
                self.F[row] += FE[item]
        
        return self.F
    
    def getDPSITE(self):
        EtaiTaui = num.array([
            [-1, 1, 1, -1, 0, 1, 0, -1],
            [-1, -1, 1, 1, -1, 0, 1, 0]])
        modEtaiTaui = num.array([list(round(val) for val in row) for row in zip(*EtaiTaui)])

        EtaTau = {-num.sqrt(0.6), 0, num.sqrt(0.6)}
        counter = 0
        for tau in EtaTau:
            for eta in EtaTau:
                for i in range(8):
                    eta_i = modEtaiTaui[i,0]
                    tau_i = modEtaiTaui[i,1]
                    if (i == 4) or (i == 6):
                        dEta = - eta * (tau * tau_i + 1)
                        dTau = 0.5 * (tau_i * (1 - eta**2))
                        self.DPSITE[counter,0,i] = dEta
                        self.DPSITE[counter,1,i] = dTau
                        continue
                    if (i == 5) or (i == 7):
                        dEta = 0.5 * (eta_i * (1 - tau**2))
                        dTau = - tau * (eta * eta_i + 1)
                        self.DPSITE[counter,0,i] = dEta
                        self.DPSITE[counter,1,i] = dTau
                        continue
                    dEta = 0.25 * (1 + tau * tau_i) * (2 * eta * num.power(eta_i,2) + eta_i * tau * tau_i)
                    dTau = 0.25 * (1 + eta * eta_i) * (2 * tau * num.power(tau_i,2) + eta * eta_i * tau_i)
                    self.DPSITE[counter,0,i] = dEta
                    self.DPSITE[counter,1,i] = dTau
                counter +=1
                    
        return self.DPSITE
    
    def getDPSIXYZ(self, XYZ_ZP):
        DPSIXYZ = num.zeros((9,3,2))

        for j, GaussNode in enumerate(self.DPSITE):
            for i,node in enumerate(XYZ_ZP):
                DPSIXYZ[j,0,0] += node[0] * GaussNode[0,i] #dx/dE
                DPSIXYZ[j,0,1] += node[0] * GaussNode[1,i] #dx/dT
                
                DPSIXYZ[j,1,0] += node[1] * GaussNode[0,i] #dy/dE
                DPSIXYZ[j,1,1] += node[1] * GaussNode[1,i] #dy/dT

                DPSIXYZ[j,2,0] += node[2] * GaussNode[0,i] #dz/dE
                DPSIXYZ[j,2,1] += node[2] * GaussNode[1,i] #dx/dT
                    
        return DPSIXYZ
    
    def calculateEquations(self):
        self.U = num.linalg.solve(self.MG, self.F)

        self.U = self.U.reshape(-1, 3)
    
    def generateGraphView(self, AKT, colors):

        matrix = self.getAKT_NT(AKT)
            
        #Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #Plot the coordinates as points
        ax.scatter(self.AKT[:,0], self.AKT[:,1], self.AKT[:,2], c=colors)

        line_color = 'black'
        # Connect the coordinates with lines
        for row in matrix:
            ax.plot([row[8][0], row[0][0]], [row[8][1], row[0][1]], [row[8][2], row[0][2]], color=line_color)
            ax.plot([row[1][0], row[8][0]], [row[1][1], row[8][1]], [row[1][2], row[8][2]], color=line_color)
            ax.plot([row[9][0], row[1][0]], [row[9][1], row[1][1]], [row[9][2], row[1][2]], color=line_color)
            ax.plot([row[2][0], row[9][0]], [row[2][1], row[9][1]], [row[2][2], row[9][2]], color=line_color)
            ax.plot([row[3][0], row[2][0]], [row[3][1], row[2][1]], [row[3][2], row[2][2]], color=line_color)
            ax.plot([row[11][0], row[3][0]], [row[11][1], row[3][1]], [row[11][2], row[3][2]], color=line_color)
            ax.plot([row[0][0], row[11][0]], [row[0][1], row[11][1]], [row[0][2], row[11][2]], color=line_color)

            ax.plot([row[12][0], row[0][0]], [row[12][1], row[0][1]], [row[12][2], row[0][2]], color=line_color)
            ax.plot([row[4][0], row[12][0]], [row[4][1], row[12][1]], [row[4][2], row[12][2]], color=line_color)

            ax.plot([row[13][0], row[1][0]], [row[13][1], row[1][1]], [row[13][2], row[1][2]], color=line_color)
            ax.plot([row[5][0], row[13][0]], [row[5][1], row[13][1]], [row[5][2], row[13][2]], color=line_color)

            ax.plot([row[14][0], row[2][0]], [row[14][1], row[2][1]], [row[14][2], row[2][2]], color=line_color)
            ax.plot([row[6][0], row[14][0]], [row[6][1], row[14][1]], [row[6][2], row[14][2]], color=line_color)

            ax.plot([row[15][0], row[3][0]], [row[15][1], row[3][1]], [row[15][2], row[3][2]], color=line_color)
            ax.plot([row[7][0], row[15][0]], [row[7][1], row[15][1]], [row[7][2], row[15][2]], color=line_color)

            ax.plot([row[16][0], row[4][0]], [row[16][1], row[4][1]], [row[16][2], row[4][2]], color=line_color)
            ax.plot([row[5][0], row[16][0]], [row[5][1], row[16][1]], [row[5][2], row[16][2]], color=line_color)
            ax.plot([row[17][0], row[5][0]], [row[17][1], row[5][1]], [row[17][2], row[5][2]], color=line_color)
            ax.plot([row[6][0], row[17][0]], [row[6][1], row[17][1]], [row[6][2], row[17][2]], color=line_color)
            ax.plot([row[18][0], row[6][0]], [row[18][1], row[6][1]], [row[18][2], row[6][2]], color=line_color)
            ax.plot([row[7][0], row[18][0]], [row[7][1], row[18][1]], [row[7][2], row[18][2]], color=line_color)
            ax.plot([row[19][0], row[7][0]], [row[19][1], row[7][1]], [row[19][2], row[7][2]], color=line_color)
            ax.plot([row[4][0], row[19][0]], [row[4][1], row[19][1]], [row[4][2], row[19][2]], color=line_color)

        # Set the axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

fem = FEM()