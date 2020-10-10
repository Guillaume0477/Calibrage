import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


img = cv2.imread('capture_mire_0.png',0)
img2 = cv2.imread('capture_mire_1.png',0)

# print(cv2.findChessboardCorners(img, (7, 7)))
found, coord_px = cv2.findChessboardCorners(img, (7, 7), None)
found, coord_px2 = cv2.findChessboardCorners(img2, (7, 7), None)
cv2.drawChessboardCorners(img, (7,7), coord_px, found)
cv2.drawChessboardCorners(img2, (7,7), coord_px2, found)
#de haut en bas de gauche à droite

print(coord_px2)



#Coordonnées des coins sur l'image en réalité (coordonnées objet de la mire)
coord_mm = [[[20*i, 20*j] for i in range(7)] for j in range(7)]
coord_mm = np.reshape(coord_mm, np.shape(coord_px))
print(coord_mm)

zToUse = 0
zToUse2 = 120
i2, i1 = np.shape(img)


i1 = i1/2
i2 = i2/2

print(i1,i2)

# mem = np.copy(coord_px[:,0,1])
# coord_px[:,0,1] = coord_px[:,0,0]
# coord_px[:,0,0] = mem

# mem2 = np.copy(coord_px2[:,0,1])
# coord_px2[:,0,1] = coord_px2[:,0,0]
# coord_px2[:,0,0] = mem2

sgnO2c =-1# (i2 > coord_px[0,0,1]) * (-1) + (i2 < coord_px[0,0,1]) * 1
coord_px = np.array(coord_px)
coord_px2 = np.array(coord_px2)

print(coord_px)
U1 = np.concatenate((coord_px[:,:,0], coord_px2[:,:,0]), axis = 0) - i1
U2 = np.concatenate((coord_px[:,:,1], coord_px2[:,:,1]), axis = 0) - i2
print(U1)

U = [[U2[i][0]]*4 + [-U1[i][0]]*3 for i in range(len(U1))]
x0 = [coord_mm[i,0,:].tolist() + [zToUse, 1] + coord_mm[i,0,:].tolist() + [zToUse] for i in range(np.shape(coord_px)[0])]
x0 += [coord_mm[i,0,:].tolist() + [zToUse2, 1] + coord_mm[i,0,:].tolist() + [zToUse2] for i in range(np.shape(coord_px2)[0])]

#Multiplication de Hadamard
print(U)
print(i1,i2)
print(np.shape(x0))

print(coord_mm[1,0,:].tolist())


A = np.multiply(U,x0)
print(A)

A_inv = np.linalg.pinv(A)
print(A.shape)
l=np.dot(A_inv,U1)
l=l.T
l=l[0]



modo2c=1/math.sqrt(l[4]**2+l[5]**2+l[6]**2)
print("modo2c",modo2c)
beta=modo2c*math.sqrt(l[0]**2+l[1]**2+l[2]**2)
print("beta",beta)
o2c = sgnO2c * modo2c
o1c=l[3]*o2c/beta
print("o1c",o1c)
r11=l[0]*o2c/beta
print("r11",r11)
r12=l[1]*o2c/beta
print("r12",r12)
r13=l[2]*o2c/beta
print("r13",r13)
r21=l[4]*o2c
print("r21",r21)
r22=l[5]*o2c
print("r22",r22)
r23=l[6]*o2c
print("r23",r23)

r1 = np.array([r11,r12,r13])
r2 = np.array([r21,r22,r23])
r3 = np.cross(r2,r1)

r31=r3[0]
print("r31",r31)
r32=r3[1]
print("r32",r32)
r33=r3[2]
print("r33",r33)

phi=-math.atan(r23/r33)
gamma=-math.atan(r12/r11)
omega=math.atan(r13/(-r23*math.sin(phi)+r33*math.cos(phi)))

print(beta, np.array([phi, gamma, omega])*180/math.pi)

x0bis = np.array([coord_mm[i,0,:].tolist() + [zToUse] for i in range(np.shape(coord_px)[0])]).T
print(x0bis)
x0bis = np.concatenate((x0bis,np.array([coord_mm[i,0,:].tolist() + [zToUse2] for i in range(np.shape(coord_px2)[0])]).T), axis = 1)
print("caca")
print(x0bis)
vecR2 = np.reshape(-(np.dot(r2, x0bis)+o2c), np.shape(U2))
print(vecR2)
print(o2c)

print(np.shape(x0bis))


B = np.concatenate([U2, vecR2], axis = 1) #ok

print(r2)
print(x0bis)
print(-np.dot(r2, x0bis))
()

vecR3 = np.reshape((np.dot(r3, x0bis)), np.shape(U2))
print(vecR3)
print(np.shape(vecR3))

R = np.multiply(vecR3, -U2) #multiply ok



r31=r3[0]
print("r31",r31)
r32=r3[1]
print("r32",r32)
r33=r3[2]
print("r33",r33)
print("vecR3",vecR3)
print("r2",r2)
print("x0bis",x0bis)
print("vecR3",vecR3)



B_inv = np.linalg.pinv(B)
M = np.dot(B_inv, R)

print(np.shape(M))

f = 4
o3c = M[0]
f2=M[1]

s2 = f/f2
s1 = s2/beta

print(f)
print(s1,s2)

f1=f/s1

x_test=[100,100,120]

test_u1 = f1*(r11*x_test[0]+r12*x_test[1]+r13*x_test[2]+o1c)/(r31*x_test[0]+r32*x_test[1]+r33*x_test[2]+o3c)
test_u2 = f2*(r21*x_test[0]+r22*x_test[1]+r23*x_test[2]+o2c)/(r31*x_test[0]+r32*x_test[1]+r33*x_test[2]+o3c)


print(int(test_u1+i1))
print(int(test_u2+i2))

print(np.vdot(r2,r1))

#cv2.circle(img2,(int(test_u1+i1),int(test_u2+i2)),5,0.5,-1)
cv2.circle(img2,(int(test_u2+i2),int(test_u1+i1)),5,0.5,-1)
cv2.circle(img2,(int(test_u1+i1),int(test_u2+i2)),5,0.5,-1)
cv2.circle(img2,(int(test_u2),int(test_u1)),5,0.5,-1)
cv2.circle(img2,(int(test_u1),int(test_u2)),5,0.5,-1)
cv2.circle(img2,(int(test_u2+i2),int(test_u1+i2)),5,0.5,-1)
cv2.circle(img2,(int(test_u1+i2),int(test_u2+i1)),5,0.5,-1)

cv2.circle(img,(int(test_u2+i2),int(test_u1+i1)),5,0.5,-1)
cv2.circle(img,(int(test_u1+i1),int(test_u2+i2)),5,0.5,-1)
cv2.circle(img,(int(test_u2),int(test_u1)),5,0.5,-1)
cv2.circle(img,(int(test_u1),int(test_u2)),5,0.5,-1)
cv2.circle(img,(int(test_u2+i2),int(test_u1+i2)),5,0.5,-1)
cv2.circle(img,(int(test_u1+i2),int(test_u2+i1)),5,0.5,-1)

img[160:164,62:66] = 0

#img[int(test_u1+i1),int(test_u2+i2)]=255
#img2[230-2:230+2,215-2:215+2]=[0,0,255]
#cv2.circle(img,(230,215),5,0.5,-1)

cv2.imshow('Capture_Affine', img2) #affichage

cv2.waitKey(0)
cv2.destroyAllWindows()









