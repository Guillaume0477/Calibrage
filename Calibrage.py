import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


img = cv2.imread('capture_mire_0.png',0)

# print(cv2.findChessboardCorners(img, (7, 7)))
found, coord_px = cv2.findChessboardCorners(img, (7, 7), None)
cv2.drawChessboardCorners(img, (7,7), coord_px, found)
#de haut en bas de gauche à droite

#Coordonnées des coins sur l'image en réalité (coordonnées objet de la mire)
coord_mm = [[[20*i, 20*j] for j in range(7)] for i in range(7)]
coord_mm = np.reshape(coord_mm, np.shape(coord_px))


coord_px = np.array(coord_px)
U1 = coord_px[:,:,0]


zToUse = 0;

U = [[coord_px[i,0,1]]*4 + [-coord_px[i,0,0]]*3 for i in range(len(U1))]
x0 = [coord_mm[i,0,:].tolist() + [zToUse, 1] + coord_mm[i,0,:].tolist() + [zToUse] for i in range(len(U1))]

#Multiplication de Hadamard
A = np.multiply(U,x0)



l = np.array([1,1,1,1,1,1,1])

o2c=1/math.sqrt(l[4]**2+l[5]**2+l[6]**2)
beta=o2c*math.sqrt(l[0]**2+l[1]**2+l[2]**2)
o1c=l[4]*o2c/beta
r11=l[0]*o2c/beta
r12=l[1]*o2c/beta
r13=l[2]*o2c/beta
r21=l[4]*o2c
r22=l[5]*o2c
r23=l[6]*o2c

u = np.array([r11,r12,r13])
v = np.array([r21,r22,r23])
r3 = np.cross(u,v)

r31=r3[0]
r32=r3[1]
r33=r3[2]


phi=-math.atan(r23/r33)
gama=-math.atan(r12/r11)
w=math.atan(r13/(r23*math.sin(phi)+r33*math.cos(phi)))




# cv2.imshow('Capture_Affine', img) #affichage
# cv2.waitKey(0)
# cv2.destroyAllWindows()









