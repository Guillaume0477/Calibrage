import numpy as np
import cv2
from matplotlib import pyplot as plt


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



# cv2.imshow('Capture_Affine', img) #affichage
# cv2.waitKey(0)
# cv2.destroyAllWindows()









