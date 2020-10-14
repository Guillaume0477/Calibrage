import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


#Paramètrage avec le nom des images à utiliser pour le calibrage
#name_img = 'capture_mire_0.png'
#name_img2 = 'capture_mire_1.png'
name_img = '111.jpg'
name_img2 = '222.jpg'
name_img3 = '333.jpg'

#Profondeur des images à considérer
zToUse = 0 # image1
zToUse2 = 100 # image2
zToUse3 = 200 # image3

#Distance focale de la caméra
f = 4



#Lecture des images
img = cv2.imread(name_img,0)
img2 = cv2.imread(name_img2,0)
img3 = cv2.imread(name_img3,0)

#Calibrage à partir des photos d'origine
#img = cv2.imread('WIN_20201012_14_35_29_Pro.jpg',0)
#img = cv2.imread('WIN_20201012_14_35_01_Pro.jpg',0)
#img2 = cv2.imread('WIN_20201012_14_34_36_Pro.jpg',0)


#Recherche des coordonnées des intersections des damiers et affichage sur les images
#NB : les intersections sont classées de gauche à droite puis de haut en bas en 
#començant en haut à gauche.
found, coord_px = cv2.findChessboardCorners(img, (7, 7), None)
found, coord_px2 = cv2.findChessboardCorners(img2, (7, 7), None)
cv2.drawChessboardCorners(img, (7,7), coord_px, found)
cv2.drawChessboardCorners(img2, (7,7), coord_px2, found)

#Coordonnées des coins sur l'image en réalité (coordonnées objet de la mire)
coord_mm = [[[20*i, 20*j] for i in range(7)] for j in range(7)]
coord_mm = np.reshape(coord_mm, np.shape(coord_px))

#Taille de l'image
i2, i1 = np.shape(img)
#Centre de l'image, origine du repère image.
i1 = i1/2
i2 = i2/2

#Détermination du signe de oc2 en fonction de la position de l'origine du repère objet
#projeté sur l'image en fonction du centre e l'image (origine du repère image)
sgnO2c =(i2 > coord_px[0,0,1]) * (-1) + (i2 < coord_px[0,0,1]) * 1

#Réagencement des variable (facilitation des calculs à l'aide de numpy.arrays)
coord_px = np.array(coord_px)
coord_px2 = np.array(coord_px2)
coord_im = np.concatenate((coord_px[:,0], coord_px2[:,0]), axis = 0)

#Séparation des coordonnées selon l'axe e1 et e2
U1 = np.concatenate((coord_px[:,:,0], coord_px2[:,:,0]), axis = 0) - i1
U2 = np.concatenate((coord_px[:,:,1], coord_px2[:,:,1]), axis = 0) - i2



###############################
# Début de la méthode de Tsaï #
###############################

##
# Récupération des valeurs |oc2|, oc1, r11, r12,  r13, r21, r22, r23 :
# Résolution du système U1 = A L 
##

#Création du système matriciel
U = [[U2[i][0]]*4 + [-U1[i][0]]*3 for i in range(len(U1))]
x0 = [coord_mm[i,0,:].tolist() + [zToUse, 1] + coord_mm[i,0,:].tolist() + [zToUse] for i in range(np.shape(coord_px)[0])]
x0 += [coord_mm[i,0,:].tolist() + [zToUse2, 1] + coord_mm[i,0,:].tolist() + [zToUse2] for i in range(np.shape(coord_px2)[0])]

#Multiplication de Hadamard pour obtenir la matrice A
A = np.multiply(U,x0)
#Inversion de la matrice A
A_inv = np.linalg.pinv(A)
#Résolution du système linéaire, récupération de lamatrice L (notée l)
l=np.dot(A_inv,U1)
l=l.T
l=l[0]

#Récupération des valeurs
modo2c=1/math.sqrt(l[4]**2+l[5]**2+l[6]**2)         # module de o2c
beta=modo2c*math.sqrt(l[0]**2+l[1]**2+l[2]**2)
o2c = sgnO2c * modo2c
o1c=l[3]*o2c/beta
r11=l[0]*o2c/beta
r12=l[1]*o2c/beta
r13=l[2]*o2c/beta
r21=l[4]*o2c
r22=l[5]*o2c
r23=l[6]*o2c

##
# Récupération des valeurs r31, r32, r33 par produit vectoriel
##

#Création de vecteurs
r1 = np.array([r11,r12,r13])
r2 = np.array([r21,r22,r23])
#Produit vectoriel
r3 = np.cross(r1,r2)
#Isolation des valeurs
r31=r3[0]
r32=r3[1]
r33=r3[2]

##
# Récupération des angles d'Euler permettant de vérifier la cohérence du calibrage, phi, gamma et omega
##
phi=-math.atan(r23/r33)
gamma=-math.atan(r12/r11)
omega=math.atan(r13/(-r23*math.sin(phi)+r33*math.cos(phi)))

##
#Information utilisateur permettant la vérification de la cohérence du calibrage
##

print("La valeur attendu de Beta est proche de 1\n Valeur de Beta obtenue après calibrage : ",beta,"\n Angles d'Euler obtenus après calibrage [phi,gamma,omega]", np.array([phi, gamma, omega])*180/math.pi)

##
# Récupération des valeurs oc3 et f2 :
# Résolution du système B M = R
##

#Création du système matriciel
x0bis = np.array([coord_mm[i,0,:].tolist() + [zToUse] for i in range(np.shape(coord_px)[0])]).T
x0bis = np.concatenate((x0bis,np.array([coord_mm[i,0,:].tolist() + [zToUse2] for i in range(np.shape(coord_px2)[0])]).T), axis = 1)
vecR2 = np.reshape(-(np.dot(r2, x0bis)+o2c), np.shape(U2))
#Obtention de B
B = np.concatenate([U2, vecR2], axis = 1) 

#Obtention de R
vecR3 = np.reshape((np.dot(r3, x0bis)), np.shape(U2))
R = np.multiply(vecR3, -U2) #multiply ok

#Inversion de B puis résolution du système matriciel
B_inv = np.linalg.pinv(B)
M = np.dot(B_inv, R)

#Isolation des valeurs
o3c = M[0]
f2 = M[1]

##
# Récupération des valeurs s2 et s1 la taille d'un pixel du capteur à partir de : 
# beta = f1/f2 = s2/s1 et f2 = f/s2
##

s2 = f/f2
s1 = s2/beta

print("Distance focale f de la caméra ",f)
print("Taille d'un pixel du capteur s1xs2 : ",s1,"x",s2)

f1=f/s1

##
# Affichage des resultats de la calibration
##

# Ajout des coordonées objets de la troisième image
x0bis_3_images = np.concatenate((x0bis,np.array([coord_mm[i,0,:].tolist() + [zToUse3] for i in range(np.shape(coord_px2)[0])]).T), axis = 1)

# Images en couleurs
img_print = cv2.imread(name_img,3)
img2_print = cv2.imread(name_img2,3)
img3_print = cv2.imread(name_img3,3)


# Paramètres de cv2.circle pour dessiner des cercles
thickness = 2
radius = 4
color = (158, 108, 253) #BRG

# Boucle sur les points à reconstruire
for loop in range(np.shape(x0bis_3_images)[1]) :

    # Reconstruction des points pixels à partir des points objets
    reproduced_U11 = f1*(r11*x0bis_3_images[0][loop]+r12*x0bis_3_images[1][loop]+r13*x0bis_3_images[2][loop]+o1c)/(r31*x0bis_3_images[0][loop]+r32*x0bis_3_images[1][loop]+r33*x0bis_3_images[2][loop]+o3c) + i1 
    reproduced_U22 = f2*(r21*x0bis_3_images[0][loop]+r22*x0bis_3_images[1][loop]+r23*x0bis_3_images[2][loop]+o2c)/(r31*x0bis_3_images[0][loop]+r32*x0bis_3_images[1][loop]+r33*x0bis_3_images[2][loop]+o3c) + i2 
    center_coordinates = (int(reproduced_U11), int(reproduced_U22))
    #if loop < np.shape(x0bis)[1]/2:
    if loop < np.shape(x0bis_3_images)[1]/3:
        cv2.circle(img_print, center_coordinates, radius, color, thickness) # dessine un cercle
    #else:
    elif loop < np.shape(x0bis_3_images)[1]*2/3:
        cv2.circle(img2_print, center_coordinates, radius, color, thickness) # dessine un cercle
    else :
        cv2.circle(img3_print, center_coordinates, radius, color, thickness) # dessine un cercle



cv2.imshow('image 1', img_print) #affichage
cv2.imshow('image 2', img2_print) #affichage
cv2.imshow('image 3', img3_print) #affichage


##
# Calibrage de la camera par cv2
##

# Points objets en float32
objpoints = np.array(x0bis.T,np.float32)
# Points images en float32
imgpoints = np.array(coord_im, np.float32)

# Image que l'on veut modifier en enlevant les distortions de la camera
img = cv2.imread('2222.jpg')
img_size = (img.shape[1], img.shape[0])

# Estimation automatique des paramètres intrinsèques de la camera par cv2
camera_matrix = cv2.initCameraMatrix2D([objpoints],[imgpoints], img_size)

# Calibrage automatique de la camera par cv2
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array([objpoints],dtype=np.float32),np.array([imgpoints],dtype=np.float32), img_size, camera_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

# Retrait des deformations de la camera
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('./1111_undist.png',dst)

cv2.imshow('undist', dst) #affichage

cv2.waitKey(0)
cv2.destroyAllWindows()

