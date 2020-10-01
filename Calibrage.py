import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('capture_mire_0.png',0)

print(cv2.findChessboardCorners(img, (7, 7)))
found, coord_px = cv2.findChessboardCorners(img, (7, 7), None)
cv2.drawChessboardCorners(img, (7,7), coord_px, found)
#de haut en bas de gauche à droite

cv2.imshow('Capture_Affine', img) #affichage
cv2.waitKey(0)
cv2.destroyAllWindows()







