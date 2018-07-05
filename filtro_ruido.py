import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft2, ifft2, fftshift, fftfreq

#Cargar la imagen 
imagen = plt.imread("moonlanding.png")
imagen_FT = fft2(imagen)
imagen_FT = fftshift(imagen_FT)

xdim = np.shape(imagen_FT)[0]

ydim = np.shape(imagen_FT)[1]

M = 50

imagenCentro_FT = np.copy(imagen_FT)
imagenCentro_FT[:xdim/2 -M,:] = 0
imagenCentro_FT[xdim/2 +M:,:] = 0
imagenCentro_FT[:,:ydim/2 -M] = 0
imagenCentro_FT[:,ydim/2 +M:] = 0

imagenBorde_FT = np.copy(imagen_FT)
imagenBorde_FT[xdim/2 -200:xdim/2 +200, ydim/2 -300:ydim/2 +300] = 0

#Graficamos la imagen con su respectiva trasnformada de Fourier
plt.figure()
plt.subplot(311)
plt.imshow(np.real(imagen), cmap='gray')
plt.subplot(312)
plt.imshow(np.real(imagen_FT), cmap='gray', vmin = 0, vmax = 200)
plt.subplot(313)
plt.imshow(np.real(imagenBorde_FT), cmap='gray', vmin = 0, vmax = 200)
plt.show() 
plt.clf()
plt.close()


lugaresPicos = np.where(np.abs(imagen_FT - imagenCentro_FT) > 120)
imagen_FT[lugaresPicos]=0
imagen_FT[xdim/2 -M:xdim/2 +M, ydim/2 -M:ydim/2 +M] = 0


plt.figure()
plt.subplot(211)
plt.imshow(np.abs(ifft2(imagen_FT + imagenCentro_FT)), cmap='gray')
plt.show()
