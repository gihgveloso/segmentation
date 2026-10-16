import cv2 
import numpy as np
import random

matriz = [
    [0, 0, 0,0,0,0,0, 0, 0,0,0,0,0, 0, 0,0,0,0,0,0],
    [0, 0, 0,0,0,0,0, 0, 0,0,0,0,0, 0, 0,0,0,0,0,0],
    [0, 0, 0,0,0,0,0, 0, 0,0,0,0,0, 0, 0,0,0,0,0,0],
    [0, 0, 0,0,0,0,0, 0, 0,0,0,0,0, 0, 0,0,0,0,0,0],
    [0, 1, 0,0,0,0,0, 1, 0,1,1,1,0, 0, 0,1,0,0,0,1],
    [1, 0, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,1,1,1],
    [1, 1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,1,1,1],
    [0, 1, 1,1,1,1,1, 0, 1,1,1,1,1, 1, 1,1,1,1,1,0],
    [0, 1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,1,1,1],
    [1, 1, 1,1,1,1,1, 0, 1,1,1,1,1, 1, 1,1,1,0,1,1],
    [1, 1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,0,1,1],
    [1, 1, 1,1,1,1,1, 0, 1,1,1,0,1, 1, 1,1,1,1,1,1],
    [1, 1, 1,1,1,1,1, 0, 1,1,1,0,1, 1, 1,1,1,0,1,0],
    [1,     1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,1,1,1],
    [1, 1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,1,1,1],
    [1, 1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,0,1,1],
    [1, 1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,1,1,0],
    [1, 1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,1,1,1],
    [1, 1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,1,1,1],
    [1, 1, 1,1,1,1,1, 1, 1,1,1,1,1, 1, 1,1,1,1,1,1],
]

matriz_aleatoria = [
    [random.randint(0, 255) for _ in linha]
    for linha in matriz
]


vizinho_cima = np.roll(matriz_aleatoria,1,axis=0)
passou_cima = vizinho_cima >= matriz_aleatoria
vizinho_baixo = np.roll(matriz_aleatoria,-1,axis=0)
passou_baixo = vizinho_baixo >= matriz_aleatoria
vizinho_direita = np.roll(matriz_aleatoria,-1,axis=1)
passou_direita = vizinho_direita >= matriz_aleatoria
vizinho_esquerda = np.roll(matriz_aleatoria,1,axis=1)
passou_esquerda = vizinho_esquerda >= matriz_aleatoria
vizinho_dig_sup_dir = np.roll(matriz_aleatoria,(1,-1),axis=(0,1))
passou_dig_sup_dir = vizinho_dig_sup_dir >= matriz_aleatoria
vizinho_dig_inf_dir = np.roll(matriz_aleatoria,(-1,-1),axis=(0,1))
passou_dig_inf_dir = vizinho_dig_inf_dir >= matriz_aleatoria
vizinho_dig_sup_esq = np.roll(matriz_aleatoria,(1,1),axis=(0,1))
passou_dig_sup_esq = vizinho_dig_sup_esq >= matriz_aleatoria
viziho_dig_inf_esq = np.roll(matriz_aleatoria,(-1,1),axis=(0,1))
passou_dig_inf_esq = viziho_dig_inf_esq >= matriz_aleatoria

c_xy = (passou_cima + passou_baixo + passou_direita + passou_esquerda + passou_dig_sup_dir + passou_dig_inf_dir + passou_dig_sup_esq + passou_dig_inf_esq).astype(np.uint8)

print(c_xy)



