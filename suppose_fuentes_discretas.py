
import numpy as np
from skimage.feature import peak_local_max

def extraer_fuentes_discretas(rho, umbral_relativo=0.75, min_distancia=3):
    max_valor = np.max(rho)
    umbral_valor = umbral_relativo * max_valor
    mascara = rho > umbral_valor
    maximos = peak_local_max(rho, min_distance=min_distancia, threshold_abs=umbral_valor, labels=mascara)
    rho_discreta = np.zeros_like(rho)
    for y, x in maximos:
        rho_discreta[y, x] = rho[y, x]
    return rho_discreta, maximos
