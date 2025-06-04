
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necesario para proyecciones 3D

def graficar_3D_reconstruccion(imagen_2d, titulo="Reconstrucción 3D SUPPoSE"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    h, w = imagen_2d.shape
    X = np.arange(w)
    Y = np.arange(h)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, imagen_2d, cmap='viridis', edgecolor='none', antialiased=True)
    ax.set_title(titulo)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Intensidad")
    plt.tight_layout()
    plt.show()

# Ejemplo de uso con una reconstrucción ficticia
if __name__ == "__main__":
    reconstruida = np.load("reconstruida.npy")  # Cargar matriz reconstruida
    graficar_3D_reconstruccion(reconstruida)
