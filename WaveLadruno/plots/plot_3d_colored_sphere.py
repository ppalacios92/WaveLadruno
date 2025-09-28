import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from WaveLadruno.tools.radiation_vectors import *
from WaveLadruno.tools.radiation_vectors import (
    create_angular_grid, 
    get_spherical_unit_vectors,
    A_FP, A_FS, A_N, A_IP, A_IS,
    get_all_patterns,
    contract_tensor
)

# === Mallado angular ===
theta_grid, phi_grid = create_angular_grid(300, 300)

# === Vectores esféricos unitarios ===
gamma, r_hat, theta_hat, phi_hat = get_spherical_unit_vectors(theta_grid, phi_grid)

# === Diccionario de patrones ===
patterns = get_all_patterns(gamma)

# Para colocar texto con los componentes no nulos en M
def get_nonzero_components(M, tol=1e-12):
    comps = []
    for i in range(3):
        for j in range(3):
            if abs(M[i, j]) > tol:
                comps.append(f"M{i+1}{j+1} = {M[i,j]:.2f}")
    return comps

def plot_pattern(M, name, projection='r', cmap='viridis', M0=1.0):
    A = patterns[name]
    vec = contract_tensor(A, M)

    if projection == 'r':  # P-wave
        norm = np.sum(vec * r_hat, axis=0)

    elif projection == 'theta':  # SV-wave
        norm = np.sum(vec * theta_hat, axis=0)

    elif projection == 'phi':  # SH-wave
        norm = np.sum(vec * phi_hat, axis=0)

    elif projection == 'S':  # Total S-wave (SV + SH)
        sv = np.sum(vec * theta_hat, axis=0)
        sh = np.sum(vec * phi_hat, axis=0)
        norm = np.sqrt(sv**2 + sh**2)

    else:
        norm = np.linalg.norm(vec, axis=0)

    norm = np.abs(norm)

    # coordenadas cartesianas
    x = norm * np.sin(theta_grid) * np.cos(phi_grid)
    y = norm * np.sin(theta_grid) * np.sin(phi_grid)
    z = norm * np.cos(theta_grid)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    cmap_obj = plt.colormaps[cmap]
    ax.plot_surface(x, y, z, facecolors=cmap_obj(norm / norm.max()),
                    rstride=5, cstride=5, linewidth=0.0, alpha=0.9)

    comps = get_nonzero_components(M)
    legend_text = "\n".join(comps) if comps else "All Mij = 0"

    # === Título informativo según proyección ===
    proj_label = {
        'r':     "P-wave (r)",
        'theta': "SV-wave (theta)",
        'phi':   "SH-wave (phi)",
        'S':     "Total S-wave (S)",
        'total': "Vector norm (P + S)"
    }

    proj_text = proj_label.get(projection, f"{projection}-proj")
    ax.set_title(f"{name} radiation pattern\nProjection: {proj_text}", fontsize=12, fontweight='bold')

    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_box_aspect([1,1,1])
    ax.grid(False)
    ax.set_xlim([-M0*1.1, M0*1.1])
    ax.set_ylim([-M0*1.1, M0*1.1])
    ax.set_zlim([-M0*1.1, M0*1.1])
    ax.set_xticks(np.linspace(-M0, M0, 3))
    ax.set_yticks(np.linspace(-M0, M0, 3))
    ax.set_zticks(np.linspace(-M0, M0, 3))
    ax.view_init(elev=15, azim=45)

    ax.text2D(0.05, 0.95, f"Active components:\n{legend_text}", transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.show()

def plot_3d_colored_sphere(pattern_name, M, projection='r', title="3D Beach Ball"):
    """
    Beach ball 3D con superficie coloreada según amplitud
    """
    # Usar tu patrón existente
    A = patterns[pattern_name]
    vec = contract_tensor(A, M)
   
    # Calcular amplitud según proyección (usando tu lógica)
    if projection == 'r':  # P-wave
        u_amplitude = np.sum(vec * r_hat, axis=0)
    elif projection == 'theta':  # SV-wave
        u_amplitude = np.sum(vec * theta_hat, axis=0)
    elif projection == 'phi':  # SH-wave
        u_amplitude = np.sum(vec * phi_hat, axis=0)
    elif projection == 'S':  # Total S-wave
        sv = np.sum(vec * theta_hat, axis=0)
        sh = np.sum(vec * phi_hat, axis=0)
        u_amplitude = np.sqrt(sv**2 + sh**2)
   
    # Coordenadas de la esfera unitaria
    x_sphere = np.sin(theta_grid) * np.cos(phi_grid)
    y_sphere = np.sin(theta_grid) * np.sin(phi_grid)
    z_sphere = np.cos(theta_grid)
   
    # Crear colores según amplitud
    if projection in ['S']:
        # S-wave: solo positiva, escala de rojos
        colors = np.zeros(u_amplitude.shape + (3,))  # RGB
        normalized_amp = u_amplitude / np.max(u_amplitude)
        colors[:,:,0] = normalized_amp  # Canal rojo
        colors[:,:,1] = 0               # Sin verde
        colors[:,:,2] = 0               # Sin azul
    else:
        # P-wave, SV, SH: rojo para +, azul para -
        colors = np.zeros(u_amplitude.shape + (3,))  # RGB
       
        # Máscara para positivos y negativos
        positive = u_amplitude > 0
        negative = u_amplitude < 0
       
        # Rojo para compresión (positivo)
        colors[positive] = [0.8, 0, 0]    # tab:red
       
        # Gris suave para dilatación (negativo)
        colors[negative] = [0.9, 0.9, 0.9]  # Gris suave
   
    # Figura 3D
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
   
    # Surface plot con colores
    surf = ax.plot_surface(x_sphere, y_sphere, z_sphere,
                          facecolors=colors,
                          rstride=1, cstride=1,
                          linewidth=0, antialiased=False,
                          alpha=1.0)
   
    # Configuración
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(f'{title}\nPattern: {pattern_name}, Projection: {projection}',
                 fontsize=12, fontweight='bold')
   
    # Límites iguales para esfera
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.grid(False)
    # Vista
    ax.view_init(elev=20, azim=45)
   
    # Información
    comps = get_nonzero_components(M)
    legend_text = "\n".join(comps) if comps else "All Mij = 0"
   
    info_text = f"Active components:\n{legend_text}\n\n"
    if projection in ['S']:
        info_text += "Red intensity = S-wave amplitude"
    else:
        info_text += "Red = Compression (+)\nGray = Dilatation (-)"
   
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
   
    plt.show()