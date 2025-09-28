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

def plot_radiation_patterns(M , name, projection='r', cmap='viridis', M0=1.0):
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