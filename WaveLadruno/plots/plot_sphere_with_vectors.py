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

def get_nonzero_components(M, tol=1e-12):
    comps = []
    for i in range(3):
        for j in range(3):
            if abs(M[i, j]) > tol:
                comps.append(f"M{i+1}{j+1} = {M[i,j]:.2f}")
    return comps

def plot_sphere_with_vectors(pattern_name, M, projection='r', title="3D Beach Ball + Vectors"):
    """
    Esfera coloreada + vectores radiales con escalas diferentes para compresión/dilatación
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
    
    # Crear colores según amplitud (igual que antes)
    if projection in ['S']:
        colors = np.zeros(u_amplitude.shape + (3,))
        normalized_amp = u_amplitude / np.max(u_amplitude)
        colors[:,:,0] = normalized_amp
        colors[:,:,1] = 0
        colors[:,:,2] = 0
    else:
        colors = np.zeros(u_amplitude.shape + (3,))
        positive = u_amplitude > 0
        negative = u_amplitude < 0
        colors[positive] = [0.8, 0, 0]      # Rojo
        colors[negative] = [0.9, 0.9, 0.9]  # Gris
    
    # Figura 3D
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. ESFERA COLOREADA (base)
    surf = ax.plot_surface(x_sphere, y_sphere, z_sphere,
                          facecolors=colors,
                          rstride=1, cstride=1,
                          linewidth=0, antialiased=False,
                          alpha=0.7)  # Más transparente para ver vectores
    
    # 2. VECTORES CON ESCALAS DIFERENTES
    # Submuestreo para vectores
    step = 5  # Cada 15 puntos
    theta_sub = theta_grid[::step, ::step]
    phi_sub = phi_grid[::step, ::step]
    u_sub = u_amplitude[::step, ::step]
    
    # Vectores unitarios submuestreados
    gx = np.sin(theta_sub) * np.cos(phi_sub)
    gy = np.sin(theta_sub) * np.sin(phi_sub)
    gz = np.cos(theta_sub)
    
    # Vector de vista para mostrar solo lado frontal
    view_dir = np.array([1, 1, 1])
    view_dir = view_dir / np.linalg.norm(view_dir)
    
    # ESCALAS DIFERENTES para compresión vs dilatación
    scale_compression = 1.0    # Mayor escala para compresiones (rojas)
    scale_dilatation = 1.0    # Menor escala para dilataciones (grises)
    
    # Dibujar vectores
    for i in range(theta_sub.shape[0]):
        for j in range(theta_sub.shape[1]):
            # Posición base en esfera unitaria
            x_base, y_base, z_base = gx[i,j], gy[i,j], gz[i,j]
            pos_vector = np.array([x_base, y_base, z_base])
            
            # Solo vectores frontales
            if np.dot(pos_vector, view_dir) > 0:
                amplitude = u_sub[i,j]
                
                # Escalas diferentes según signo
                if amplitude > 0:
                    # COMPRESIÓN: vectores más largos
                    vector_length = abs(amplitude) * scale_compression
                    direction = +1
                    color = 'darkred'
                    alpha = 0.9
                    linewidth = 1.0
                else:
                    # DILATACIÓN: vectores más cortos
                    vector_length = abs(amplitude) * scale_dilatation
                    direction = -1
                    color = 'dimgray'
                    alpha = 0.7
                    linewidth = 1.0
                
                # Vector radial
                dx = x_base * vector_length * direction
                dy = y_base * vector_length * direction
                dz = z_base * vector_length * direction
                
                # Dibujar vector si tiene magnitud significativa
                if abs(amplitude) > 1e-3:
                    ax.quiver(x_base, y_base, z_base, dx, dy, dz,
                             color=color, alpha=alpha, 
                             arrow_length_ratio=0.12,
                             linewidths=linewidth)
    
    # Configuración
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(f'{title}\nPattern: {pattern_name}, Projection: {projection}',
                 fontsize=12, fontweight='bold')
    
    # Límites
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.grid(False)
    ax.set_box_aspect([1,1,1])
    # Vista
    ax.view_init(elev=20, azim=45)
    
    # Información
    comps = get_nonzero_components(M)
    legend_text = "\n".join(comps) if comps else "All Mij = 0"
    
    info_text = f"Active components:\n{legend_text}\n\n"

    if projection in ['S']:
        info_text += "Red = S-wave amplitude"
    else:
        info_text += "Red vectors = Compression\nGray vectors = Dilatation"
    
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(facecolor='lightblue', alpha=0.8))

    plt.show()