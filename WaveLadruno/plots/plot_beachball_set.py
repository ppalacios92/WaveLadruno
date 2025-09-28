import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_beachball_set(M, mechanism_name="Fault Mechanism"):
    """
    Visualiza en subplots los patrones de radiación de ondas P, SV, SH y S-total.
    """
    plt.close('all')  # Cerrar figuras previas
    
    def draw_subplot(ax, x, y, u, title):
        circle = Circle((0, 0), 1, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        if not np.all(np.isnan(u)):
            ax.contourf(x, y, u, levels=[-1e10, 0, 1e10], colors=['white', 'gray'], alpha=0.8)
            ax.contour(x, y, u, levels=[0], colors='red', linewidths=2)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # === Mallado esférico ===
    n = 200
    theta, phi = np.linspace(0, np.pi, n), np.linspace(0, 2*np.pi, n)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    gx = np.sin(theta_grid) * np.cos(phi_grid)
    gy = np.sin(theta_grid) * np.sin(phi_grid)
    gz = np.cos(theta_grid)
    r_hat = np.array([gx, gy, gz])
    theta_hat = np.array([
        np.cos(theta_grid) * np.cos(phi_grid),
        np.cos(theta_grid) * np.sin(phi_grid),
        -np.sin(theta_grid)
    ])
    phi_hat = np.array([
        -np.sin(phi_grid),
        np.cos(phi_grid),
        np.zeros_like(phi_grid)
    ])
    
    # === Desplazamiento u = M · γ ===
    u = np.zeros((3,) + theta_grid.shape)
    for i in range(3):
        for j, g in enumerate([gx, gy, gz]):
            u[i] += M[i, j] * g
    
    # === Proyecciones ===
    u_P = np.sum(u * r_hat, axis=0)
    u_SV = np.sum(u * theta_hat, axis=0)
    u_SH = np.sum(u * phi_hat, axis=0)
    u_S = np.sqrt(u_SV**2 + u_SH**2)
    
    # === Proyección estereográfica (hemisferio inferior) ===
    lower = theta_grid >= np.pi/2
    r_proj = np.where(lower, np.tan((np.pi - theta_grid) / 4), np.nan)
    x = r_proj * np.cos(phi_grid)
    y = r_proj * np.sin(phi_grid)
    r_max = np.nanmax(r_proj)
    x, y = x / r_max, y / r_max
    
    u_dict = {
        "P-wave\n(Standard Beach Ball)": np.where(lower, u_P, np.nan),
        "SV-wave\n(Vertical Shear)":     np.where(lower, u_SV, np.nan),
        "SH-wave\n(Horizontal Shear)":   np.where(lower, u_SH, np.nan),
        "S-wave Total\n(|SV| + |SH|)":   np.where(lower, u_S, np.nan)
    }
    
    # === Crear figura ===
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (title, u_plot) in zip(axes.flat, u_dict.items()):
        draw_subplot(ax, x, y, u_plot, title)
    
    # === Mostrar tensor
    info = "Moment Tensor M:\n" + "\n".join(
        f"M{i+1}{j+1} = {M[i,j]:.2f}" for i in range(3) for j in range(3) if abs(M[i,j]) > 1e-10
    )
    fig.text(0.02, 0.98, info, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'{mechanism_name}\nRadiation Patterns: P, SV, SH, S',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    # return fig