# **WaveLadruno**

A Python package for 3D seismic wave propagation visualization and radiation pattern analysis. This tool provides comprehensive functionality for understanding seismic wave behavior through moment tensor mechanics, radiation patterns, and focal mechanism visualization.

---

## ⚙️ Features

- **Moment Tensor Operations**: Calculate moment tensors from strike, dip, and rake parameters using standard seismological conventions
- **3D Radiation Patterns**: Generate far-field and near-field radiation patterns for P-waves, S-waves, SV, and SH components
- **Beach Ball Visualization**: Create traditional 2D stereographic projections and modern 3D beach ball representations
- **Interactive 3D Plots**: Visualize radiation patterns as deformed surfaces with directional vector fields
- **Multiple Projection Types**: Support for radial (P-wave), theta (SV), phi (SH), and total S-wave projections
- **Comprehensive Tensor Patterns**: Implements A_FP, A_FS, A_N, A_IP, A_IS tensor formulations for complete wave field analysis

---

## 📦 Requirements

- Python 3.8 or higher
- Core dependencies:
  - numpy
  - matplotlib
  - mpl_toolkits (for 3D plotting)
- Optional for enhanced visualization:
  - jupyter (for interactive notebooks)

---

## 🚀 Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/ppalacios92/WaveLadruno.git
cd WaveLadruno
pip install -e .
```

---

## 📁 Repository Structure

```bash
WaveLadruno/
├── WaveLadruno/              # Main package directory
│   ├── tools/                # Core computational tools
│   │   ├── moment_tensor_tools.py    # Moment tensor calculations
│   │   └── radiation_tools.py        # Radiation pattern tensors
│   ├── plots/                # Visualization modules
│   │   ├── radiation_patterns.py     # 3D radiation plotting
│   │   ├── plot_beachball_2d.py     # 2D beach ball plots
│   │   └── plot_sphere_with_vectors.py # 3D vector field plots
│   └── __init__.py
├── examples/                 # Jupyter notebooks and examples
├── tests/                    # Unit tests and validation
├── pyproject.toml           # Package configuration
└── README.md                # Project documentation
```

---

## 🔬 Scientific Background

This package implements the mathematical framework for seismic wave radiation patterns based on:

- **Moment Tensor Theory**: Represents seismic sources through symmetric 3×3 tensors describing equivalent body forces
- **Far-field Approximations**: Uses Green's function solutions for elastic wave propagation in infinite media
- **Radiation Pattern Tensors**: Implements the complete set of A^X tensors (A_FP, A_FS, A_N, A_IP, A_IS) for near-field and far-field wave analysis
- **Stereographic Projection**: Standard seismological visualization using lower hemisphere projection

---

## 📖 Quick Start

### Basic Moment Tensor Creation

```python
import numpy as np
from WaveLadruno.tools import moment_tensor_from_strike_dip_rake

# Create moment tensor from fault parameters
M = moment_tensor_from_strike_dip_rake(strike=90, dip=90, rake=0)
print(M)
```

### 3D Radiation Pattern Visualization

```python
from WaveLadruno.plots import plot_radiation_patterns

# Plot P-wave radiation pattern
plot_radiation_patterns(M, "AFP", projection='r')

# Plot S-wave radiation pattern
plot_radiation_patterns(M, "AFS", projection='S')
```

### Traditional Beach Ball Plots

```python
from WaveLadruno.plots import plot_beachball_set

# Generate complete set of 2D projections
plot_beachball_set(M, mechanism_name="Strike-Slip Fault")
```

### Advanced 3D Visualization

```python
from WaveLadruno.plots import plot_sphere_with_vectors

# 3D beach ball with directional vectors
plot_sphere_with_vectors("AFP", M, projection='r', 
                        title="P-wave: Compression/Dilatation Pattern")
```

---

## 🧮 Mathematical Framework

The package implements the complete mathematical formulation for seismic radiation patterns:

### Moment Tensor Components
- **M₁₁, M₂₂, M₃₃**: Normal stress components
- **M₁₂, M₁₃, M₂₃**: Shear stress components (symmetric tensor)

### Radiation Pattern Types
- **A_FP**: Far-field P-wave patterns
- **A_FS**: Far-field S-wave patterns  
- **A_N**: Near-field patterns
- **A_IP**: Intermediate P-field patterns
- **A_IS**: Intermediate S-field patterns

### Wave Projections
- **P-wave (r)**: Radial displacement component
- **SV-wave (θ)**: Vertical shear component
- **SH-wave (φ)**: Horizontal shear component
- **S-total**: Combined S-wave amplitude

---

## 🎯 Applications

- **Seismological Research**: Analyze focal mechanisms and fault kinematics
- **Educational Tools**: Visualize wave propagation concepts for teaching
- **Earthquake Analysis**: Interpret moment tensor solutions from seismic data
- **Structural Seismology**: Understand source radiation characteristics

---

## 🛑 Disclaimer

This tool is provided for educational and research purposes. While the implementation follows standard seismological formulations, users should:

- Validate results against established seismological software
- Understand the underlying physical assumptions and limitations
- Use appropriate discretization and numerical parameters for their applications
- Consult relevant literature for theoretical background

The authors assume no responsibility for incorrect interpretations or applications of the results.

---

## 👨‍💻 Author

Developed by Patricio Palacios B.
Structural Engineer | Python Developer | Seismic Modeler
GitHub: @ppalacios92

## 📚 How to Cite

If you use this tool in your work, please cite it as follows:

```bibtex
@misc{palacios2025waveladruno,
  author       = {Patricio Palacios B.},
  title        = {WaveLadruno: A Python package for 3D seismic wave propagation and radiation pattern visualization},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/ppalacios92/WaveLadruno}}
}
```

## 📄 Citation in APA (7th Edition)

Palacios B., P. (2025). *WaveLadruno: A Python package for 3D seismic wave propagation and radiation pattern visualization* [Computer software]. GitHub. https://github.com/ppalacios92/WaveLadruno

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features through the GitHub issues page.