# spintext — Analysis of Localized Magnetic Spin Textures

`spintext` is a lightweight Python library for representing, visualizing, and
analyzing **localized magnetic textures** such as skyrmions, antiskyrmions, or
composite multi-skyrmion states.

It was developed specifically for the publication: add pre-print here.

It provides:

- A fully standalone `SpinTexture` class  
- A symmetry-analysis module (`TextureSymmetry`)  
- Tools for rotation and mirror-symmetry detection  
- Visualization helpers

Originally part of a larger internal codebase, this repository contains a clean,
dependency-free subset focused on symmetry detection.

---

## Features

### ✔ SpinTexture
- Load from `.dat` file or NumPy arrays  
- Detect center of magnetization  
- Convert to polar coordinates  
- Compute geodesic distances  
- Embed textures into a lattice-defined circular region  
- Rotate textures around the z-axis  
- Save and load textures  

### ✔ TextureSymmetry
- Detect rotation symmetries: ``C_2`` , ``C_3``, ...  
- Detect mirror symmetries: ``D_n`` groups  
- Handle infinite symmetries  
- Visual diagnostics  
- Consistent lattice-based padding for stable results  

---

## Installation

Use a local install for now:

```bash
git clone <your-gitlab-url>
pip install -e spintext/
```

## Basic Usage
```python
import numpy as np
from spintext import SpinTexture, TextureSymmetry

# Load from CSV
tex = SpinTexture(path_texture="spintexture_1.dat")

# Or from arrays
# points = np.array([[x1, y1, z1], ...])
# spins = np.array([[sx1, sy1, sz1], ...])
# tex = SpinTexture(points=points, spins=spins)

# Symmetry analysis
tsym = TextureSymmetry(texture=tex, beta=np.pi / 2)  # beta: DMI angle (Néel ~ pi/2)

# Detect rotations (e.g. 2-fold, 3-fold,…)
tsym.detect_rotation_symmetries(prime_rotations=(2, 3, 4, 6), geodesic_crit=1.0)
print("Rotational symmetries:", tsym.rot_syms)

# Detect mirrors (may use rotation info)
tsym.detect_mirror_symmetries(geodesic_crit=1.0)
print("Mirror axes:", tsym.mirrors)

# Save summary
tsym.save_symmetries("symmetries.json")
```
## Visualization
```python
from spintext.plotting import plot_texture, plot_symmetry_axes
import matplotlib.pyplot as plt

tex = SpinTexture(path_texture="texture.dat")
tsym = TextureSymmetry(texture=tex, beta=np.pi / 2)
tsym.detect_rotation_symmetries()
tsym.detect_mirror_symmetries()

ax = plot_texture(tex, show=False, title="My texture")
plot_symmetry_axes(tex, tsym, ax=ax, show=True)
```

## Command Line Usage
You can also use the command line to deal with the script in scripts/. The options are e.g.
```bash
python visualize_texture.py texture.dat --detect-rotation --detect-mirror --loglevel DEBUG --geodesic-crit 0.007 --beta 0.5235987755982988 --show-rotation 3
```
