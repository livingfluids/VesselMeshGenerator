Helical / Spiral Vessel Mesh Generator

This small Python project generates 3D helical vessel meshes and can produce "spiral artery" variants where the helix radius changes along revolutions.

Files
- `spiral.py`: Main module. Contains the `VesselMesh` class with methods to generate, visualize, and export meshes (OBJ, ASCII STL).
- `helical_vessel.obj` / `helical_vessel.stl`: Example outputs produced by running the script.
- `spiral_artery.obj` / `spiral_artery.stl`: Example outputs produced by running the script.

Dependencies
- Python 3.8+
- numpy
- matplotlib

Install dependencies (recommended in a virtual environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib
```

Basic usage

Run the script (this will generate and export example meshes):

```bash
python3 spiral.py
```

This writes `helical_vessel.obj`, `helical_vessel.stl`, `spiral_artery.obj`, and `spiral_artery.stl` to the script's current working directory.

Running without opening GUI windows

If you run on a headless machine or want to avoid opening GUI windows and related macOS IMK log lines, set the matplotlib backend to `Agg` or use the environment variable:

```bash
MPLBACKEND=Agg python3 spiral.py
```

Or edit `spiral.py` near the top and add:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

API (quick)

```python
from spiral import VesselMesh
# helical (constant radius)
v = VesselMesh(mode='helical')
v.generate_mesh()
v.visualize()

# spiral artery (radius varies along the helix; taper optionally reduces cross-section)
s = VesselMesh(mode='spiral', radius_variation= -2.0, taper_ratio=0.2)
s.generate_mesh()
s.visualize()
```

Notes and tips
- Outputs are written to the current working directory by default. Use absolute paths or create an `output/` folder for organization.
- The `mode` parameter accepts `helical` (constant radius) or `spiral` (radius changes along revolutions). `radius_variation` is the total radial change across the full helix; `taper_ratio` optionally reduces the cross-section radius along the helix.
- If you want caps (watertight mesh) or inner surface support, I can add optional flags for that.

License
- This project is licensed under the MIT License — see the `LICENSE` file for full terms. If you use this code in research, please cite the project (`CITATION.cff`) and contact the author: Leo Liu <leo.liu@eng.famu.fsu.edu>.
