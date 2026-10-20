import pyvista as pv, json
from pathlib import Path

# tract bounds (adult MNI)
t = pv.read(str(sorted(Path("tract_atlases").glob("cluster_*.vtp"))[0]))
print("tract bounds:", t.bounds)

# current brain bounds (infant template)
with open("atlas.json") as f:
    atlas = json.load(f)
age = sorted(atlas.keys(), key=lambda k: int(k))[0]
path = atlas[age]["left"]["outer"]
b = pv.read(path)
print(f"brain ({age}mo left) bounds:", b.bounds)
