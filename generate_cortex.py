
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent / "data"

SURFACES = {
    "outer": "OuterSurf",
    "middle": "MiddleSurf",
    "inner": "InnerSurf",
    "inflated": "InflateSurf",
    "sphere": "SphereSurf",
}

manifest = {}

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir() or not folder.name.isdigit():
        continue

    age = int(folder.name)
    entry = {}

    for hemi, suffix in [("left", "lh"), ("right", "rh")]:
        surfaces = {}

        for key, surface_name in SURFACES.items():
            filename = (
                f"SparseAtlas_{age:02d}_{suffix}.{surface_name}.vtk"
            )
            path = folder / filename

            if path.exists():
                surfaces[key] = path.relative_to(
                    Path(__file__).resolve().parent
                ).as_posix()

        if surfaces:
            entry[hemi] = surfaces

    if entry:
        manifest[str(age)] = entry

output = Path(__file__).resolve().parent / "atlas.json"
output.write_text(json.dumps(manifest, indent=2))

print(f"Saved {output}")
for age, entry in manifest.items():
    print(
        f"{age} months: "
        f"left={len(entry.get('left', {}))}, "
        f"right={len(entry.get('right', {}))} surfaces"
    )

