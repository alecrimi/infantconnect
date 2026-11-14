from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent / "data"
PROJECT_ROOT = Path(__file__).resolve().parent

SURFACES = {
    "outer": "OuterSurf",
    "middle": "MiddleSurf",
    "inner": "InnerSurf",
    "inflated": "InflateSurf",
    "sphere": "SphereSurf",
}


def build_manifest():
    """Build atlas.json from the extracted VTK surfaces in data/."""
    manifest = {}
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Atlas data directory not found: {ROOT}")

    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue

        age = int(folder.name)
        entry = {}
        for hemi, suffix in (("left", "lh"), ("right", "rh")):
            surfaces = {}
            for key, surface_name in SURFACES.items():
                filename = f"SparseAtlas_{age:02d}_{suffix}.{surface_name}.vtk"
                path = folder / filename
                if path.is_file() and path.stat().st_size > 0:
                    surfaces[key] = path.relative_to(PROJECT_ROOT).as_posix()
            if surfaces:
                entry[hemi] = surfaces
        if entry:
            manifest[str(age)] = entry

    if not any(entry.get(hemi, {}).get("outer") for entry in manifest.values()
               for hemi in ("left", "right")):
        raise RuntimeError(
            "No cortical outer surfaces were found. Check the extracted VTK atlas files."
        )

    output = PROJECT_ROOT / "atlas.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    for age, entry in manifest.items():
        print(
            f"{age} months: "
            f"left={len(entry.get('left', {}))}, "
            f"right={len(entry.get('right', {}))} surfaces"
        )
    return manifest


if __name__ == "__main__":
    build_manifest()
