from pathlib import Path
import sys
import zipfile
import pooch

NABA_ZIP_URL = "https://zenodo.org/api/records/18227691/files/NABA-800FC.zip/content"
NABA_ZIP_NAME = "NABA-800FC.zip"
TRACT_DIR_NAME = "tract_atlases"  # viewer expects this name


def is_tract_vtp(base: str) -> bool:
    return base.endswith(".vtp") and (
        base.startswith("cluster_") or base == "atlas.vtp"
    )


def main():
    root = Path(__file__).resolve().parent
    zip_path = root / NABA_ZIP_NAME
    out_dir = root / TRACT_DIR_NAME

    # If already extracted, done.
    existing = list(out_dir.glob("cluster_*.vtp")) if out_dir.is_dir() else []
    if existing:
        print(f"Already extracted: {len(existing)} cluster_*.vtp in {out_dir}")
        if zip_path.is_file():
            zip_path.unlink()
            print(f"Removed stale zip: {zip_path.name}")
        return

    # Download if not already present.
    if not zip_path.is_file():
        print(f"Downloading {NABA_ZIP_NAME} (~960 MB, one-time) ...")
        pooch.retrieve(
            url=NABA_ZIP_URL,
            known_hash=None,
            fname=NABA_ZIP_NAME,
            path=str(root),
            progressbar=True,
        )
    else:
        print(f"Using existing {zip_path.name} on disk.")

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(
            f"{zip_path} is not a valid zip. "
            f"Delete it and re-run (must use /content URL, not /container)."
        )

    out_dir.mkdir(exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            base = Path(name).name
            if not is_tract_vtp(base):
                continue
            dest = out_dir / base
            with zf.open(name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted.append(base)

    if not extracted:
        raise RuntimeError(
            f"No cluster_*.vtp or atlas.vtp found in {zip_path.name}. "
            f"Sample: {zf.namelist()[:10]}"
        )

    zip_path.unlink()
    print(f"Extracted {len(extracted)} .vtp files into {out_dir}")
    print(f"Deleted {zip_path.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
