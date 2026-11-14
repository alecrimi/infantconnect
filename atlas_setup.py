"""First-run downloader and installer for the UNC infant cortical surface atlas."""
from pathlib import Path
import json
import re
import urllib.request
import zipfile

from PySide6.QtWidgets import QApplication, QProgressDialog
from PySide6.QtCore import Qt

from generate_cortex import build_manifest

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST = ROOT / "atlas.json"
ARCHIVE = ROOT / ".UNC-Infant-Cortical-Surface-Atlas_V1.11.download.zip"

# Public NITRC file-release download for the V1.11 package.
ATLAS_URL = (
    "https://www.nitrc.org/frs/download.php/13713/"
    "UNC-Infant-Cortical-Surface-Atlas_V1.11.zip"
)

EXPECTED_AGES = (1, 3, 6, 9, 12, 18, 24, 36, 48, 60, 72)
SURFACE_FILES = ("OuterSurf", "MiddleSurf", "InnerSurf", "InflateSurf", "SphereSurf")
VTK_NAME = re.compile(
    r"^SparseAtlas_(\d{1,2})_(lh|rh)\.(OuterSurf|MiddleSurf|InnerSurf|InflateSurf|SphereSurf)\.vtk$",
    re.IGNORECASE,
)


def _manifest_is_usable():
    """Return True only when atlas.json points to real cortical outer meshes."""
    if not MANIFEST.is_file():
        return False
    try:
        atlas = json.loads(MANIFEST.read_text(encoding="utf-8"))
        # The viewer exposes the complete 1–72 month atlas series. A manifest
        # with only one surviving age is not a valid completed installation.
        for age in EXPECTED_AGES:
            entry = atlas.get(str(age), {})
            for hemi in ("left", "right"):
                surfaces = entry.get(hemi, {})
                for surface in ("outer", "middle", "inner", "inflated", "sphere"):
                    relative = surfaces.get(surface)
                    if not relative or not (ROOT / relative).is_file():
                        return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def _download(progress):
    if ARCHIVE.is_file() and zipfile.is_zipfile(ARCHIVE):
        progress.setValue(65)
        progress.setLabelText("Using the previously downloaded atlas archive…")
        QApplication.processEvents()
        return
    request = urllib.request.Request(
        ATLAS_URL,
        headers={"User-Agent": "InfantBrainViewer/1.0 (atlas installer)"},
    )
    temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".part")
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            while True:
                if progress.wasCanceled():
                    raise InterruptedError("Atlas download cancelled by user.")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = min(65, int(downloaded * 65 / total))
                    progress.setValue(percent)
                    progress.setLabelText(
                        "Downloading UNC infant cortical atlas…\n"
                        f"{downloaded / (1024**2):,.0f} MB / {total / (1024**2):,.0f} MB"
                    )
                else:
                    progress.setValue(min(64, progress.value() + 1))
                    progress.setLabelText(
                        f"Downloading UNC infant cortical atlas… {downloaded / (1024**2):,.0f} MB"
                    )
                QApplication.processEvents()
        if downloaded < 1024 * 1024:
            raise RuntimeError("The NITRC download appears incomplete (file is unexpectedly small).")
        temporary.replace(ARCHIVE)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _extract_surfaces(progress):
    """Extract only the 5 mesh types used by generate_cortex.py."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        matches = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            match = VTK_NAME.match(Path(info.filename).name)
            if match:
                age = int(match.group(1))
                hemi, surface = match.group(2).lower(), match.group(3)
                matches.append((info, age, hemi, surface))

        if not matches:
            raise RuntimeError(
                "The downloaded NITRC archive contains no SparseAtlas_* VTK surfaces. "
                "The archive layout or release format may have changed."
            )

        total_bytes = sum(info.file_size for info, *_ in matches) or 1
        extracted_bytes = 0
        for index, (info, age, hemi, surface) in enumerate(matches, start=1):
            if progress.wasCanceled():
                raise InterruptedError("Atlas extraction cancelled by user.")
            canonical_name = f"SparseAtlas_{age:02d}_{hemi}.{surface}.vtk"
            destination = DATA_DIR / str(age) / canonical_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with archive.open(info, "r") as src, temporary.open("wb") as dst:
                while True:
                    if progress.wasCanceled():
                        raise InterruptedError("Atlas extraction cancelled by user.")
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    extracted_bytes += len(chunk)
                    progress.setValue(65 + int(30 * extracted_bytes / total_bytes))
                    progress.setLabelText(
                        f"Extracting cortical surfaces…\n"
                        f"{index}/{len(matches)} files · age {age} months · {hemi.upper()} {surface}"
                    )
                    QApplication.processEvents()
            temporary.replace(destination)

        # The viewer needs a complete age series, not merely a partially valid archive.
        missing = []
        for age in EXPECTED_AGES:
            for hemi in ("lh", "rh"):
                filename = f"SparseAtlas_{age:02d}_{hemi}.OuterSurf.vtk"
                if not (DATA_DIR / str(age) / filename).is_file():
                    missing.append(f"{age} months {hemi}")
        if missing:
            raise RuntimeError(
                "The archive did not contain all expected outer cortical surfaces: "
                + ", ".join(missing)
            )


def ensure_atlas_ready(parent=None):
    """Install and index the atlas on first launch; return False if cancelled."""
    if _manifest_is_usable():
        return True

    progress = QProgressDialog(
        "Preparing the UNC infant cortical atlas…", "Cancel", 0, 100, parent
    )
    progress.setWindowTitle("First-time atlas setup")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setValue(0)
    QApplication.processEvents()

    try:
        # If a previous setup already extracted the expected meshes, only rebuild the index.
        complete_data = all(
            (DATA_DIR / str(age) / f"SparseAtlas_{age:02d}_{hemi}.{surface}.vtk").is_file()
            for age in EXPECTED_AGES
            for hemi in ("lh", "rh")
            for surface in SURFACE_FILES
        )
        if not complete_data:
            progress.setLabelText(
                "Downloading the UNC 4D infant cortical atlas (about 977 MB)…"
            )
            QApplication.processEvents()
            _download(progress)
            progress.setValue(66)
            progress.setLabelText("Download complete. Extracting required VTK surfaces…")
            QApplication.processEvents()
            _extract_surfaces(progress)

        progress.setValue(96)
        progress.setLabelText("Building atlas.json index…")
        QApplication.processEvents()
        build_manifest()
        if not _manifest_is_usable():
            raise RuntimeError("Atlas preparation finished, but atlas.json is not usable.")
        progress.setValue(100)
        progress.setLabelText("Atlas is ready. Starting the 3D viewer…")
        QApplication.processEvents()
        ARCHIVE.unlink(missing_ok=True)
        progress.close()
        return True
    except InterruptedError:
        progress.close()
        return False
    except Exception:
        progress.close()
        raise
