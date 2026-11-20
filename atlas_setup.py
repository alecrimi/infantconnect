
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
TEMP_ARCHIVE = ARCHIVE.with_suffix(ARCHIVE.suffix + ".part")

ATLAS_URL = (
    "https://www.nitrc.org/frs/download.php/13713/"
    "UNC-Infant-Cortical-Surface-Atlas_V1.11.zip"
)

EXPECTED_AGES = (1, 3, 6, 9, 12, 18, 24, 36, 48, 60, 72)
HEMISPHERES = ("lh", "rh")

SURFACE_FILES = (
    "OuterSurf",
    "MiddleSurf",
    "InnerSurf",
    "InflateSurf",
    "SphereSurf",
)

MANIFEST_SURFACES = (
    "outer",
    "middle",
    "inner",
    "inflated",
    "sphere",
)

VTK_NAME = re.compile(
    r"^SparseAtlas_(\d{1,2})_(lh|rh)\."
    r"(OuterSurf|MiddleSurf|InnerSurf|InflateSurf|SphereSurf)\.vtk$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Paths and validation
# ---------------------------------------------------------------------

def _expected_surface_path(age, hemi, surface):
    """Return the canonical path using zero-padded age directories."""

    filename = f"SparseAtlas_{age:02d}_{hemi}.{surface}.vtk"
    return DATA_DIR / f"{age:02d}" / filename


def _missing_surfaces():
    """Return all missing or empty required atlas surfaces."""

    missing = []

    for age in EXPECTED_AGES:
        for hemi in HEMISPHERES:
            for surface in SURFACE_FILES:
                path = _expected_surface_path(age, hemi, surface)

                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(path)

    return missing


def _data_is_complete():
    """True if all 110 required VTK surfaces are present."""
    return not _missing_surfaces()


def _manifest_is_usable():
    """Check that atlas.json references all expected existing surfaces."""

    if not MANIFEST.is_file():
        return False

    try:
        atlas = json.loads(MANIFEST.read_text(encoding="utf-8"))

        if not isinstance(atlas, dict):
            return False

        for age in EXPECTED_AGES:
            entry = atlas.get(str(age))

            if not isinstance(entry, dict):
                return False

            for hemi in ("left", "right"):
                surfaces = entry.get(hemi)

                if not isinstance(surfaces, dict):
                    return False

                for surface in MANIFEST_SURFACES:
                    relative = surfaces.get(surface)

                    if not relative:
                        return False

                    path = ROOT / relative

                    if not path.is_file() or path.stat().st_size == 0:
                        return False

        return True

    except (OSError, ValueError, TypeError):
        return False


# ---------------------------------------------------------------------
# Archive validation
# ---------------------------------------------------------------------

def _archive_contains_expected_surfaces(archive_path):
    """Check whether the ZIP contains all 110 expected surfaces."""

    if not archive_path.is_file():
        return False

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:

            found = set()
            surface_lookup = {
                surface.lower(): surface
                for surface in SURFACE_FILES
            }

            for info in archive.infolist():

                if info.is_dir():
                    continue

                match = VTK_NAME.match(Path(info.filename).name)

                if not match:
                    continue

                age = int(match.group(1))
                hemi = match.group(2).lower()
                surface = surface_lookup.get(
                    match.group(3).lower()
                )

                if age not in EXPECTED_AGES or surface is None:
                    continue

                found.add((age, hemi, surface))

            expected = {
                (age, hemi, surface)
                for age in EXPECTED_AGES
                for hemi in HEMISPHERES
                for surface in SURFACE_FILES
            }

            return expected.issubset(found)

    except (OSError, zipfile.BadZipFile):
        return False


# ---------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------

def _download(progress):
    """Download the UNC atlas, reusing a valid cached archive."""

    if _archive_contains_expected_surfaces(ARCHIVE):
        progress.setValue(65)
        progress.setLabelText(
            "Using the previously downloaded atlas archive…"
        )
        QApplication.processEvents()
        return

    ARCHIVE.unlink(missing_ok=True)
    TEMP_ARCHIVE.unlink(missing_ok=True)

    request = urllib.request.Request(
        ATLAS_URL,
        headers={
            "User-Agent": "InfantBrainViewer/1.0 (atlas installer)"
        },
    )

    downloaded = 0

    try:
        with urllib.request.urlopen(request, timeout=45) as response:

            total = int(response.headers.get("Content-Length") or 0)

            with TEMP_ARCHIVE.open("wb") as out:

                while True:

                    if progress.wasCanceled():
                        raise InterruptedError(
                            "Atlas download cancelled by user."
                        )

                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    out.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        percent = min(
                            65,
                            int(downloaded * 65 / total),
                        )

                        progress.setValue(percent)
                        progress.setLabelText(
                            "Downloading UNC infant cortical atlas…\n"
                            f"{downloaded / (1024**2):,.0f} MB / "
                            f"{total / (1024**2):,.0f} MB"
                        )
                    else:
                        progress.setValue(
                            min(64, progress.value() + 1)
                        )

                        progress.setLabelText(
                            "Downloading UNC infant cortical atlas… "
                            f"{downloaded / (1024**2):,.0f} MB"
                        )

                    QApplication.processEvents()

        if downloaded < 1024 * 1024:
            raise RuntimeError(
                "The NITRC download appears incomplete."
            )

        if not _archive_contains_expected_surfaces(TEMP_ARCHIVE):
            raise RuntimeError(
                "The downloaded archive is invalid or does not "
                "contain all expected UNC atlas surfaces."
            )

        TEMP_ARCHIVE.replace(ARCHIVE)

    except Exception:
        TEMP_ARCHIVE.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------

def _extract_surfaces(progress):
    """Extract only the five required surface types."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not _archive_contains_expected_surfaces(ARCHIVE):
        raise RuntimeError(
            "The atlas ZIP archive is invalid or incomplete."
        )

    surface_lookup = {
        surface.lower(): surface
        for surface in SURFACE_FILES
    }

    with zipfile.ZipFile(ARCHIVE, "r") as archive:

        matches = []

        for info in archive.infolist():

            if info.is_dir():
                continue

            match = VTK_NAME.match(Path(info.filename).name)

            if not match:
                continue

            age = int(match.group(1))
            hemi = match.group(2).lower()
            surface = surface_lookup.get(
                match.group(3).lower()
            )

            if (
                age not in EXPECTED_AGES
                or hemi not in HEMISPHERES
                or surface is None
            ):
                continue

            matches.append((info, age, hemi, surface))

        total_bytes = sum(
            info.file_size
            for info, *_ in matches
        ) or 1

        extracted_bytes = 0

        for index, (info, age, hemi, surface) in enumerate(
            matches,
            start=1,
        ):

            if progress.wasCanceled():
                raise InterruptedError(
                    "Atlas extraction cancelled by user."
                )

            # IMPORTANT: zero-padded directory names.
            destination = _expected_surface_path(
                age,
                hemi,
                surface,
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = destination.with_suffix(
                destination.suffix + ".part"
            )

            try:
                with archive.open(info, "r") as src:
                    with temporary.open("wb") as dst:

                        while True:

                            if progress.wasCanceled():
                                raise InterruptedError(
                                    "Atlas extraction cancelled by user."
                                )

                            chunk = src.read(1024 * 1024)

                            if not chunk:
                                break

                            dst.write(chunk)
                            extracted_bytes += len(chunk)

                            percent = 65 + int(
                                30 * extracted_bytes / total_bytes
                            )

                            progress.setValue(min(95, percent))

                            progress.setLabelText(
                                "Extracting cortical surfaces…\n"
                                f"{index}/{len(matches)} files · "
                                f"age {age} months · "
                                f"{hemi.upper()} {surface}"
                            )

                            QApplication.processEvents()

                if temporary.stat().st_size == 0:
                    raise RuntimeError(
                        f"Extracted file is empty: {destination}"
                    )

                temporary.replace(destination)

            finally:
                temporary.unlink(missing_ok=True)

    missing = _missing_surfaces()

    if missing:
        formatted = "\n".join(
            str(path.relative_to(ROOT))
            for path in missing
        )

        raise RuntimeError(
            "Atlas extraction is incomplete. Missing files:\n"
            + formatted
        )


# ---------------------------------------------------------------------
# Main installer
# ---------------------------------------------------------------------

def ensure_atlas_ready(parent=None):
    """
    Ensure the UNC infant cortical atlas is installed and indexed.

    Returns:
        True  - atlas is ready.
        False - installation was cancelled.

    Existing complete data is never downloaded again.
    """

    print("[Atlas installer] Script:", Path(__file__).resolve())
    print("[Atlas installer] ROOT:", ROOT)
    print("[Atlas installer] DATA_DIR:", DATA_DIR)

    # Fast path: data and manifest are both valid.
    if _data_is_complete() and _manifest_is_usable():
        print("[Atlas installer] Atlas already installed. Skipping setup.")
        return True

    progress = QProgressDialog(
        "Preparing the UNC infant cortical atlas…",
        "Cancel",
        0,
        100,
        parent,
    )

    progress.setWindowTitle("First-time atlas setup")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setValue(0)

    QApplication.processEvents()

    try:

        missing = _missing_surfaces()

        # ---------------------------------------------------------
        # All 110 files exist: rebuild only the manifest.
        # ---------------------------------------------------------
        if not missing:

            print(
                "[Atlas installer] All 110 surfaces found. "
                "Download skipped."
            )

            progress.setValue(90)
            progress.setLabelText(
                "Atlas files already exist. Rebuilding index…"
            )

            QApplication.processEvents()

        # ---------------------------------------------------------
        # Some files are missing: download and extract.
        # ---------------------------------------------------------
        else:

            print(
                f"[Atlas installer] {len(missing)} required "
                "files missing."
            )

            for path in missing[:10]:
                print(
                    "[Atlas installer] Missing:",
                    path.relative_to(ROOT),
                )

            if len(missing) > 10:
                print(
                    f"[Atlas installer] ... and "
                    f"{len(missing) - 10} more."
                )

            progress.setLabelText(
                f"Missing {len(missing)} atlas files. "
                "Downloading UNC infant cortical atlas…"
            )

            QApplication.processEvents()

            _download(progress)

            progress.setValue(66)
            progress.setLabelText(
                "Download complete. Extracting required VTK surfaces…"
            )

            QApplication.processEvents()

            _extract_surfaces(progress)

        # ---------------------------------------------------------
        # Generate manifest.
        # ---------------------------------------------------------
        progress.setValue(96)
        progress.setLabelText(
            "Building atlas.json index…"
        )

        QApplication.processEvents()

        build_manifest()

        if not _data_is_complete():
            raise RuntimeError(
                "Atlas validation failed: some required surfaces "
                "are still missing."
            )

        if not _manifest_is_usable():
            raise RuntimeError(
                "Atlas preparation finished, but atlas.json "
                "is not usable."
            )

        progress.setValue(100)
        progress.setLabelText(
            "Atlas is ready. Starting the 3D viewer…"
        )

        QApplication.processEvents()

        ARCHIVE.unlink(missing_ok=True)
        TEMP_ARCHIVE.unlink(missing_ok=True)

        progress.close()

        print("[Atlas installer] Installation successful.")

        return True

    except InterruptedError:

        TEMP_ARCHIVE.unlink(missing_ok=True)
        progress.close()

        print("[Atlas installer] Installation cancelled.")

        return False

    except Exception:

        progress.close()
        raise

