"""On-demand downloader for the NABA-800FC tractography atlas."""
from pathlib import Path
import hashlib
import json
import urllib.parse
import urllib.request
import zipfile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressDialog

ROOT = Path(__file__).resolve().parent
TRACT_DIR = ROOT / "tract_atlases"
RECORD_ID = "18227691"
ARCHIVE_NAME = "NABA-800FC.zip"
ARCHIVE = ROOT / f".{ARCHIVE_NAME}.download"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK_SIZE = 1024 * 1024


def _request(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "InfantBrainViewer/1.0 (tractography atlas installer)"}
    )
    return urllib.request.urlopen(request, timeout=60)


def _record_file():
    """Resolve the named archive and its checksum from Zenodo's record API."""
    with _request(API_URL) as response:
        record = json.loads(response.read().decode("utf-8"))
    for item in record.get("files", []):
        if item.get("key") == ARCHIVE_NAME:
            links = item.get("links", {})
            url = links.get("download") or links.get("self")
            if not url:
                break
            return url, item.get("checksum"), int(item.get("size") or 0)
    raise RuntimeError(
        f"Could not find {ARCHIVE_NAME} in Zenodo record {RECORD_ID}. "
        "The record contents may have changed."
    )


def _download(progress):
    url, checksum, expected_size = _record_file()
    temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".part")
    digest = hashlib.md5() if checksum and checksum.startswith("md5:") else None
    downloaded = 0
    try:
        with _request(url) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length") or expected_size or 0)
            while True:
                if progress.wasCanceled():
                    raise InterruptedError("Tractography download cancelled.")
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if digest:
                    digest.update(chunk)
                if total:
                    percent = min(55, int(downloaded * 55 / total))
                    progress.setValue(percent)
                    progress.setLabelText(
                        "Downloading NABA-800FC tractography atlas…\n"
                        f"{downloaded / 1024**2:,.0f} / {total / 1024**2:,.0f} MB"
                    )
                else:
                    progress.setValue(min(54, progress.value() + 1))
                    progress.setLabelText(
                        f"Downloading NABA-800FC tractography atlas… "
                        f"{downloaded / 1024**2:,.0f} MB"
                    )
                QApplication.processEvents()
        if expected_size and downloaded != expected_size:
            raise RuntimeError(
                f"Download size mismatch: received {downloaded} bytes; "
                f"Zenodo reports {expected_size} bytes."
            )
        if digest and digest.hexdigest().lower() != checksum.split(":", 1)[1].lower():
            raise RuntimeError("The downloaded NABA-800FC archive failed its MD5 checksum.")
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError("The downloaded NABA-800FC file is not a valid ZIP archive.")
        temporary.replace(ARCHIVE)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _extract(progress):
    TRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        members = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            lower = name.lower()
            if lower.startswith("cluster_") and lower.endswith(".vtp"):
                members.append((info, name))
        if not members:
            raise RuntimeError(
                f"No cluster_*.vtp files were found inside {ARCHIVE_NAME}. "
                "The archive layout may have changed."
            )

        total_bytes = sum(info.file_size for info, _ in members) or 1
        extracted = 0
        for index, (info, name) in enumerate(members, start=1):
            if progress.wasCanceled():
                raise InterruptedError("Tractography extraction cancelled.")
            destination = TRACT_DIR / name
            temporary = destination.with_suffix(destination.suffix + ".part")
            with archive.open(info, "r") as source, temporary.open("wb") as output:
                while True:
                    if progress.wasCanceled():
                        raise InterruptedError("Tractography extraction cancelled.")
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    extracted += len(chunk)
                    progress.setValue(55 + int(43 * extracted / total_bytes))
                    progress.setLabelText(
                        "Extracting tractography cluster files…\n"
                        f"{index}/{len(members)} files · {name}"
                    )
                    QApplication.processEvents()
            temporary.replace(destination)

    installed = list(TRACT_DIR.glob("cluster_*.vtp"))
    if not installed:
        raise RuntimeError("Extraction completed, but no cluster_*.vtp files were installed.")
    return len(installed)


def ensure_tract_atlas_ready(parent=None):
    """Download and extract the tractography atlas on first use; False means cancelled."""
    TRACT_DIR.mkdir(parents=True, exist_ok=True)
    # If files were already installed manually, reuse them. If a prior
    # interrupted setup left its archive behind, resume extraction instead of
    # mistaking a partial set of files for a completed installation.
    if any(TRACT_DIR.glob("cluster_*.vtp")) and not ARCHIVE.exists():
        return True

    progress = QProgressDialog(
        "Preparing NABA-800FC tractography atlas…", "Cancel", 0, 100, parent
    )
    progress.setWindowTitle("First-time tractography setup")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setValue(0)
    QApplication.processEvents()
    try:
        if not ARCHIVE.is_file() or not zipfile.is_zipfile(ARCHIVE):
            progress.setLabelText("Connecting to Zenodo and downloading NABA-800FC.zip…")
            QApplication.processEvents()
            _download(progress)
        else:
            progress.setValue(55)
            progress.setLabelText("Using previously downloaded NABA-800FC archive…")
            QApplication.processEvents()

        count = _extract(progress)
        progress.setValue(100)
        progress.setLabelText(f"Tractography atlas ready ({count} cluster files).")
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
