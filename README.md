# Infant Brain 3D Visualizer

A Python desktop application for visualizing infant brain surfaces, scalp and EEG electrode positions, and brain tractography in a shared 3D view. The viewer supports age-specific infant templates and can optionally compute and display effective connectivity from EEG recordings.

## Features

- Interactive 3D visualization of cortical surfaces and tractography.
- Infant scalp and skull geometry with EEG electrode locations.
- Age/template selection for infant head geometry (from birth to 2 years in the supplied templates; the viewer's age control extends to 72 months).
- Overlay of electrodes and tractography for spatial inspection.
- Optional effective-connectivity analysis using **Granger causality** or **transfer entropy** from EEG data.
- Loading of local atlas and surface files from the repository.

## Screenshots

### EEG electrode locations

The scalp and EEG electrode positions are shown in 3D.

![EEG electrodes on the infant scalp](EEG_electrodes.png)

### Brain tractography and electrode overlay

Example screenshot showing the brain tractography together with the EEG electrode overlay.

![Brain tractography overlaid with EEG electrodes](tracts.png)

> Keep `EEG_electrodes.png` and `tracts.png` in the repository root (alongside this README) so GitHub can display the images.

## Repository files

- `infant_connect.py` — main interactive 3D viewer and application entry point.
- `generate_cortex.py` — prepares/generates the cortical surface files used by the viewer.
- `fetch_tract_atlas.py` — fetches the tractography atlas used for the tract overlay.
- `atlas.json` — atlas manifest/configuration used by the viewer.
- `tract_atlases/` — directory for tractography atlas data.
- `effconn.py` — standalone effective-connectivity implementation (optional alternative to installing `effconnpy`).
- `requirements.txt` — Python dependencies for the viewer.
- `EEG_electrodes.png` — screenshot/figure showing EEG electrode positions.
- `tracts.png` — screenshot/figure showing the tractography and electrode overlay.

The cortical surfaces should be prepared with `generate_cortex.py`, and the tractography atlas should be retrieved with `fetch_tract_atlas.py` before launching the viewer. The scalp, skull, and electrode/montage template data are downloaded automatically by the viewer when needed and cached locally.

## Requirements

- Python 3.10 or newer is recommended.
- A desktop environment capable of running Qt and rendering VTK/PyVista 3D scenes.
- Python packages listed in `requirements.txt`.

## Installation

Clone the repository and enter its directory:

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-DIRECTORY>
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare the data

Run the preparation scripts from the repository root:

```bash
python generate_cortex.py
python fetch_tract_atlas.py
```

These scripts prepare the cortical surfaces and fetch the tractography atlas, respectively. Check their console output for any download or preparation errors. Make sure the resulting files are placed where the viewer expects them, and that `atlas.json` and `tract_atlases/` are present.

The viewer downloads the required infant scalp, skull, and electrode/montage template data on demand. An internet connection is needed the first time each template is requested. Downloaded template files are cached locally for later runs.

## Run the viewer

From the repository root, run:

```bash
python infant_connect.py
```

Run the command from the repository root after completing the preparation steps above.

Use the viewer controls to toggle the scalp/electrodes, cortical surfaces, and tractography layers. Available controls and displayed options may depend on the atlas and surface files that have been prepared.

## Effective connectivity

The viewer can compute and display effective connectivity from an EEG recording using **Granger causality** or **transfer entropy**.

There are two ways to provide the implementation:

1. **Install `effconnpy`** from [piwheels](https://www.piwheels.org/project/effconnpy/):

   ```bash
   python -m pip install effconnpy
   ```

   The installed package must expose `compute_effective_connectivity` from the `effconn` module, as expected by the viewer.

2. **Use the standalone implementation** included in this repository as `effconn.py`. Keep `effconn.py` alongside the main viewer script so that `from effconn import compute_effective_connectivity` can resolve locally. If you choose this option, you generally do not need to install a separate package providing the same module.

To run the analysis, load an EEG recording through the viewer and enable the effective-connectivity control. The available methods are Granger causality and transfer entropy. The recording's channel names need to correspond to the electrode labels supported by the selected montage/template for the electrode positions to be fitted correctly.

## Data and attribution

The infant scalp/skull and montage templates are downloaded from the public [infant template paper repository](https://github.com/christian-oreilly/infant_template_paper/) when required. The cortical surfaces and tractography atlas are prepared or fetched by the scripts and should retain the licensing and attribution requirements of their original sources. Check the source projects and data licences before redistribution.

## Troubleshooting

- **Missing surface or atlas files:** run `generate_cortex.py` and `fetch_tract_atlas.py` from the repository root and check their output.
- **Missing Python module:** activate the intended environment and reinstall dependencies with `python -m pip install -r requirements.txt`.
- **Qt/VTK display problems:** run the viewer in a desktop session with working graphics support; headless environments may not support the interactive window.
- **Effective-connectivity import error:** ensure either the repository's `effconn.py` is alongside the viewer or the compatible `effconnpy` package is installed.
- **Electrodes do not align with the recording:** check the EEG channel names and the montage/template selected in the viewer.

## Citation

If you use this software in research, please cite this repository and the original sources of any cortical surfaces, infant templates, and tractography atlas data used. Add a formal software citation (for example, a Zenodo DOI) here when one is available.

## Licence

**The repository's software licence must be specified in a `LICENSE` file before public release.** Do not assume that third-party templates, cortical surfaces, or tractography atlas data are covered by the same licence as this code. Check each source project's terms and preserve any required attribution and usage restrictions.

If you intend to release your own code under a permissive licence such as MIT, add the corresponding `LICENSE` file and confirm that this is compatible with all code you authored and redistributed. The licence for this repository does not automatically relicense external datasets or assets.
