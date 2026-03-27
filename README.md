# Raman Spectrum Analyser

Streamlit app for Raman spectrum analysis with database matching, functional group interpretation, report export, and support for both calibrated Raman CSV files and raw CCD-style CSV files.

## Included Files

- Application code in `app.py`, `core/`, `ui/`, `utils/`, and `tools/`
- Example databases in `data/`
- Example calibrated Raman spectra in `data/Synthetic_Rover_Spectra.csv`
- Example raw CCD files in `data/ccd_data.csv` and `data/synthetic_raw_ccd_3000.csv`

## Requirements

- Python 3.10+
- `pip`

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Add your Gemini API key if you want AI features.
5. Start the Streamlit app.

```bash
git clone <your-repo-url>
cd raman_analyzer

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and set GEMINI_API_KEY

streamlit run app.py
```

If you do not want AI features, you can skip the secrets file and leave the API key blank in the app.

## API Key

Add your Gemini key in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

The real secrets file is ignored by git. Only `.streamlit/secrets.toml.example` should be committed.

## Input Formats

### 1. Raman CSV

Use this when your file is already calibrated.

```csv
Raman Shift,Intensity
100,245.3
102,247.1
104,252.8
```

- First column: Raman shift / wavenumber
- Remaining columns: one or more intensity series

### 2. Raw CCD CSV

Use this when your file contains per-frame ADC values for many CCD pixels.

```csv
timestamp,frame,pixel_0_adc,pixel_1_adc,pixel_2_adc
1774616024,1,753,1879,2576
1774616024,2,825,1940,2653
```

- Each row is a frame
- Each `pixel_*_adc` column is a CCD pixel
- The app collapses frames into one spectrum, converts the pixel span into Raman shift, and then runs the normal analysis pipeline

## Raw CCD Calibration Modes

The sidebar supports two raw CCD calibration modes:

### Estimated Raman Range

Use this when you do not have real instrument coefficients yet.

- `Laser wavelength (nm)`
- `Estimated Raman start (cm⁻¹)`
- `Estimated Raman end (cm⁻¹)`

The app spreads the CCD pixel span across that Raman range and generates a Raman-style intermediate CSV before analysis.

### Manual Wavelength Calibration

Use this when you know instrument calibration values.

- `Laser wavelength (nm)`
- `Reference pixel`
- `Reference wavelength (nm)`
- `Wavelength step (nm/pixel)`

The app converts pixel index to wavelength, then wavelength to Raman shift.

## What the App Does

- Preprocesses spectra with despiking, baseline correction, and normalisation
- Detects peaks and estimates peak shape / width
- Matches spectra against uploaded JSON databases
- Assigns functional groups from uploaded rule files
- Compares uploaded spectra by similarity
- Exports CSV results and PDF reports

## Database Files

You can upload your own JSON databases in the sidebar, or use the sample files in `data/`.

- `data/sample_database.json`
- `data/rruff_database.json`
- `data/functional_groups.json`

## Running the App

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.
