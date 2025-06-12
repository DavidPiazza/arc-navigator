# Audio arc-navigator

An interactive tool for exploring large collections of audio samples using dimensionality-reduction techniques and a monome Arc controller. Transform any folder of audio files into a navigable 2D "sound map" that can be explored in real-time.

## Overview

Audio arc-navigator consists of two main components:
1. **Build Pipeline**: Processes audio files, extracts features, and creates a 2D audio map
2. **Runtime Navigator**: Interactive exploration using a monome Arc controller

## Requirements

- macOS 13+ (Apple Silicon or Intel)
- Python ≥ 3.10
- monome Arc controller
- serialosc (for monome device communication)

## Installation

1. Ensure you have Python 3.10+ installed (via Homebrew recommended)
2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Dependencies

The project requires the following Python packages:
- `librosa` ≥ 0.10 (audio feature extraction)
- `numpy` (numerical computing)
- `scipy` (scientific computing)
- `scikit-learn` (machine learning utilities)
- `umap-learn` ≥ 0.5 (dimensionality reduction)
- `sounddevice` (audio playback)
- `torchaudio` (audio file loading)
- `typer` (CLI framework)
- `pymonome` (monome device communication)

## Usage

### Building an Audio Map

Process a folder of audio files to create a navigable sound map:

```bash
audio_navigator build /path/to/audio/folder
```

This command will:
- Scan the folder for audio files (WAV, AIFF, MP3, etc.)
- Resample all audio to 22,050 Hz mono
- Slice each file into 1-second non-overlapping segments
- Extract 29-dimensional feature vectors (13 MFCC + 12 chroma + spectral features)
- Reduce dimensionality to 2D using UMAP
- Save the resulting map as `map.pkl`

### Running the Navigator

Launch the interactive Arc controller interface:

```bash
audio_navigator run
```

Optional parameters:
- `--device ID`: Specify CoreAudio output device ID

### Arc Controller Interface

The Arc controller provides intuitive navigation of your audio map:

- **Ring 0 (Left)**: X-axis navigation
- **Ring 1**: Y-axis navigation  
- **Ring 2**: Zoom level control
- **Ring 3 (Right)**: Playback mode selection
- **Push Button**: Reset cursor to center (0, 0)

#### LED Feedback

- **Ring 0**: Shows point density using logarithmic brightness, with cursor position at full brightness
- **Ring 1**: Displays cursor angle
- **Ring 2**: Indicates zoom level
- **Ring 3**: Shows current playback mode (16-LED block)

#### Playback Modes

- **Mode 0**: Play single nearest audio sample
- **Mode 1**: Cycle through 8 nearest samples

## Technical Details

### Feature Extraction

Audio files are processed using the following pipeline:
1. Conversion to mono, 22,050 Hz
2. Slicing into 1-second segments
3. Feature extraction per segment:
   - 13 Mel-frequency cepstral coefficients (MFCC)
   - 12 chroma features
   - Spectral centroid
   - Spectral bandwidth
   - Zero crossing rate

### Dimensionality Reduction

UMAP (Uniform Manifold Approximation and Projection) is used with these parameters:
- `n_neighbors=30`
- `min_dist=0.1` 
- `metric='cosine'`
- Output scaled to [-1, 1] range

### Performance Targets

- Build time: ≤ 3 minutes for 2,000 files (M2 Max)
- Audio latency: < 50ms
- Peak memory usage: ≤ 4GB during build

## Project Structure

```
audio_map/
├── __init__.py       # Package initialization
├── cli.py           # Command-line interface
├── build.py         # Feature extraction and UMAP processing
├── runtime.py       # Arc navigator application
└── README.md        # This file
```

## Development

For development setup:

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install in development mode:
   ```bash
   pip install -e .
   ```

## License

This project is part of the pymonome library and follows the same MIT license.