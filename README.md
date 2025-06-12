# Audio Map

Audio sample exploration and analysis tool using monome controllers

## Overview

Audio Map is an interactive tool for exploring large collections of audio samples using dimensionality-reduction techniques and an intuitive monome Arc controller interface. It processes audio files, extracts features, and creates a 2D map that can be navigated in real-time for audio discovery and performance.

## Features

- **Audio Processing**: Converts audio files to mono at 22,050 Hz and slices them into configurable segments
- **Feature Extraction**: Extracts 28-dimensional feature vectors including MFCC, chroma, and spectral features
- **Dimensionality Reduction**: Uses UMAP to create intuitive 2D audio maps
- **Real-time Navigation**: Interactive exploration using monome Arc controller
- **Flexible Playback**: Pitch-preserving time-stretch with variable playback rates (0.25x to 4.0x)

## Requirements

- Python 3.8+
- monome Arc controller
- Audio files (WAV, AIFF, MP3, etc.)

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Building an Audio Map

Process a folder of audio files to create an explorable map:

```bash
audio-map build ~/Music/samples
```

Options:
- `--slice-duration, -s`: Duration of each audio slice in seconds (default: 0.5)
- `--max-points, -m`: Maximum number of points in the map
- `--workers, -w`: Number of parallel workers for processing
- `--fast, -f`: Use fast mode with simplified features

### Running the Navigator

Start the interactive audio map navigator:

```bash
audio-map run
```

Options:
- `--device, -d`: Specify CoreAudio output device ID

### Arc Controller Interface

The monome Arc provides intuitive control:

- **Ring 0**: X-axis navigation
- **Ring 1**: Y-axis navigation  
- **Ring 2**: Zoom level control
- **Ring 3**: Playback rate control (0.25x to 4.0x speed)
- **Push button**: Toggle playback mode (single/morph)

## Examples

### Basic Usage
```bash
# Build a map from your sample library
audio-map build ~/Music/samples --slice-duration 1.0

# Start exploring with default audio device
audio-map run

# Use specific audio device
audio-map run --device 5
```

### Arc Test Example

The `examples/arc.py` file demonstrates basic Arc connectivity and control:

```bash
python examples/arc.py
```

## Development

### Code Quality Tools

- **Black**: Code formatting (`black .`)
- **Ruff**: Linting (`ruff check .`)
- **MyPy**: Type checking (`mypy .`)
- **Pytest**: Testing (`pytest`)

### Project Structure

```
audio_map/
├── __init__.py         # Package initialization
├── cli.py             # Command-line interface
├── build.py           # Audio processing and map building
└── runtime.py         # Real-time navigation and playback

examples/
└── arc.py             # Basic Arc controller test

audio_files/           # Sample audio files
└── *.wav
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and code quality checks
5. Submit a pull request

## Links

- [Homepage](https://github.com/artfwo/pymonome)
- [Issues](https://github.com/artfwo/pymonome/issues)
- [monome Documentation](https://monome.org/)