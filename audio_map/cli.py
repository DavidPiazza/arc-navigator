"""Command-line interface for Audio arc-navigator.

Provides 'build' and 'run' subcommands for creating and navigating audio mapes.
"""
from __future__ import annotations

import typer
from pathlib import Path


app = typer.Typer(
    name="audio_navigator",
    help="Audio arc-navigator - Transform and explore audio collections with monome Arc",
    add_completion=False,
    rich_markup_mode="rich"
)


@app.command()
def build(
    folder: Path = typer.Argument(
        ..., 
        help="Path to folder containing audio files (supports WAV, AIFF, MP3, etc.)",
        metavar="FOLDER"
    ),
    slice_duration: float = typer.Option(
        0.5,
        "--slice-duration",
        "-s",
        help="Duration of each audio slice in seconds",
        min=0.01,
        max=10.0
    ),
    max_points: int = typer.Option(
        None,
        "--max-points",
        "-m",
        help="Maximum number of points to include in the map (random sampling if exceeded)",
        min=100
    ),
    num_workers: int = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers for processing (default: auto-detect CPU count)",
        min=1
    ),
    fast_mode: bool = typer.Option(
        False,
        "--fast",
        "-f",
        help="Use fast mode with simplified features (10 dims instead of 28)"
    )
):
    """
    Build an audio map from a folder of audio files.
    
    This command recursively scans the specified folder for audio files, processes each file by:
    - Converting to mono at 22,050 Hz sample rate
    - Slicing into non-overlapping segments (default: 0.5 seconds, configurable)
    - Extracting 28-dimensional feature vectors (MFCC, chroma, spectral features)
    - Reducing dimensionality to 2D coordinates using UMAP
    - Saving the resulting map as 'map.pkl' in the current directory
    
    Examples:
        audio_navigator build ~/Music/samples
        audio_navigator build ~/Music/samples --slice-duration 1.0
    """
    # Validate that the folder exists and is a directory
    if not folder.exists():
        typer.echo(f"Error: {folder} does not exist", err=True)
        raise typer.Exit(1)
    
    if not folder.is_dir():
        typer.echo(f"Error: {folder} is not a directory", err=True)
        raise typer.Exit(1)
    
    # Import and call the build module
    try:
        from . import build as build_module
        build_module.build_map(
            folder, 
            slice_duration=slice_duration, 
            max_points=max_points,
            num_workers=num_workers,
            fast_mode=fast_mode
        )
    except ImportError:
        typer.echo("Error: Build module not yet implemented", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error during build: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def run(
    device: int = typer.Option(
        None, 
        "--device", 
        "-d", 
        help="CoreAudio output device ID (use sounddevice.query_devices() to list)",
        metavar="ID"
    ),
    neighbors: int = typer.Option(
        8,
        "--neighbors",
        "-n",
        help="Number of nearest neighbors to use in morph mode (default: 8)",
        min=1,
        max=32
    ),
    loop: bool = typer.Option(
        False,
        "--loop",
        "-l",
        help="Enable continuous looping of audio slices. If omitted, slices play once.",
        is_flag=True,
    )
):
    """
    Start the interactive audio map navigator.
    
    Launches the runtime navigator that connects to a monome Arc controller for real-time
    exploration of the audio map. The Arc provides:
    
    - Ring 0: X-axis navigation
    - Ring 1: Y-axis navigation  
    - Ring 2: Zoom level control
    - Ring 3: Playback rate control (0.25x to 4.0x speed, varispeed effect affecting pitch)
    - Push button: Toggle playback mode (single/morph)
    
    Audio output uses CoreAudio for low-latency playback. If no device is specified,
    the system default output device will be used.
    
    Examples:
        audio_navigator run                    # Use default audio device
        audio_navigator run --device 5        # Use specific CoreAudio device
        audio_navigator run --neighbors 12    # Use 12 nearest neighbors in morph mode
    
    Note: Requires 'map.pkl' file in the current directory (created with 'build' command)
    and a connected monome Arc controller.
    """
    # Validate audio device if specified
    if device is not None:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            if device < 0 or device >= len(devices):
                typer.echo(f"Error: Device ID {device} not found", err=True)
                typer.echo("Available devices:")
                for i, dev in enumerate(devices):
                    typer.echo(f"  {i}: {dev['name']}")
                raise typer.Exit(1)
                
            # Verify device has output capabilities
            device_info = devices[device]
            if device_info['max_output_channels'] == 0:
                typer.echo(f"Error: Device {device} ({device_info['name']}) has no output channels", err=True)
                raise typer.Exit(1)
                
        except ImportError:
            typer.echo("Error: sounddevice module not available", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Error querying audio devices: {e}", err=True)
            raise typer.Exit(1)
    
    # Check for map.pkl file
    map_path = Path("map.pkl")
    if not map_path.exists():
        typer.echo("Error: map.pkl not found in current directory", err=True)
        typer.echo("Run 'audio_navigator build <folder>' first to create an map", err=True)
        raise typer.Exit(1)
    
    # Import and call the runtime module
    try:
        from . import runtime as runtime_module
        runtime_module.run_navigator(str(map_path), device, neighbors, loop)
    except ImportError:
        typer.echo("Error: Runtime module not yet implemented", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error starting navigator: {e}", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main() 