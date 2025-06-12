"""Build utilities for Audio Map.

This module includes functions to preprocess audio collections, extract features,
run UMAP dimensionality reduction, and save artifacts for interactive runtime
exploration.
"""

import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import multiprocessing as mp
from functools import partial
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
from sklearn.preprocessing import StandardScaler
import umap
import typer
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=(UserWarning))
warnings.filterwarnings("ignore", category=(FutureWarning))

def load_and_slice_audio(
    file_path: str, 
    target_sr: int = 22050, 
    slice_duration: float = 0.5
) -> List[np.ndarray]:
    """Load an audio file, convert to mono, resample, and slice into segments.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (default: 22050 Hz)
        slice_duration: Duration of each slice in seconds (default: 1.0)
        
    Returns:
        List of audio slices as numpy arrays
        
    Raises:
        RuntimeError: If the audio file cannot be loaded
        ValueError: If invalid parameters are provided
        FileNotFoundError: If the audio file doesn't exist
    """
    # Input validation
    if slice_duration <= 0:
        raise ValueError("Slice duration must be positive")
    if target_sr <= 0:
        raise ValueError("Target sample rate must be positive")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    try:
        # Load audio file with torchaudio
        waveform, original_sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if original_sr != target_sr:
            resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
            waveform = resampler(waveform)
            
        # Convert to numpy array and squeeze to 1D
        audio = waveform.squeeze().numpy()
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")
        
    # Calculate slice length in samples
    slice_length = int(target_sr * slice_duration)
    
    # Slice audio into non-overlapping segments
    slices = []
    for i in range(0, len(audio), slice_length):
        slice_end = min(i + slice_length, len(audio))
        audio_slice = audio[i:slice_end]
        
        # Only keep slices that are the full duration
        if len(audio_slice) == slice_length:
            slices.append(audio_slice)
            
    return slices


def is_silent(audio_slice: np.ndarray, threshold_db: float = -40.0) -> bool:
    """Check if an audio slice is silent based on RMS level.
    
    Args:
        audio_slice: 1D numpy array containing audio samples
        threshold_db: Silence threshold in decibels (default: -40.0 dB)
        
    Returns:
        True if the slice is considered silent, False otherwise
    """
    # Fast RMS calculation using NumPy
    rms = np.sqrt(np.mean(np.square(audio_slice)))
    
    # Early return for very quiet signals
    if rms < 1e-10:
        return True
    
    # Convert to dB (avoid log(0))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    return rms_db < threshold_db


def find_audio_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """Find all audio files in a directory and its subdirectories.
    
    Args:
        directory: Path to the directory to search
        extensions: List of file extensions to search for (default: common audio formats)
        
    Returns:
        List of paths to audio files
        
    Raises:
        ValueError: If directory doesn't exist or is not a directory
        TypeError: If extensions is not a list
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.aiff', '.aif', '.m4a', '.ogg']
    elif not isinstance(extensions, list):
        raise TypeError("Extensions must be a list of strings")
    
    # Validate that all extensions start with a dot
    for ext in extensions:
        if not isinstance(ext, str):
            raise TypeError("All extensions must be strings")
        if not ext.startswith('.'):
            raise ValueError(f"Extension must start with a dot: {ext}")
    
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    audio_files = []
    try:
        for ext in extensions:
            # Use glob to find files with each extension (case-insensitive)
            pattern = f"**/*{ext}"
            audio_files.extend(directory_path.glob(pattern))
            # Also search for uppercase extensions
            pattern = f"**/*{ext.upper()}"
            audio_files.extend(directory_path.glob(pattern))
    except Exception as e:
        raise RuntimeError(f"Error searching for audio files: {e}")
    
    # Remove duplicates and convert to strings
    audio_files = list(set(str(f) for f in audio_files))
    audio_files.sort()
    
    return audio_files


def extract_features_fast(audio_slice: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
    """Extract simplified audio features for fast processing.
    
    Args:
        audio_slice: 1D numpy array containing audio samples
        sample_rate: Sample rate of the audio (default: 22050)
        
    Returns:
        10-dimensional feature vector (reduced from 28)
    """
    try:
        # Only extract essential features
        # 5 MFCCs (reduced from 13)
        mfccs = librosa.feature.mfcc(y=audio_slice, sr=sample_rate, n_mfcc=5)
        mfcc_features = np.mean(mfccs, axis=1)
        
        # 3 chroma features (reduced from 12)
        chroma = librosa.feature.chroma_cqt(y=audio_slice, sr=sample_rate, n_chroma=3)
        chroma_features = np.mean(chroma, axis=1)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_slice, sr=sample_rate)
        centroid_feature = np.mean(spectral_centroid)
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(audio_slice)
        zcr_feature = np.mean(zcr)
        
        # Combine features (10 dimensions total)
        feature_vector = np.concatenate([
            mfcc_features,      # 5 dimensions
            chroma_features,    # 3 dimensions
            [centroid_feature], # 1 dimension
            [zcr_feature]       # 1 dimension
        ])
        
        return feature_vector
        
    except Exception as e:
        raise RuntimeError(f"Fast feature extraction failed: {e}")


def extract_features_optimized(audio_slice: np.ndarray, sample_rate: int = 22050, stft_precomputed: Optional[np.ndarray] = None) -> np.ndarray:
    """Extract audio features with optional pre-computed STFT for efficiency.
    
    Args:
        audio_slice: 1D numpy array containing audio samples
        sample_rate: Sample rate of the audio (default: 22050)
        stft_precomputed: Pre-computed STFT if available (optional)
        
    Returns:
        28-dimensional feature vector
    """
    if len(audio_slice) == 0:
        raise ValueError("Audio slice cannot be empty")
        
    try:
        # Compute STFT once if not provided
        if stft_precomputed is None:
            stft = librosa.stft(audio_slice)
            stft_abs = np.abs(stft)
        else:
            stft_abs = stft_precomputed
            
        # Extract all features using the shared STFT
        # MFCCs
        mel_spec = librosa.feature.melspectrogram(S=stft_abs**2, sr=sample_rate)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=13)
        mfcc_features = np.mean(mfccs, axis=1)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(S=stft_abs, sr=sample_rate, n_chroma=12)
        chroma_features = np.mean(chroma, axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=stft_abs, sr=sample_rate)
        centroid_feature = np.mean(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft_abs, sr=sample_rate)
        bandwidth_feature = np.mean(spectral_bandwidth)
        
        # ZCR (doesn't use STFT)
        zcr = librosa.feature.zero_crossing_rate(audio_slice)
        zcr_feature = np.mean(zcr)
        
        # Combine features
        feature_vector = np.concatenate([
            mfcc_features,
            chroma_features,
            [centroid_feature],
            [bandwidth_feature],
            [zcr_feature]
        ])
        
        return feature_vector
        
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")


def extract_features(audio_slice: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
    """Extract comprehensive audio features from an audio slice.
    
    Args:
        audio_slice: 1D numpy array containing audio samples
        sample_rate: Sample rate of the audio (default: 22050)
        
    Returns:
        28-dimensional feature vector containing:
        - 13 MFCC coefficients 
        - 12 chroma features
        - 1 spectral centroid
        - 1 spectral bandwidth  
        - 1 zero crossing rate
        
    Raises:
        ValueError: If audio_slice is empty or invalid
        RuntimeError: If feature extraction fails
    """
    if len(audio_slice) == 0:
        raise ValueError("Audio slice cannot be empty")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
        
    try:
        # Extract 13 MFCCs
        mfccs = librosa.feature.mfcc(y=audio_slice, sr=sample_rate, n_mfcc=13)
        mfcc_features = np.mean(mfccs, axis=1)  # Average over time
        
        # Extract 12 chroma features
        chroma = librosa.feature.chroma_stft(y=audio_slice, sr=sample_rate, n_chroma=12)
        chroma_features = np.mean(chroma, axis=1)  # Average over time
        
        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_slice, sr=sample_rate)
        centroid_feature = np.mean(spectral_centroid)
        
        # Extract spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_slice, sr=sample_rate)
        bandwidth_feature = np.mean(spectral_bandwidth)
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_slice)
        zcr_feature = np.mean(zcr)
        
        # Combine all features into a single vector (28 dimensions)
        feature_vector = np.concatenate([
            mfcc_features,      # 13 dimensions
            chroma_features,    # 12 dimensions  
            [centroid_feature], # 1 dimension
            [bandwidth_feature], # 1 dimension
            [zcr_feature]       # 1 dimension
        ])
        
        return feature_vector
        
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")


def reduce_dimensions(features: np.ndarray, max_samples: int = 50000) -> np.ndarray:
    """Reduce feature vectors to 2D coordinates using UMAP.
    
    Args:
        features: 2D numpy array where each row is a feature vector
        max_samples: Maximum number of samples to use for UMAP (default: 50000)
        
    Returns:
        2D numpy array with coordinates scaled to [-1, 1] range
        
    Raises:
        ValueError: If features array is empty or has invalid shape
        RuntimeError: If dimensionality reduction fails
    """
    if len(features) == 0:
        raise ValueError("Features array cannot be empty")
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array")
        
    try:
        # Standardize features (z-score normalization)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # For very large datasets, use sampling to fit UMAP
        if len(scaled_features) > max_samples:
            typer.echo(f"Dataset has {len(scaled_features)} samples, using stratified sampling of {max_samples} samples for UMAP fitting")
            
            # Random sampling for UMAP fitting
            rng = np.random.RandomState(42)
            sample_indices = rng.choice(len(scaled_features), size=max_samples, replace=False)
            sample_features = scaled_features[sample_indices]
            
            # Fit UMAP on sample
            umap_reducer = umap.UMAP(
                n_neighbors=min(30, max_samples - 1),
                min_dist=0.1,
                n_components=2,
                metric='cosine',
                random_state=42,
                low_memory=True,  # Use low memory mode
                n_jobs=1  # Single thread to avoid issues
            )
            
            # Fit on sample, then transform all data
            umap_reducer.fit(sample_features)
            coords = umap_reducer.transform(scaled_features)
        else:
            # Apply UMAP dimensionality reduction normally
            umap_reducer = umap.UMAP(
                n_neighbors=min(30, len(scaled_features) - 1),
                min_dist=0.1,
                n_components=2,
                metric='cosine',
                random_state=42,
                n_jobs=1  # Single thread to avoid issues
            )
            
            coords = umap_reducer.fit_transform(scaled_features)
        
        # Scale coordinates to [-1, 1] range
        max_abs = np.max(np.abs(coords))
        if max_abs > 0:  # Avoid division by zero
            coords = coords / max_abs
        
        return coords
        
    except Exception as e:
        # Try a fallback with even more conservative settings
        typer.echo(f"UMAP failed with error: {e}")
        typer.echo("Attempting fallback with PCA...")
        
        try:
            from sklearn.decomposition import PCA
            
            # Use PCA as fallback
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(scaled_features)
            
            # Scale coordinates to [-1, 1] range
            max_abs = np.max(np.abs(coords))
            if max_abs > 0:
                coords = coords / max_abs
                
            typer.echo("Successfully used PCA as fallback for dimensionality reduction")
            return coords
            
        except Exception as e2:
            raise RuntimeError(f"Both UMAP and PCA failed. UMAP error: {e}, PCA error: {e2}")


def process_audio_file(args: Tuple[str, float, str]) -> Tuple[List[np.ndarray], List[str], List[float], int, Optional[str]]:
    """Process a single audio file for multiprocessing.
    
    Args:
        args: Tuple of (file_path, slice_duration, feature_mode)
        
    Returns:
        Tuple of (features, paths, slice_times, silent_count, skip_reason)
    """
    file_path, slice_duration, feature_mode = args
    features = []
    paths = []
    slice_times = []
    silent_count = 0
    skip_reason = None
    
    try:
        # Load and slice audio
        slices = load_and_slice_audio(file_path, slice_duration=slice_duration)
        
        # Check if we got any valid slices
        if not slices:
            skip_reason = f"Too short (< {slice_duration}s)"
            return features, paths, slice_times, silent_count, skip_reason
        
        # Extract features for each slice
        for slice_idx, slice_data in enumerate(slices):
            # Skip silent slices
            if is_silent(slice_data):
                silent_count += 1
                continue
            
            try:
                # Extract features based on mode
                if feature_mode == 'fast':
                    slice_features = extract_features_fast(slice_data)
                elif feature_mode == 'optimized':
                    slice_features = extract_features_optimized(slice_data)
                else:
                    slice_features = extract_features(slice_data)
                    
                features.append(slice_features)
                paths.append(str(file_path))
                slice_times.append(slice_idx * slice_duration)
            except Exception as feat_error:
                # Skip this slice if feature extraction fails
                skip_reason = f"Feature extraction error: {str(feat_error)}"
                continue
            
    except Exception as e:
        # Return empty results on error (will be filtered out)
        skip_reason = f"Loading error: {str(e)}"
        
    return features, paths, slice_times, silent_count, skip_reason


def build_map(folder_path: str, output_path: str = "map.pkl", slice_duration: float = 0.5, 
              max_points: Optional[int] = None, num_workers: Optional[int] = None, 
              fast_mode: bool = False) -> dict:
    """Build the audio map from a folder of audio files.
    
    Args:
        folder_path: Path to folder containing audio files
        output_path: Path where map.pkl will be saved (default: "map.pkl")
        slice_duration: Duration of each slice in seconds (default: 0.5)
        max_points: Maximum number of points to include in the map (default: None for no limit)
        num_workers: Number of parallel workers (default: None for auto-detect)
        fast_mode: Use fast feature extraction (10 dims instead of 28)
        
    Returns:
        Dictionary containing the map data
        
    Raises:
        ValueError: If folder doesn't exist or contains no audio files
        RuntimeError: If build process fails
    """
    start_time = time.time()
    
    # Find all audio files
    typer.echo(f"Scanning folder: {folder_path}")
    audio_files = find_audio_files(folder_path)
    
    if not audio_files:
        raise ValueError(f"No audio files found in {folder_path}")
        
    typer.echo(f"Found {len(audio_files)} audio files")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(audio_files))
    
    typer.echo(f"Processing with {num_workers} parallel workers...")
    if fast_mode:
        typer.echo("Fast mode enabled: using simplified 10-dimensional features")
    
    # Prepare arguments for multiprocessing
    feature_mode = 'fast' if fast_mode else 'optimized'
    process_args = [(file_path, slice_duration, feature_mode) for file_path in audio_files]
    
    # Process files in parallel
    all_features = []
    all_paths = []
    all_slice_times = []
    slice_count = 0
    silent_count = 0
    skipped_files = []
    skip_reasons = {}
    
    # Use multiprocessing with progress bar
    with mp.Pool(processes=num_workers) as pool:
        # Process with tqdm progress bar
        results = list(tqdm(
            pool.imap_unordered(process_audio_file, process_args),
            total=len(audio_files),
            desc="Processing audio files",
            unit="files"
        ))
        
        # Aggregate results
        for features, paths, times, silent, skip_reason in results:
            if features:  # Only add if processing succeeded
                all_features.extend(features)
                all_paths.extend(paths)
                all_slice_times.extend(times)
                slice_count += len(features)
                silent_count += silent
            elif skip_reason:  # Track skipped files
                # Extract file path from skip_reason (it contains the path info)
                if skip_reason not in skip_reasons:
                    skip_reasons[skip_reason] = 0
                skip_reasons[skip_reason] += 1
    
    if not all_features:
        typer.echo("\nError summary:")
        for reason, count in skip_reasons.items():
            typer.echo(f"  - {reason}: {count} files")
        raise RuntimeError("No features could be extracted from any audio files")
        
    feature_time = time.time() - start_time
    slices_per_second = slice_count / feature_time if feature_time > 0 else 0
    
    typer.echo(f"Extracted features from {slice_count} slices across {len(audio_files)} files")
    if silent_count > 0:
        typer.echo(f"Discarded {silent_count} silent slices")
    
    # Report skipped files
    total_skipped = sum(skip_reasons.values())
    if total_skipped > 0:
        typer.echo(f"\nSkipped {total_skipped} files:")
        for reason, count in skip_reasons.items():
            typer.echo(f"  - {reason}: {count} files")
    
    typer.echo(f"Processing speed: {slices_per_second:.1f} slices/second")
    
    # Apply max_points limit if specified
    if max_points and len(all_features) > max_points:
        typer.echo(f"Limiting to {max_points} points (from {len(all_features)}) using random sampling")
        
        # Random sampling
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(all_features), size=max_points, replace=False)
        sample_indices.sort()  # Keep order for consistency
        
        # Sample all arrays
        all_features = [all_features[i] for i in sample_indices]
        all_paths = [all_paths[i] for i in sample_indices]
        all_slice_times = [all_slice_times[i] for i in sample_indices]
    
    # Convert to numpy array
    feature_array = np.array(all_features)
    typer.echo(f"Feature array shape: {feature_array.shape}")
    
    # Reduce dimensions
    typer.echo("Reducing dimensions with UMAP...")
    coords = reduce_dimensions(feature_array)
    
    # Build KDTree for efficient spatial queries at runtime
    kdtree = None
    kdtree_built = False
    try:
        from scipy.spatial import cKDTree
        typer.echo("Building KDTree for fast spatial queries...")
        kdtree = cKDTree(coords)
        kdtree_built = True
        typer.echo(f"✓ KDTree built successfully with {len(coords)} points")
        
        # Test if KDTree is pickleable
        try:
            pickle.dumps(kdtree)
            typer.echo("✓ KDTree is pickleable and will be saved with the map")
        except:
            typer.echo("Warning: KDTree cannot be pickled, will need to rebuild at runtime")
            kdtree = None
            
    except ImportError:
        typer.echo("Warning: scipy not available - KDTree will not be built")
        typer.echo("Runtime performance may be degraded for large datasets")
    except Exception as e:
        typer.echo(f"Warning: KDTree construction failed: {e}")
        typer.echo("Runtime will use fallback nearest neighbor search")
    
    # Create map dictionary
    map = {
        'coords': coords.astype(np.float32),
        'paths': all_paths,
        'slice_times': all_slice_times,
        'kdtree': kdtree,  # Save the KDTree if available
        'metadata': {
            'total_files': len(audio_files),
            'total_slices': slice_count,
            'silent_slices_discarded': silent_count,
            'files_skipped': sum(skip_reasons.values()),
            'skip_reasons': skip_reasons,
            'slice_duration': slice_duration,
            'feature_dimensions': feature_array.shape[1],
            'build_time': time.time() - start_time,
            'kdtree_built': kdtree_built
        }
    }
    
    # Save map to pickle file
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(map, f)
    except Exception as e:
        raise RuntimeError(f"Failed to save map to {output_path}: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    typer.echo(f"Map built successfully in {duration:.2f} seconds")
    typer.echo(f"Map contains {len(map['coords'])} points")
    typer.echo(f"Map saved to {output_path}")
    
    return map