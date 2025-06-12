"""Runtime orchestration for Audio Map.

This module coordinates real-time interaction with the monome Arc controller,
loads precomputed embeddings, handles audio playback, and manages user navigation
through the 2D audio map.
"""

import asyncio
import sys
import os
import sounddevice as sd
import torchaudio
import torch
import numpy as np
import threading
import pickle
import math
from collections import deque, OrderedDict

# Add parent directory to path first
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import KDTree with fallback
KDTREE_AVAILABLE = False
try:
    from scipy.spatial import cKDTree
    KDTREE_AVAILABLE = True
    import sys as _sys
    print(f"✓ scipy.spatial.cKDTree imported successfully", file=_sys.stderr)
except ImportError as e:
    import sys as _sys
    print(f"Warning: scipy import failed: {e}", file=_sys.stderr)
    print("Using fallback nearest neighbor search", file=_sys.stderr)

from monome import SerialOsc


class LRUCache:
    """Simple LRU cache implementation for audio data."""
    
    def __init__(self, max_size_mb=100):
        """Initialize LRU cache with maximum size in megabytes."""
        self.cache = OrderedDict()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.lock = threading.Lock()
    
    def get(self, key):
        """Get item from cache and move to end (most recent)."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key, value):
        """Put item in cache, evicting oldest items if needed."""
        with self.lock:
            # Calculate size of audio data (float32 = 4 bytes per sample)
            value_size = value.nbytes if hasattr(value, 'nbytes') else len(value) * 4
            
            # Remove old entry if updating
            if key in self.cache:
                old_value = self.cache[key]
                old_size = old_value.nbytes if hasattr(old_value, 'nbytes') else len(old_value) * 4
                self.current_size_bytes -= old_size
                del self.cache[key]
            
            # Evict oldest items if necessary
            while self.current_size_bytes + value_size > self.max_size_bytes and self.cache:
                oldest_key, oldest_value = self.cache.popitem(last=False)
                oldest_size = oldest_value.nbytes if hasattr(oldest_value, 'nbytes') else len(oldest_value) * 4
                self.current_size_bytes -= oldest_size
            
            # Add new item
            self.cache[key] = value
            self.current_size_bytes += value_size
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0


class ArcApp:
    """Base class for Arc applications that handles monome Arc controller communication."""
    
    def __init__(self):
        self.arc = None
        self.serialosc = SerialOsc()
        self.serialosc.device_added_event.add_handler(self._on_device_added)
        self.serialosc.device_removed_event.add_handler(self._on_device_removed)
        
        # Arc state
        self.encoder_positions = [0, 0, 0, 0]  # 4 encoders
        self.encoder_deltas = [0, 0, 0, 0]     # Change since last update
        self.button_state = False               # Push button state
        
        # LED state (64 LEDs per ring, brightness 0-15)
        self.led_brightness = [[0] * 64 for _ in range(4)]
    
    def _on_device_added(self, id, type, port):
        """Handle Arc device connection."""
        if "arc" in type.lower():  # Handle both "arc" and "monome arc"
            print(f"Arc connected: {id} (type: {type})")
            # Schedule the async connection in the event loop
            asyncio.create_task(self._connect_arc(id, port))
    
    def _on_device_removed(self, id, type, port):
        """Handle Arc device disconnection."""
        if "arc" in type.lower() and self.arc:
            print(f"Arc disconnected: {id}")
            self.arc = None
    
    async def _connect_arc(self, id, port):
        """Async method to connect to Arc device."""
        # Import the Arc class from monome module
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from monome import Arc
        
        # Create Arc device and connect
        self.arc = Arc()
        
        # Set up message handlers for Arc
        self.arc.add_handler('/monome/enc/delta', self._on_delta)
        self.arc.add_handler('/monome/enc/key', self._on_button)
        
        await self.arc.connect('127.0.0.1', port)
        await self.setup()
    
    def _on_delta(self, addr, path, ring, delta):
        """Handle encoder rotation."""
        if 0 <= ring < 4:
            self.encoder_positions[ring] = (self.encoder_positions[ring] + delta) % 64
            self.encoder_deltas[ring] = delta
    
    def _on_button(self, addr, path, ring, state):
        """Handle button press/release."""
        self.button_state = state
        if state:  # Button pressed
            # Schedule async button press handler
            asyncio.create_task(self.on_button_press())
    
    async def update_leds(self):
        """Update LEDs on the Arc - only send changes."""
        if self.arc:
            # Initialize last brightness if needed
            if self.last_led_brightness is None:
                self.last_led_brightness = [[0] * 64 for _ in range(4)]
            
            # Only update LEDs that have changed
            for ring in range(4):
                for led in range(64):
                    brightness = self.led_brightness[ring][led]
                    if brightness != self.last_led_brightness[ring][led]:
                        self.arc.ring_set(ring, led, brightness)
                        self.last_led_brightness[ring][led] = brightness
    
    async def setup(self):
        """Called when Arc is connected. Override in subclasses."""
        pass
    
    async def on_button_press(self):
        """Called when button is pressed. Override in subclasses."""
        pass
    
    async def update(self):
        """Called every update cycle. Override in subclasses."""
        pass
    
    async def run(self):
        """Start the SerialOsc discovery and run the app."""
        await self.serialosc.connect()
        print("Waiting for Arc controller...")
        
        while True:
            await asyncio.sleep(0.005)  # 5ms update interval
            await self.update()
            await self.update_leds()


class AudioPlayer:
    """Audio playback system using sounddevice with low-latency CoreAudio."""
    
    def __init__(self, device=None, sample_rate=None, loop_enabled: bool = False):
        self.device = device
        # Store looping preference
        self.loop_enabled = loop_enabled
        
        # Auto-detect optimal sample rate if not specified
        if sample_rate is None:
            try:
                import sounddevice as sd
                if device is not None:
                    device_info = sd.query_devices(device)
                else:
                    device_info = sd.query_devices(kind='output')
                # Use device's native sample rate for best compatibility
                self.sample_rate = int(device_info['default_samplerate'])
                print(f"Using device native sample rate: {self.sample_rate} Hz")
            except:
                self.sample_rate = 22050  # Fallback
                print(f"Fallback to sample rate: {self.sample_rate} Hz")
        else:
            self.sample_rate = sample_rate
        self.current_audio = None
        self.next_audio = None  # For crossfading
        self.position = 0
        self.crossfade_samples = int(0.01 * self.sample_rate)  # 10ms crossfade
        self.crossfade_position = 0
        self.is_playing = False
        self.is_crossfading = False
        self.stream = None
        self.audio_cache = LRUCache(max_size_mb=200)  # 200MB LRU cache for audio
        self.playback_queue = deque(maxlen=8)  # For mode 1 (morphing between k nearest)
        self.current_mode = 0  # 0: single, 1: morph
        self.buffer_size = 256  # Smaller buffer for lower latency
        
        # For morph mode - multiple simultaneous sources
        self.morph_sources = []  # List of (audio_data, position, amplitude) tuples
        self.morph_lock = threading.Lock()
        
        # Playback rate control (handled in real-time during callback)
        self.playback_rate = 1.0
        
        # Start the audio stream once
        self.start_stream()
    
    def __del__(self):
        """Cleanup when AudioPlayer is destroyed."""
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.stop_stream()
            except:
                pass
        
    def load_audio(self, file_path, start_time=0, duration=1.0):
        """Load audio file segment into memory with LRU caching.
        
        Args:
            file_path: Path to audio file
            start_time: Start time in seconds  
            duration: Duration in seconds
        """
        # Cache key without rate since we no longer pre-process for different rates
        cache_key = f"{file_path}_{start_time}_{duration}"
        
        # Check cache first
        cached_audio = self.audio_cache.get(cache_key)
        if cached_audio is not None:
            return cached_audio
        
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Extract segment
            start_sample = int(start_time * self.sample_rate)
            duration_samples = int(duration * self.sample_rate)
            end_sample = min(start_sample + duration_samples, waveform.shape[1])
            
            segment = waveform[0, start_sample:end_sample].numpy()
            
            # Pad if needed
            if len(segment) < duration_samples:
                segment = np.pad(segment, (0, duration_samples - len(segment)))
            
            # No time-stretching applied - rate control now handled in real-time during playback
            
            # Store in LRU cache
            self.audio_cache.put(cache_key, segment)
            return segment
        except Exception as e:
            print(f"Error loading audio {file_path}: {e}")
            return np.zeros(int(duration * self.sample_rate))
    
    def callback(self, outdata, frames, time, status):
        """Audio callback for sounddevice with crossfading and morph support."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Fill with zeros by default
        outdata.fill(0)
        
        # Handle morph mode - mix multiple sources
        if self.current_mode == 1 and self.morph_sources:
            with self.morph_lock:
                # Mix all morph sources
                for i in range(frames):
                    mixed_sample = 0.0
                    sources_to_remove = []
                    
                    for idx, (audio_data, position, amplitude) in enumerate(self.morph_sources):
                        if position < len(audio_data):
                            # Add this source's contribution scaled by amplitude
                            mixed_sample += audio_data[int(position)] * amplitude
                            # Update position with rate control
                            new_position = position + self.playback_rate
                            self.morph_sources[idx] = (audio_data, new_position, amplitude)
                        else:
                            # Mark for removal if at end
                            sources_to_remove.append(idx)
                    
                    # Remove finished sources
                    for idx in reversed(sources_to_remove):
                        self.morph_sources.pop(idx)
                    
                    # Apply soft clipping to prevent distortion
                    if mixed_sample > 1.0:
                        mixed_sample = 1.0 - (2.0 / (1.0 + np.exp(2.0 * mixed_sample)))
                    elif mixed_sample < -1.0:
                        mixed_sample = -1.0 + (2.0 / (1.0 + np.exp(-2.0 * mixed_sample)))
                    
                    outdata[i, 0] = mixed_sample
            
            return
        
        # Original single/crossfade mode handling
        if not self.is_playing and not self.is_crossfading:
            return
            
        if self.current_audio is None:
            self.is_playing = False
            self.is_crossfading = False
            return
        
        # Handle crossfading
        if self.is_crossfading and self.next_audio is not None:
            for i in range(frames):
                if self.crossfade_position < self.crossfade_samples:
                    # Crossfade ratio (0 to 1)
                    fade_ratio = self.crossfade_position / self.crossfade_samples
                    
                    # Get current audio sample (with rate control)
                    current_sample = 0.0
                    if int(self.position) < len(self.current_audio):
                        current_sample = self.current_audio[int(self.position)]
                    
                    # Get next audio sample (with rate control)
                    next_sample = 0.0
                    if int(self.crossfade_position) < len(self.next_audio):
                        next_sample = self.next_audio[int(self.crossfade_position)]
                    
                    # Crossfade: fade out current, fade in next
                    mixed_sample = current_sample * (1.0 - fade_ratio) + next_sample * fade_ratio
                    outdata[i, 0] = mixed_sample
                    
                    # Update positions with rate control
                    self.position += self.playback_rate
                    self.crossfade_position += self.playback_rate
                else:
                    # Crossfade complete, switch to next audio
                    self.current_audio = self.next_audio
                    self.next_audio = None
                    self.position = self.crossfade_position
                    self.is_crossfading = False
                    self.crossfade_position = 0
                    break
        else:
            # Normal playback with real-time rate control
            if self.position >= len(self.current_audio):
                self.is_playing = False
                return
            
            # Sample-by-sample playback with rate control
            for i in range(frames):
                if int(self.position) < len(self.current_audio):
                    outdata[i, 0] = self.current_audio[int(self.position)]
                    self.position += self.playback_rate
                else:
                    self.is_playing = False
                    break
    
    def start_stream(self):
        """Start the audio stream with optimized settings."""
        if self.stream is None or not self.stream.active:
            try:
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self.callback,
                    device=self.device,
                    blocksize=self.buffer_size,
                    latency='low',
                    clip_off=True,  # Prevent clipping artifacts
                    never_drop_input=False,
                    prime_output_buffers_using_stream_callback=True
                )
                self.stream.start()
            except Exception as e:
                print(f"Error starting audio stream: {e}")
                # Fallback to larger buffer size
                try:
                    self.buffer_size = 512
                    self.stream = sd.OutputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        callback=self.callback,
                        device=self.device,
                        blocksize=self.buffer_size,
                        latency='low'
                    )
                    self.stream.start()
                    print(f"Fallback to buffer size {self.buffer_size}")
                except Exception as e2:
                    print(f"Audio stream fallback failed: {e2}")
    
    def stop_stream(self):
        """Stop the audio stream."""
        if self.stream is not None and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def play_audio(self, file_path, start_time=0, duration=1.0, rate=1.0):
        """Play audio file segment with crossfading and real-time rate control.
        
        Args:
            file_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration in seconds  
            rate: Playback rate multiplier (affects both speed and pitch)
        """
        # Set the playback rate for real-time control
        self.playback_rate = rate
        new_audio = self.load_audio(file_path, start_time, duration)
        
        if self.is_playing and self.current_audio is not None:
            # Start crossfade to new audio
            self.next_audio = new_audio
            self.is_crossfading = True
            self.crossfade_position = 0
        else:
            # Direct playback
            self.current_audio = new_audio
            self.position = 0
            self.is_playing = True
            self.is_crossfading = False
    
    def queue_audio(self, file_path, start_time=0, duration=1.0):
        """Queue audio for mode 1 (cycling)."""
        self.playback_queue.append((file_path, start_time, duration))
    
    def play_next_in_queue(self):
        """Play next audio in the queue (for mode 1)."""
        if not self.playback_queue:
            return
        
        # Allow overlapping in cycle mode for smoother transitions
        if self.current_mode == 1 or not self.is_playing:
            file_path, start_time, duration = self.playback_queue.popleft()
            self.play_audio(file_path, start_time, duration)
    
    def set_mode(self, mode):
        """Set playback mode (0: single, 1: morph)."""
        self.current_mode = mode
        self.playback_queue.clear()
        if mode == 1:
            # Clear morph sources when switching to morph mode
            with self.morph_lock:
                self.morph_sources.clear()
    
    def set_playback_rate(self, rate):
        """Set the playback rate for real-time rate control.
        
        Args:
            rate: Playback rate multiplier (0.25 to 4.0). Affects both speed and pitch.
        """
        self.playback_rate = max(0.25, min(4.0, rate))
    
    def set_morph_sources(self, sources_with_distances, rate=1.0):
        """Set multiple audio sources for morph mode with distance-based amplitudes.
        
        Args:
            sources_with_distances: List of tuples (file_path, start_time, duration, distance)
            rate: Playback rate multiplier (affects both speed and pitch)
        """
        if self.current_mode != 1:
            return
        
        # Set the playback rate for real-time control
        self.playback_rate = rate
        
        with self.morph_lock:
            self.morph_sources.clear()
            
            # Find minimum distance for normalization
            if sources_with_distances:
                min_distance = min(s[3] for s in sources_with_distances)
                max_distance = max(s[3] for s in sources_with_distances)
                distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
                
                for file_path, start_time, duration, distance in sources_with_distances:
                    # Load audio data at original speed
                    audio_data = self.load_audio(file_path, start_time, duration)
                    
                    # Calculate amplitude based on distance (inverse relationship)
                    # Closer points get higher amplitude
                    normalized_distance = (distance - min_distance) / distance_range if distance_range > 0 else 0
                    # Use exponential decay for more natural morphing
                    amplitude = np.exp(-2.0 * normalized_distance)
                    
                    # Normalize so total amplitude doesn't exceed 1.0
                    total_sources = len(sources_with_distances)
                    amplitude = amplitude / np.sqrt(total_sources)
                    
                    # Add to morph sources with position starting at 0
                    self.morph_sources.append((audio_data, 0, amplitude))
    
    def update(self):
        """Update playback state."""
        # Morph mode: restart sources only if looping is enabled
        if self.current_mode == 1:
            with self.morph_lock:
                new_sources = []
                for audio_data, position, amplitude in self.morph_sources:
                    if position >= len(audio_data):
                        if self.loop_enabled:
                            # Restart from beginning if looping
                            new_sources.append((audio_data, 0, amplitude))
                        # If looping disabled, simply drop this source (stop playback)
                    else:
                        new_sources.append((audio_data, position, amplitude))
                self.morph_sources = new_sources


class mapApp(ArcApp):
    """arc-navigator Application for exploring 2D audio space with Arc controller."""
    
    def __init__(self, map_path, audio_device=None, k_nearest=8, loop_enabled: bool = False):
        super().__init__()
        self.map_path = map_path
        self.map_data = None
        # Pass looping preference down to the AudioPlayer
        self.audio_player = AudioPlayer(device=audio_device, loop_enabled=loop_enabled)
        self.loop_enabled = loop_enabled
        
        # Navigation state
        self.cursor_x = 0.0  # Cursor position in [-1, 1] range
        self.cursor_y = 0.0
        self.zoom_level = 1.0  # Zoom level (0.1 to 10.0)
        self.playback_mode = 0  # 0: single nearest, 1: morph between k nearest
        
        # Interaction parameters
        self.movement_speed = 0.02  # How fast cursor moves per encoder delta
        self.zoom_speed = 0.05     # How fast zoom changes per encoder delta
        self.k_nearest = k_nearest         # Number of nearest points for morph mode
        
        # Rate control parameters (Ring 3)
        self.playback_rate = 1.0   # Playback rate multiplier (0.25x to 4.0x)
        self.rate_speed = 0.02     # How fast rate changes per encoder delta
        self.last_rate_change_time = 0  # Timestamp of last rate change
        self.rate_display_timeout = 2.0  # Seconds to show rate dial before reverting to map
        self.showing_rate_dial = False   # Whether Ring 3 is showing rate dial
        
        # Audio throttling
        self.last_audio_change = 0
        self.audio_throttle_ms = 100  # Minimum 100ms between audio changes
        self.last_cursor_pos = (0.0, 0.0)
        self.cursor_threshold = 0.01  # Minimum cursor movement to trigger new audio
        
        # Initialize KDTree for fast nearest neighbor search
        self.kdtree = None
        
        # Load map data
        self.load_map()
        
        # Density calculation cache
        self.density_cache = None
        self.density_cache_position = (None, None)
        self.density_cache_zoom = None
        self.last_density_update = 0
        self.density_update_interval = 50  # Update density every 50ms instead of 5ms
        
        # Smart update flags
        self.navigation_changed = False
        self.last_led_brightness = None
        
        # Performance monitoring
        self.density_calc_count = 0
        self.density_calc_time = 0
        self.last_perf_report = 0
        self._last_points_in_view = 0
        
        # Track last set of nearest indices to avoid redundant retriggers
        self.last_nearest_indices: tuple[int, ...] = tuple()
    
    def load_map(self):
        """Load the precomputed map data from pickle file."""
        try:
            with open(self.map_path, 'rb') as f:
                self.map_data = pickle.load(f)
            
            # Validate map structure
            required_keys = ['coords', 'paths', 'slice_times']
            for key in required_keys:
                if key not in self.map_data:
                    raise ValueError(f"Map missing required key: {key}")
            
            print(f"Map loaded: {self.map_data['metadata']['total_slices']} audio slices")
            
            # Debug: Check coordinate ranges
            coords = self.map_data['coords']
            print(f"Coordinate ranges: X=[{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}], "
                  f"Y=[{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
            
            # Check if KDTree was pre-built during analysis
            print(f"Checking for pre-built KDTree in map file...")
            print(f"Map keys: {list(self.map_data.keys())}")
            
            if 'kdtree' in self.map_data and self.map_data['kdtree'] is not None:
                self.kdtree = self.map_data['kdtree']
                print("✓ Pre-built KDTree loaded from map file")
                print("✓ Spatial indexing enabled for density visualization")
                # Test the KDTree - use a point in the middle of the data range
                try:
                    # Use the center of the coordinate space for testing
                    test_point = [
                        (self.map_data['coords'][:, 0].min() + self.map_data['coords'][:, 0].max()) / 2,
                        (self.map_data['coords'][:, 1].min() + self.map_data['coords'][:, 1].max()) / 2
                    ]
                    test_result = self.kdtree.query_ball_point(test_point, 0.1)
                    print(f"✓ KDTree test query successful (found {len(test_result)} points near center)")
                except Exception as e:
                    print(f"Warning: Pre-built KDTree test failed: {e}")
                    self.kdtree = None
            
            # If no pre-built KDTree, try to build one now
            else:
                print("No pre-built KDTree found in map file")
                if KDTREE_AVAILABLE:
                    try:
                        print(f"Building KDTree at runtime with {len(self.map_data['coords'])} points...")
                        self.kdtree = cKDTree(self.map_data['coords'])
                        print("✓ KDTree built successfully at runtime")
                    except Exception as e:
                        print(f"ERROR: Runtime KDTree creation failed: {e}")
                        self.kdtree = None
                else:
                    self.kdtree = None
                    if 'metadata' in self.map_data and 'kdtree_built' in self.map_data['metadata'] and self.map_data['metadata']['kdtree_built']:
                        print("Warning: KDTree was built during analysis but scipy is not available at runtime")
                    else:
                        print("Warning: scipy not available - using brute-force search")
            
            print(f"✓ Smart LED updates enabled (50ms density refresh)")
            print(f"✓ LRU audio cache enabled (200MB limit)")
            
        except Exception as e:
            print(f"Error loading map: {e}")
            sys.exit(1)
    
    def find_nearest_points(self, x, y, k=1):
        """Find k nearest points to cursor position using KDTree or fallback method."""
        if self.map_data is None:
            return []
        
        if self.kdtree is not None:
            # Use KDTree for O(log n) performance
            try:
                query_point = np.array([x, y])
                distances, indices = self.kdtree.query(query_point, k=k)
                
                # Handle single point case (kdtree returns scalar instead of array)
                if k == 1:
                    distances = [distances]
                    indices = [indices]
                
                results = []
                for i, idx in enumerate(indices):
                    results.append({
                        'index': idx,
                        'distance': distances[i],
                        'coord': self.map_data['coords'][idx],
                        'path': self.map_data['paths'][idx],
                        'slice_time': self.map_data['slice_times'][idx]
                    })
                
                return results
                
            except Exception as e:
                print(f"KDTree query failed: {e}, falling back to brute force")
                # Fall through to brute force method
        
        # Fallback: Original brute-force method
        coords = self.map_data['coords']
        
        # Calculate distances
        distances = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)
        
        # Get k nearest indices
        nearest_indices = np.argsort(distances)[:k]
        
        results = []
        for idx in nearest_indices:
            results.append({
                'index': idx,
                'distance': distances[idx],
                'coord': coords[idx],
                'path': self.map_data['paths'][idx],
                'slice_time': self.map_data['slice_times'][idx]
            })
        
        return results
    
    def calculate_point_density(self, ring, sector_count=64, force_update=False):
        """Calculate point density for each sector of an encoder ring using spatial indexing."""
        if self.map_data is None:
            return [0] * sector_count
        
        import time
        current_time = time.time() * 1000  # ms
        
        # Check if we can use cached density
        if (not force_update and 
            self.density_cache is not None and
            self.density_cache_position == (self.cursor_x, self.cursor_y) and
            self.density_cache_zoom == self.zoom_level and
            current_time - self.last_density_update < self.density_update_interval):
            return self.density_cache
        
        # Track performance
        calc_start = time.time()
        
        densities = [0] * sector_count
        
        # Define sector angles
        sector_angle = 2 * math.pi / sector_count
        
        # Calculate visible area based on zoom
        view_radius = 1.0 / self.zoom_level
        
        # Use KDTree for efficient spatial query if available
        if self.kdtree is not None:
            try:
                # Query all points within view radius of cursor
                query_point = np.array([self.cursor_x, self.cursor_y])
                # Use L-infinity norm (max distance) for rectangular query region
                point_indices = self.kdtree.query_ball_point(query_point, view_radius, p=np.inf)
                
                # If we get an empty result, also try with L2 norm as fallback
                if len(point_indices) == 0 and view_radius < 2.0:
                    # Try Euclidean distance instead
                    point_indices = self.kdtree.query_ball_point(query_point, view_radius, p=2)
                
                # Debug: Store points in view for performance monitoring
                self._last_points_in_view = len(point_indices)
                
                # Process only the visible points
                coords = self.map_data['coords']
                for idx in point_indices:
                    x, y = coords[idx]
                    
                    # Calculate angle from cursor to point
                    dx = x - self.cursor_x
                    dy = y - self.cursor_y
                    angle = math.atan2(dy, dx) + math.pi  # Normalize to [0, 2π]
                    
                    # Determine which sector this point belongs to
                    sector = int(angle / sector_angle) % sector_count
                    densities[sector] += 1
                
                return densities
                
            except Exception as e:
                # Fallback to brute force if KDTree query fails
                print(f"KDTree density query failed: {e}")
                print(f"Query point: {query_point}, radius: {view_radius}")
                import traceback
                traceback.print_exc()
                self._last_points_in_view = 0
        
        # Fallback: Original brute-force method
        if self.kdtree is None and not hasattr(self, '_kdtree_warning_shown'):
            print("Warning: Using brute-force search because KDTree is not available")
            self._kdtree_warning_shown = True
        
        coords = self.map_data['coords']
        visible_points = 0
        for i, (x, y) in enumerate(coords):
            # Skip points outside visible area
            if abs(x - self.cursor_x) > view_radius or abs(y - self.cursor_y) > view_radius:
                continue
            
            visible_points += 1
            # Calculate angle from cursor to point
            dx = x - self.cursor_x
            dy = y - self.cursor_y
            angle = math.atan2(dy, dx) + math.pi  # Normalize to [0, 2π]
            
            # Determine which sector this point belongs to
            sector = int(angle / sector_angle) % sector_count
            densities[sector] += 1
        
        self._last_points_in_view = visible_points
        
        # Update cache
        self.density_cache = densities
        self.density_cache_position = (self.cursor_x, self.cursor_y)
        self.density_cache_zoom = self.zoom_level
        self.last_density_update = current_time
        
        # Track performance
        calc_time = (time.time() - calc_start) * 1000  # ms
        self.density_calc_count += 1
        self.density_calc_time += calc_time
        
        return densities
    
    def update_navigation(self):
        """Update cursor position and zoom based on encoder deltas."""
        # Track if navigation state changed
        navigation_changed = False
        
        # Ring 0 & 1: Cursor X & Y movement
        if self.encoder_deltas[0] != 0:
            self.cursor_x += self.encoder_deltas[0] * self.movement_speed / self.zoom_level
            # Wrap coordinates around the edges for seamless navigation
            if self.cursor_x > 1.0:
                self.cursor_x = -1.0 + (self.cursor_x - 1.0)
            elif self.cursor_x < -1.0:
                self.cursor_x = 1.0 + (self.cursor_x + 1.0)
            navigation_changed = True
        
        if self.encoder_deltas[1] != 0:
            self.cursor_y += self.encoder_deltas[1] * self.movement_speed / self.zoom_level
            # Wrap coordinates around the edges for seamless navigation
            if self.cursor_y > 1.0:
                self.cursor_y = -1.0 + (self.cursor_y - 1.0)
            elif self.cursor_y < -1.0:
                self.cursor_y = 1.0 + (self.cursor_y + 1.0)
            navigation_changed = True
        
        # Ring 2: Zoom control
        if self.encoder_deltas[2] != 0:
            zoom_factor = 1.0 + (self.encoder_deltas[2] * self.zoom_speed)
            self.zoom_level *= zoom_factor
            self.zoom_level = max(0.1, min(10.0, self.zoom_level))
            navigation_changed = True
        
        # Ring 3: Rate control
        if self.encoder_deltas[3] != 0:
            rate_change = self.encoder_deltas[3] * self.rate_speed
            self.playback_rate += rate_change
            self.playback_rate = max(0.25, min(4.0, self.playback_rate))
            # Record time of rate change and show rate dial
            import time
            self.last_rate_change_time = time.time()
            self.showing_rate_dial = True
            navigation_changed = True
        
        self.navigation_changed = navigation_changed
    
    def update_audio_playback(self):
        """Update audio playback based on cursor position and mode with throttling."""
        import time
        
        # Check if enough time has passed and cursor moved significantly
        current_time = time.time() * 1000  # Convert to milliseconds
        cursor_moved = (abs(self.cursor_x - self.last_cursor_pos[0]) > self.cursor_threshold or
                       abs(self.cursor_y - self.last_cursor_pos[1]) > self.cursor_threshold)
        
        if (current_time - self.last_audio_change < self.audio_throttle_ms and 
            not cursor_moved):
            return
        
        nearest_points = self.find_nearest_points(
            self.cursor_x, self.cursor_y,
            k=self.k_nearest if self.playback_mode == 1 else 1
        )

        if not nearest_points:
            return

        # Extract indices of nearest points and compare with last playback
        new_indices = tuple(sorted(p['index'] for p in nearest_points))
        if new_indices == self.last_nearest_indices:
            # Same neighbour(s) – skip retriggering audio
            return
        
        if self.playback_mode == 0:  # Single mode
            # Play the nearest point with crossfading and current rate
            point = nearest_points[0]
            self.audio_player.play_audio(
                point['path'], 
                point['slice_time'], 
                duration=1.0,
                rate=self.playback_rate
            )
            self.last_audio_change = current_time
            self.last_cursor_pos = (self.cursor_x, self.cursor_y)
            self.last_nearest_indices = new_indices
        
        else:  # Morph mode
            # Set up multiple sources with distance-based amplitudes
            sources_with_distances = []
            for point in nearest_points:
                sources_with_distances.append((
                    point['path'],
                    point['slice_time'],
                    1.0,  # duration
                    point['distance']
                ))
            
            self.audio_player.set_morph_sources(sources_with_distances, rate=self.playback_rate)
            self.last_audio_change = current_time
            self.last_cursor_pos = (self.cursor_x, self.cursor_y)
            self.last_nearest_indices = new_indices
    
    def update_led_visualization(self):
        """Update LED patterns to visualize navigation state."""
        import time
        current_time = time.time()
        
        # Check if we should hide the rate dial and revert to map visualization
        if self.showing_rate_dial and (current_time - self.last_rate_change_time) > self.rate_display_timeout:
            self.showing_rate_dial = False
            # Force update Ring 3 to show map visualization
            self.navigation_changed = True
        
        # Only update position/zoom LEDs if navigation changed
        if self.navigation_changed:
            # Ring 0: X position indicator
            center_x = int((self.cursor_x + 1.0) * 32)  # Map [-1,1] to [0,64]
            for led in range(64):
                distance = min(abs(led - center_x), 64 - abs(led - center_x))
                brightness = max(0, 15 - distance * 2) if distance < 8 else 0
                self.led_brightness[0][led] = brightness
            
            # Ring 1: Y position indicator  
            center_y = int((self.cursor_y + 1.0) * 32)  # Map [-1,1] to [0,64]
            for led in range(64):
                distance = min(abs(led - center_y), 64 - abs(led - center_y))
                brightness = max(0, 15 - distance * 2) if distance < 8 else 0
                self.led_brightness[1][led] = brightness
            
            # Ring 2: Zoom level indicator
            zoom_leds = int(self.zoom_level * 6.4)  # Map [0,10] to [0,64]
            for led in range(64):
                if led < zoom_leds:
                    self.led_brightness[2][led] = 8
                else:
                    self.led_brightness[2][led] = 0
            
            # Ring 3: Rate control or map visualization
            if self.showing_rate_dial:
                # Show rate dial similar to zoom control
                # Map rate [0.25, 4.0] to LED range [0, 64]
                rate_leds = int((self.playback_rate - 0.25) / (4.0 - 0.25) * 64)
                rate_leds = max(0, min(63, rate_leds))  # Clamp to valid range
                
                # Clear all LEDs first
                for led in range(64):
                    self.led_brightness[3][led] = 0
                
                # Fill LEDs up to rate position (similar to zoom)
                for led in range(rate_leds + 1):
                    self.led_brightness[3][led] = 8
                
                # Highlight the exact rate position
                self.led_brightness[3][rate_leds] = 15
                
                # Add special indicator for 1.0x speed (around position 21)
                center_pos = int((1.0 - 0.25) / (4.0 - 0.25) * 64)  # Position for 1.0x rate
                if abs(rate_leds - center_pos) <= 1:
                    # Blink for 1.0x rate
                    center_brightness = 15 if (int(current_time * 4) % 2) else 5
                    self.led_brightness[3][center_pos] = center_brightness
            else:
                # Show map visualization - calculate density for Ring 3
                densities = self.calculate_point_density(ring=3)
                max_density = max(densities) if densities else 1
                
                for led in range(64):
                    if max_density > 0:
                        # Scale density to brightness (0-15)
                        brightness = int((densities[led] / max_density) * 15)
                        self.led_brightness[3][led] = brightness
                    else:
                        self.led_brightness[3][led] = 0
    
    async def setup(self):
        """Called when Arc is connected."""
        print("Arc connected - arc-navigator ready")
        print("Ring 0: X position | Ring 1: Y position")
        print("Ring 2: Zoom level | Ring 3: Playback rate (varispeed)")
        print(f"Mode: {'morph' if self.playback_mode else 'single'}")
        print(f"Initial playback rate: {self.playback_rate:.2f}x (affects speed and pitch)")
        print(f"Morph neighbors: {self.k_nearest}")
        print(f"Looping {'enabled' if self.loop_enabled else 'disabled'}")
    
    async def on_button_press(self):
        """Handle button press to toggle playback mode."""
        self.playback_mode = 1 - self.playback_mode
        self.audio_player.set_mode(self.playback_mode)
        if self.playback_mode == 1:
            print(f"Playback mode: morph (using {self.k_nearest} nearest neighbors)")
        else:
            print("Playback mode: single")
        # Reset nearest tracking so next update triggers playback for new mode
        self.last_nearest_indices = tuple()
    
    async def update(self):
        """Main update loop called every 5ms."""
        # Update navigation based on encoder input
        self.update_navigation()
        
        # Update audio playback
        self.update_audio_playback()
        
        # Update audio player state
        self.audio_player.update()
        
        # Update LED visualization
        self.update_led_visualization()
        
        # Reset encoder deltas for next frame
        self.encoder_deltas = [0, 0, 0, 0]
        
        # Print performance stats every 5 seconds
        import time
        current_time = time.time()
        if current_time - self.last_perf_report > 5.0:
            if self.density_calc_count > 0:
                avg_density_time = self.density_calc_time / self.density_calc_count
                print(f"Performance: Avg density calc: {avg_density_time:.2f}ms, "
                      f"Points in view: {self._last_points_in_view}, "
                      f"Cache size: {self.audio_player.audio_cache.current_size_bytes / 1024 / 1024:.1f}MB, "
                      f"Using: {'KDTree' if self.kdtree is not None else 'Brute-force'}, "
                      f"Cursor: ({self.cursor_x:.2f}, {self.cursor_y:.2f}), Zoom: {self.zoom_level:.2f}, "
                      f"Rate: {self.playback_rate:.2f}x{'*' if self.showing_rate_dial else ''}, "
                      f"Neighbors: {self.k_nearest}")
            self.last_perf_report = current_time
    
    def cleanup(self):
        """Cleanup resources when application exits."""
        if self.audio_player:
            self.audio_player.stop_stream()
        if self.arc:
            # Clear all LEDs
            for ring in range(4):
                self.arc.ring_all(ring, 0)


def run_navigator(map_path, device_id=None, neighbors=8, loop=False):
    """Enhanced startup function for the arc-navigator with device listing and user feedback.
    The `loop` flag controls whether audio slices should loop continuously.
    """
    # Re-check scipy availability at runtime
    try:
        from scipy.spatial import cKDTree as test_import
        print("✓ scipy.spatial.cKDTree verified at runtime")
    except ImportError as e:
        print(f"ERROR: scipy not available at runtime: {e}")
    
    try:
        # Validate map file exists
        if not os.path.exists(map_path):
            print(f"Error: Map file not found: {map_path}")
            return
        
        # List available audio devices for user information
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            print("Available audio devices:")
            for i, dev in enumerate(devices):
                device_type = "input" if dev['max_input_channels'] > 0 else ""
                device_type += "/" if dev['max_input_channels'] > 0 and dev['max_output_channels'] > 0 else ""
                device_type += "output" if dev['max_output_channels'] > 0 else ""
                print(f"  {i}: {dev['name']} ({device_type})")
            
            # Show which device will be used
            if device_id is not None:
                if 0 <= device_id < len(devices):
                    selected_device = devices[device_id]
                    print(f"\nUsing audio device {device_id}: {selected_device['name']}")
                else:
                    print(f"Warning: Device ID {device_id} not found, using default device")
                    device_id = None
            
            if device_id is None:
                default_device = sd.query_devices(kind='output')
                print(f"\nUsing default audio device: {default_device['name']}")
            
        except ImportError:
            print("Warning: sounddevice not available, audio device listing disabled")
        except Exception as e:
            print(f"Warning: Could not query audio devices: {e}")
        
        print(f"\nStarting Audio-arc-navigator with map: {map_path}")
        print("\nControls:")
        print("  Ring 0: X position navigation")
        print("  Ring 1: Y position navigation")  
        print("  Ring 2: Zoom level control")
        print("  Ring 3: Playback rate control (0.25x to 4.0x, varispeed effect)")
        print("  Button: Toggle playback mode (single/morph)")
        print(f"  Morph mode using {neighbors} nearest neighbors")
        print(f"  Looping {'enabled' if loop else 'disabled'}")
        print("\nWaiting for Arc controller connection...")
        
        # Create and start the arc-navigator
        app = mapApp(map_path, audio_device=device_id, k_nearest=neighbors, loop_enabled=loop)
        
        # Run the asyncio event loop
        asyncio.run(app.run())
        
    except KeyboardInterrupt:
        print("\nShutting down arc-navigator...")
    except Exception as e:
        print(f"Error running arc-navigator: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup on exit
        if 'app' in locals():
            app.cleanup()


def start_navigator(device=None, neighbors=8, loop=False):
    """Legacy compatibility function for the CLI interface."""
    # Default map path
    map_path = "map.pkl"
    
    # Check if map exists
    if not os.path.exists(map_path):
        print("Error: map.pkl not found. Run 'audio_navigator build <folder>' first.")
        return
    
    # Call the main run_navigator function
    run_navigator(map_path, device, neighbors, loop) 