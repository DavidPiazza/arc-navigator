# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pymonome is a pure Python library for interfacing with monome grid and arc controllers via serialosc. The library provides:

- **Core Protocol Classes**: `Device`, `Grid`, `Arc` - handle OSC communication with monome devices
- **Discovery Service**: `SerialOsc` - automatically detects connected devices
- **Application Base Classes**: `GridApp`, `ArcApp` - provide event handlers and device management
- **Buffer Classes**: `GridBuffer`, `ArcBuffer` - software framebuffers for complex visuals
- **Page Management**: `GridPageManager`, `SeqGridPageManager`, `SumGridPageManager` - multi-page applications
- **Grid Sectioning**: `GridSection`, `GridSplitter` - divide grids into independent sections

## Development Environment

**Virtual Environment**: Always activate the project's virtual environment before running Python commands:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

**Installation**: Install in development mode:
```bash
pip install -e .
```

**Dependencies**: Core dependency is `aiosc` for OSC communication. Install requirements:
```bash
pip install aiosc
```

## Architecture

The library follows an asyncio-based event-driven architecture:

1. **OSC Protocol Layer**: Built on `aiosc.OSCProtocol` for UDP communication with serialosc
2. **Device Abstraction**: `Device` base class handles connection lifecycle and system messages
3. **Event System**: Custom `Event` class for dispatching device events (key presses, encoder deltas)
4. **Buffer System**: Separate buffer classes allow offline composition before rendering to hardware
5. **Application Framework**: App classes provide structured development patterns with lifecycle callbacks

## Key Files

- `monome.py` - Main library with all core classes and functionality
- `monome_arc.py` - Arc-only subset for projects that only need arc support
- `setup.py` - Package configuration and dependencies
- `examples/arc.py` - Reference implementation showing arc usage patterns
- `audio_map/` - Separate project using pymonome for audio sample exploration

## Testing

No formal test suite is currently present. Test applications by:
1. Connecting physical monome hardware
2. Running example scripts
3. Verifying device interaction and LED feedback

## Common Patterns

**Device Connection**:
```python
grid = monome.Grid()
await grid.connect('127.0.0.1', port)
```

**SerialOsc Discovery**:
```python
serialosc = monome.SerialOsc()
serialosc.device_added_event.add_handler(device_handler)
await serialosc.connect()
```

**Application Structure**:
```python
class MyApp(monome.GridApp):
    def on_grid_key(self, x, y, s):
        self.grid.led_set(x, y, s)
```

## Development Notes

- The library uses Python 3.6+ asyncio patterns
- OSC messages follow monome's standard protocol
- Varibright detection automatically determines LED capabilities
- Page managers handle complex multi-page applications
- Grid sections allow independent control of grid regions