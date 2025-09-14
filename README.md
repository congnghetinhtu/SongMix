# Apple Music-Style Automix Script

Create seamless DJ-style mixes from a folder of audio tracks, mimicking Apple Music's Automix feature with intelligent transitions, tempo matching, and crossfading.

## Features

🎵 **Smart Track Analysis**
- Automatic tempo detection
- Key detection for harmonic mixing
- Energy level analysis
- Spectral feature analysis

🔄 **Intelligent Transitions**
- Equal-power crossfading
- Tempo adjustment for better transitions
- Track reordering for optimal flow
- Compatibility scoring between tracks

🎛️ **Professional Mixing**
- Configurable crossfade duration
- Audio normalization
- Support for multiple audio formats
- High-quality output

## Installation

1. **Clone or download this script**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Supported Audio Formats
- MP3
- WAV
- FLAC
- M4A
- AAC
- OGG

## Usage

### Basic Usage
```bash
python automix.py /path/to/your/music/folder
```

### Advanced Options
```bash
python automix.py /path/to/music -o "my_mix.wav" -c 6.0 -s 48000
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_folder` | Path to folder containing audio tracks | Required |
| `-o, --output` | Output filename | `automix_output.wav` |
| `-c, --crossfade` | Crossfade duration in seconds | `8.0` |
| `-s, --sample-rate` | Sample rate for processing | `44100` |

## Examples

### Create a mix with default settings:
```bash
python automix.py ~/Music/MyPlaylist
```

### Create a mix with custom crossfade and output name:
```bash
python automix.py ~/Music/PartyMix -o "party_night_mix.wav" -c 10.0
```

### High-quality mix with longer crossfades:
```bash
python automix.py ~/Music/ChillOut -o "chill_mix.wav" -c 12.0 -s 48000
```

## How It Works

### 1. **Audio Analysis**
The script analyzes each track for:
- **Tempo (BPM)**: For beat-matching
- **Musical Key**: For harmonic compatibility
- **Energy Level**: For smooth energy transitions
- **Spectral Characteristics**: For timbral matching

### 2. **Smart Ordering**
Tracks are automatically reordered using a compatibility algorithm that considers:
- Tempo similarity
- Key relationships (circle of fifths)
- Energy level progression
- Spectral characteristics

### 3. **Seamless Transitions**
- **Equal-power crossfading**: Maintains consistent volume
- **Tempo adjustment**: Slight tempo changes for better sync
- **Beat alignment**: Crossfades happen at musical boundaries
- **Audio normalization**: Prevents clipping and volume jumps

## Output

The script creates:
- A single mixed audio file with all tracks seamlessly blended
- Detailed logging showing the mixing process
- Compatibility scores for each transition

## Tips for Best Results

### 📁 **Organize Your Music**
- Use tracks from similar genres for best results
- Ensure good audio quality (avoid heavily compressed files)
- Include 4-8 tracks for optimal mix length

### 🎵 **Track Selection**
- Similar BPM ranges work best (±20 BPM)
- Tracks in related keys create smoother transitions
- Consider energy progression (start calm, build energy, wind down)

### ⚙️ **Settings Optimization**
- **Short crossfades (4-6s)**: For energetic, quick mixes
- **Long crossfades (10-15s)**: For ambient, chill mixes
- **Higher sample rates**: For audiophile-quality output

## Troubleshooting

### Common Issues

**"No audio files found"**
- Check that your folder contains supported audio formats
- Verify folder path is correct

**"Error analyzing [filename]"**
- File may be corrupted or in unsupported format
- Try with different audio files

**Memory issues with large files**
- Process smaller batches of tracks
- Use lower sample rates for testing

### Performance Tips
- Use WAV/FLAC for best quality
- Limit to 10-15 tracks per mix for reasonable processing time
- Close other applications if memory usage is high

## Technical Details

### Dependencies
- **librosa**: Audio analysis and processing
- **soundfile**: Audio file I/O
- **numpy**: Numerical computations
- **scipy**: Signal processing

### Algorithm Overview
1. **Feature Extraction**: Extract musical features from each track
2. **Compatibility Matrix**: Calculate transition scores between all track pairs
3. **Optimal Ordering**: Use greedy algorithm to find best track sequence
4. **Crossfade Generation**: Create smooth transitions with equal-power curves
5. **Final Assembly**: Combine all tracks with transitions into final mix

## License

This script is provided as-is for educational and personal use. Ensure you have rights to the audio files you're processing.

## Contributing

Feel free to improve the script by:
- Adding more sophisticated beat-matching
- Implementing different crossfade curves
- Adding support for more audio formats
- Improving the track ordering algorithm

---

*Enjoy creating professional-quality mixes with the power of Python! 🎧*