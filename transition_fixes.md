# Fixing Transition Smoothness Issues in Automix

## Common Problems and Solutions

### 1. **Tempo/BPM Transition Issues**

**Problem**: Abrupt tempo changes or pitch artifacts
**Solutions**:
- Use longer crossfades (15-25 seconds for big tempo differences)
- Implement gradual tempo curves instead of linear changes
- Add more micro-segments for tempo transitions
- Use phase vocoder for better pitch preservation

### 2. **Volume/Level Mismatches**

**Problem**: Volume jumps between tracks
**Solutions**:
- Normalize all tracks to same LUFS level (-23 LUFS broadcast standard)
- Use RMS-based volume matching
- Apply gentle compression during crossfades
- Implement look-ahead peak limiting

### 3. **Key/Harmonic Clashes**

**Problem**: Musical dissonance during transitions
**Solutions**:
- Analyze key compatibility before mixing
- Use circle of fifths for harmonic mixing
- Apply EQ to reduce clashing frequencies
- Choose transition points at harmonic intervals

### 4. **Beat Alignment Issues**

**Problem**: Rhythmic misalignment causing "train-wreck" effect
**Solutions**:
- Detect beat grid accurately
- Align crossfade points to strong beats
- Use beat-locked crossfade timing
- Apply micro-timing adjustments

### 5. **Frequency Conflicts**

**Problem**: Muddy or harsh sound during crossfades
**Solutions**:
- Apply frequency-aware crossfading
- Use EQ to separate tracks in frequency spectrum
- Implement multi-band crossfading
- Add gentle low-pass filtering

## Enhanced Implementation Fixes

### Fix 1: Improved Tempo Transitions
```python
# Use exponential curves instead of linear
def smooth_tempo_curve(progress):
    # Exponential ease-in-out
    if progress < 0.5:
        return 2 * progress * progress
    else:
        return 1 - 2 * (1 - progress) * (1 - progress)
```

### Fix 2: Better Beat Alignment
```python
# Find nearest strong beat for crossfade start
def find_optimal_crossfade_point(beats, target_time, crossfade_duration):
    # Look for downbeats (every 4th beat)
    strong_beats = beats[::4]  # Assuming 4/4 time
    best_beat = min(strong_beats, key=lambda x: abs(x - target_time))
    return best_beat
```

### Fix 3: Volume Envelope Matching
```python
# Match volume envelopes during crossfade
def match_volume_envelopes(track1_end, track2_start):
    # Analyze volume envelope of each track
    envelope1 = librosa.feature.rms(track1_end, hop_length=512)
    envelope2 = librosa.feature.rms(track2_start, hop_length=512)
    
    # Apply envelope matching
    return apply_envelope_matching(track1_end, track2_start, envelope1, envelope2)
```

### Fix 4: Frequency-Aware Crossfading
```python
# Separate frequency bands for crossfading
def multi_band_crossfade(track1, track2, fade_curve):
    # Split into frequency bands
    low1, mid1, high1 = split_frequency_bands(track1)
    low2, mid2, high2 = split_frequency_bands(track2)
    
    # Apply different fade curves to each band
    # Bass fades slower, highs fade faster
    low_mixed = crossfade_band(low1, low2, fade_curve ** 0.5)
    mid_mixed = crossfade_band(mid1, mid2, fade_curve)
    high_mixed = crossfade_band(high1, high2, fade_curve ** 2)
    
    return low_mixed + mid_mixed + high_mixed
```

## Quick Fixes to Try

### 1. Increase Crossfade Duration
```bash
python automix.py tracks -c 20.0  # Try 20-second crossfades
```

### 2. Use Different Tracks Order
The script automatically reorders tracks, but you can test with more compatible tracks:
- Similar BPM (within 10-15 BPM)
- Compatible keys (same key, fifth relationship, or relative minor/major)
- Similar energy levels

### 3. Adjust Tempo Sync Threshold
Currently set to 3 BPM difference. You can modify this in the code:
```python
if abs(tempo_change) < 1:  # Make it more sensitive
```

### 4. Add Pre-Processing
- Use high-quality source files (WAV/FLAC instead of MP3)
- Ensure consistent sample rates
- Remove silence from track beginnings/endings

## Diagnostic Commands

### Check Track Compatibility
```bash
# Add debug logging to see compatibility scores
python automix.py tracks -c 15.0 2>&1 | grep "Compatibility"
```

### Analyze Tempo Differences
```bash
# Look for large tempo jumps
python automix.py tracks -c 15.0 2>&1 | grep "BPM"
```

### Monitor Volume Levels
```bash
# Check for volume normalization messages
python automix.py tracks -c 15.0 2>&1 | grep "volume\|normalization\|limiting"
```

## Advanced Solutions

### 1. Manual Track Preparation
- Trim tracks to remove long intros/outros
- Apply EQ to problematic frequencies
- Pre-normalize tracks to same loudness

### 2. Custom Crossfade Points
- Analyze tracks manually to find best transition points
- Use audio editing software to mark optimal crossfade regions
- Apply custom fade curves for each transition

### 3. Multi-Pass Processing
- First pass: Basic tempo matching
- Second pass: Fine-tune crossfade points
- Third pass: Apply frequency-aware mixing

## Testing Different Settings

Try these combinations:

### For Similar Tempo Tracks:
```bash
python automix.py tracks -c 8.0   # Shorter crossfades
```

### For Very Different Tempos:
```bash
python automix.py tracks -c 25.0  # Much longer crossfades
```

### For Electronic/Dance Music:
- Use beat-grid aligned crossfades
- Shorter crossfades (4-8 seconds)
- More aggressive tempo matching

### For Acoustic/Folk Music:
- Longer crossfades (15-30 seconds)
- Gentler tempo changes
- Focus on harmonic compatibility

Let me know which specific issues you're experiencing and I can provide more targeted fixes!