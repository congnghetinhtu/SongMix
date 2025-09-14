#!/usr/bin/env python3
"""
Apple Music-style Automix Script
Creates seamless transitions between audio tracks in a folder
"""

import os
import sys
import argparse
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from scipy import signal
from scipy.interpolate import interp1d

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoMixer:
    def __init__(self, input_folder: str, output_file: str = "automix_output.wav", 
                 crossfade_duration: float = 8.0, sample_rate: int = 44100):
        """
        Initialize the AutoMixer
        
        Args:
            input_folder: Path to folder containing audio tracks
            output_file: Output file name for the mixed result
            crossfade_duration: Duration of crossfade between tracks (seconds)
            sample_rate: Target sample rate for processing
        """
        self.input_folder = Path(input_folder)
        self.output_file = output_file
        self.crossfade_duration = crossfade_duration
        self.sample_rate = sample_rate
        self.supported_formats = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
        
    def get_audio_files(self) -> List[Path]:
        """Get all supported audio files from the input folder"""
        audio_files = []
        for file_path in self.input_folder.iterdir():
            if file_path.suffix.lower() in self.supported_formats:
                audio_files.append(file_path)
        
        # Sort files alphabetically
        audio_files.sort(key=lambda x: x.name.lower())
        logger.info(f"Found {len(audio_files)} audio files")
        return audio_files
    
    def analyze_audio(self, file_path: Path) -> Dict:
        """
        Analyze audio file for mixing parameters with enhanced volume normalization
        
        Returns:
            Dictionary containing tempo, key, energy, beats, and other features
        """
        try:
            logger.info(f"Analyzing: {file_path.name}")
            
            # Load audio file
            y, sr = librosa.load(str(file_path), sr=self.sample_rate)
            
            # Normalize volume to consistent level before analysis
            y_normalized = self._normalize_audio(y)
            
            # Enhanced tempo and beat detection
            tempo, beats = librosa.beat.beat_track(y=y_normalized, sr=sr, units='time')
            beat_frames = librosa.beat.beat_track(y=y_normalized, sr=sr, units='frames')[1]
            
            # Key detection using chroma features and key profiles
            chroma = librosa.feature.chroma_stft(y=y_normalized, sr=sr)
            key_profile = np.mean(chroma, axis=1)
            key = np.argmax(key_profile)
            
            # Enhanced energy analysis with RMS and spectral features
            rms = librosa.feature.rms(y=y_normalized, frame_length=2048, hop_length=512)[0]
            energy = np.mean(rms)
            energy_variation = np.std(rms)
            
            # Spectral features for mixing compatibility
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_normalized, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_normalized, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_normalized, sr=sr))
            
            # Zero crossing rate for rhythm analysis
            zcr = np.mean(librosa.feature.zero_crossing_rate(y_normalized))
            
            # MFCC for timbral analysis
            mfccs = librosa.feature.mfcc(y=y_normalized, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Vocal detection using multiple techniques
            vocal_segments = self._detect_vocals(y_normalized, sr)
            
            # Onset detection for transition points
            onset_frames = librosa.onset.onset_detect(y=y_normalized, sr=sr, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Find potential intro/outro sections for smooth flow
            intro_end, outro_start = self._detect_intro_outro(y_normalized, sr, beats)
            
            # Calculate peak and RMS levels for volume matching
            peak_level = np.max(np.abs(y_normalized))
            rms_level = np.sqrt(np.mean(y_normalized**2))
            
            return {
                'file_path': file_path,
                'duration': len(y_normalized) / sr,
                'tempo': float(tempo),
                'beats': beats,
                'beat_frames': beat_frames,
                'key': int(key),
                'energy': float(energy),
                'energy_variation': float(energy_variation),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'spectral_bandwidth': float(spectral_bandwidth),
                'zcr': float(zcr),
                'mfcc_mean': mfcc_mean,
                'vocal_segments': vocal_segments,
                'onset_times': onset_times,
                'intro_end': intro_end,
                'outro_start': outro_start,
                'peak_level': float(peak_level),
                'rms_level': float(rms_level),
                'audio_data': y_normalized,  # Use normalized audio
                'sample_rate': sr
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path.name}: {str(e)}")
            return None
    
    def _detect_vocals(self, y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        Detect vocal segments using spectral analysis and harmonic-percussive separation
        """
        try:
            # Harmonic-percussive separation to isolate vocals
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Compute spectral features that indicate vocal presence
            # 1. Spectral centroid (vocals typically have higher centroids)
            spec_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
            
            # 2. Spectral rolloff (vocals have characteristic rolloff patterns)
            spec_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr, roll_percent=0.85)[0]
            
            # 3. Chroma features (vocals often follow harmonic progressions)
            chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
            chroma_strength = np.sum(chroma, axis=0)
            
            # 4. Zero crossing rate (speech-like patterns)
            zcr = librosa.feature.zero_crossing_rate(y_harmonic)[0]
            
            # 5. MFCCs (vocal timbre characteristics)
            mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=5)
            mfcc_var = np.var(mfccs, axis=0)
            
            # Create vocal probability score
            # Normalize features
            spec_centroid_norm = (spec_centroid - np.mean(spec_centroid)) / (np.std(spec_centroid) + 1e-8)
            chroma_strength_norm = (chroma_strength - np.mean(chroma_strength)) / (np.std(chroma_strength) + 1e-8)
            mfcc_var_norm = (mfcc_var - np.mean(mfcc_var)) / (np.std(mfcc_var) + 1e-8)
            
            # Vocal probability based on multiple features
            vocal_prob = (
                np.clip(spec_centroid_norm * 0.3, -1, 1) +  # Higher centroid suggests vocals
                np.clip(chroma_strength_norm * 0.3, -1, 1) + # Strong harmonic content
                np.clip(mfcc_var_norm * 0.4, -1, 1)          # Vocal timbre variation
            ) / 3.0
            
            # Apply smoothing to reduce noise
            if len(vocal_prob) > 10:
                from scipy import signal
                window_size = min(21, len(vocal_prob) // 5)
                if window_size >= 5:
                    vocal_prob = signal.savgol_filter(vocal_prob, window_size | 1, 2)
            
            # Convert frame indices to time
            hop_length = 512
            frame_times = librosa.frames_to_time(np.arange(len(vocal_prob)), sr=sr, hop_length=hop_length)
            
            # Find vocal segments (threshold for vocal detection)
            vocal_threshold = 0.1  # Lower threshold for better detection
            vocal_frames = vocal_prob > vocal_threshold
            
            # Find continuous vocal segments
            vocal_segments = []
            in_vocal = False
            start_time = 0
            
            for i, is_vocal in enumerate(vocal_frames):
                current_time = frame_times[i] if i < len(frame_times) else frame_times[-1]
                
                if is_vocal and not in_vocal:
                    # Start of vocal segment
                    start_time = current_time
                    in_vocal = True
                elif not is_vocal and in_vocal:
                    # End of vocal segment
                    if current_time - start_time > 1.0:  # Keep segments longer than 1 second
                        vocal_segments.append((start_time, current_time))
                    in_vocal = False
            
            # Handle case where track ends during vocal
            if in_vocal and len(frame_times) > 0:
                if frame_times[-1] - start_time > 1.0:
                    vocal_segments.append((start_time, frame_times[-1]))
            
            logger.info(f"    Detected {len(vocal_segments)} vocal segments")
            return vocal_segments
            
        except Exception as e:
            logger.warning(f"    Vocal detection failed: {e}, using fallback")
            # Fallback: assume vocals in middle sections of track
            duration = len(y) / sr
            return [(duration * 0.2, duration * 0.4), (duration * 0.6, duration * 0.8)]

    def _detect_intro_outro(self, y: np.ndarray, sr: int, beats: np.ndarray) -> Tuple[float, float]:
        """
        Detect intro and outro sections for better transition points
        """
        duration = len(y) / sr
        
        # Simple heuristic: assume intro is first 16-32 beats, outro is last 16-32 beats
        if len(beats) > 32:
            intro_end = beats[min(16, len(beats)//4)]
            outro_start = beats[max(-16, -len(beats)//4)]
        else:
            intro_end = duration * 0.15  # 15% of track
            outro_start = duration * 0.85  # 85% of track
            
        return intro_end, outro_start
    
    def _normalize_audio(self, audio: np.ndarray, target_lufs: float = -20.0) -> np.ndarray:
        """
        Normalize audio to stable volume level with advanced dynamics preservation
        """
        # Calculate multiple volume metrics for better stability
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        if rms > 0 and peak > 0:
            # Target RMS level (more conservative than broadcast standard)
            target_rms = 10**(target_lufs/20)
            
            # Calculate crest factor (peak-to-RMS ratio) to preserve dynamics
            crest_factor = peak / rms
            
            # Calculate initial gain
            gain = target_rms / rms
            
            # Limit gain based on crest factor to preserve dynamics
            max_gain = 0.9 / peak  # Never exceed 90% of maximum
            gain = min(gain, max_gain)
            
            # Apply very conservative gain limiting
            if gain > 2.0:  # Never amplify more than 6dB
                gain = 2.0
            elif gain < 0.3:  # Never attenuate more than -10dB
                gain = 0.3
                
            # Apply gain smoothly
            normalized = audio * gain
            
            # Advanced soft limiting with smooth curve
            new_peak = np.max(np.abs(normalized))
            if new_peak > 0.85:
                # Soft compression curve instead of hard limiting
                threshold = 0.85
                ratio = 0.3  # Gentle 3:1 compression above threshold
                
                # Apply smooth compression
                mask = np.abs(normalized) > threshold
                excess = np.abs(normalized) - threshold
                compressed_excess = excess * ratio
                
                # Apply compression while preserving sign
                compressed = np.where(mask,
                    np.sign(normalized) * (threshold + compressed_excess),
                    normalized
                )
                
                normalized = compressed
                
            return normalized
        else:
            return audio
    
    def _ensure_smooth_flow(self, track1: Dict, track2: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preserve full song duration while optimizing vocal-to-vocal transitions
        """
        audio1 = track1['audio_data']
        audio2 = track2['audio_data']
        sr = track1['sample_rate']
        
        # Get vocal segments for both tracks
        vocal_segments1 = track1.get('vocal_segments', [])
        vocal_segments2 = track2.get('vocal_segments', [])
        
        # Find optimal vocal-to-vocal transition points
        crossfade_start_time, intro_skip_time = self._find_vocal_transition_points(
            vocal_segments1, vocal_segments2, len(audio1) / sr, len(audio2) / sr
        )
        
        # Convert to samples
        intro_skip_samples = int(intro_skip_time * sr)
        
        # Keep full songs with minimal intro skip for vocal alignment
        full_audio1 = audio1  # Keep complete first song
        vocal_aligned_audio2 = audio2[intro_skip_samples:] if intro_skip_samples > 0 else audio2
        
        logger.info(f"    Vocal-to-vocal transition: Crossfade at {crossfade_start_time:.1f}s, vocal intro skip {intro_skip_time:.1f}s")
        
        return full_audio1, vocal_aligned_audio2
    
    def _find_vocal_transition_points(self, vocal_segments1: List[Tuple[float, float]], 
                                    vocal_segments2: List[Tuple[float, float]], 
                                    duration1: float, duration2: float) -> Tuple[float, float]:
        """
        Find optimal transition points for vocal-to-vocal crossfading
        """
        # Default fallback positions
        default_crossfade_start = duration1 - self.crossfade_duration
        default_intro_skip = 0.0
        
        if not vocal_segments1 or not vocal_segments2:
            # If no vocals detected in one track, use fallback but still log info
            if vocal_segments1:
                logger.info(f"    Track1 has {len(vocal_segments1)} vocal segments, Track2 instrumental")
            elif vocal_segments2:
                logger.info(f"    Track1 instrumental, Track2 has {len(vocal_segments2)} vocal segments")
            else:
                logger.info(f"    Both tracks appear instrumental, using standard transition")
            return default_crossfade_start, default_intro_skip
        
        # Find vocal segments in the outro section of track1
        outro_start_time = duration1 * 0.7  # Look for vocals in last 30% of track
        track1_outro_vocals = [
            (start, end) for start, end in vocal_segments1 
            if start >= outro_start_time and end <= duration1
        ]
        
        # Find vocal segments in the intro section of track2
        intro_end_time = duration2 * 0.3  # Look for vocals in first 30% of track
        track2_intro_vocals = [
            (start, end) for start, end in vocal_segments2 
            if start >= 0 and end <= intro_end_time
        ]
        
        if not track1_outro_vocals and not track2_intro_vocals:
            logger.info(f"    No outro/intro vocals found, using standard transition")
            return default_crossfade_start, default_intro_skip
        
        # Strategy 1: Vocal outro to vocal intro
        if track1_outro_vocals and track2_intro_vocals:
            # Find the last vocal segment in track1 outro
            last_outro_vocal = max(track1_outro_vocals, key=lambda x: x[1])
            # Find the first vocal segment in track2 intro
            first_intro_vocal = min(track2_intro_vocals, key=lambda x: x[0])
            
            # Time crossfade to blend vocals naturally
            crossfade_start = last_outro_vocal[0] + (last_outro_vocal[1] - last_outro_vocal[0]) * 0.7
            intro_skip = max(0, first_intro_vocal[0] - 1.0)  # Start 1 second before vocals
            
            logger.info(f"    Vocal-to-vocal: outro vocal at {last_outro_vocal[0]:.1f}-{last_outro_vocal[1]:.1f}s, intro vocal at {first_intro_vocal[0]:.1f}-{first_intro_vocal[1]:.1f}s")
            return crossfade_start, intro_skip
        
        # Strategy 2: Vocal outro to instrumental intro (let vocals finish)
        elif track1_outro_vocals and not track2_intro_vocals:
            last_outro_vocal = max(track1_outro_vocals, key=lambda x: x[1])
            # Start crossfade near end of last vocal
            crossfade_start = last_outro_vocal[1] - self.crossfade_duration * 0.3
            
            logger.info(f"    Vocal outro to instrumental: outro vocal ends at {last_outro_vocal[1]:.1f}s")
            return crossfade_start, 0.0
        
        # Strategy 3: Instrumental outro to vocal intro (prepare for vocals)
        elif not track1_outro_vocals and track2_intro_vocals:
            first_intro_vocal = min(track2_intro_vocals, key=lambda x: x[0])
            # Start crossfade earlier to build up to vocals
            crossfade_start = duration1 - self.crossfade_duration * 1.2
            intro_skip = max(0, first_intro_vocal[0] - 2.0)  # Start 2 seconds before vocals
            
            logger.info(f"    Instrumental to vocal intro: intro vocal starts at {first_intro_vocal[0]:.1f}s")
            return crossfade_start, intro_skip
        
        # Fallback to standard transition
        return default_crossfade_start, default_intro_skip
    
    def calculate_compatibility(self, track1: Dict, track2: Dict) -> float:
        """
        Calculate compatibility score between two tracks (0-1) with enhanced metrics
        Higher score means better transition
        """
        if not track1 or not track2:
            return 0.0
        
        # Tempo compatibility (prefer similar tempos or harmonic ratios)
        tempo1, tempo2 = track1['tempo'], track2['tempo']
        tempo_ratio = max(tempo1, tempo2) / min(tempo1, tempo2)
        
        # Check for harmonic ratios (2:1, 3:2, 4:3)
        if abs(tempo_ratio - 2.0) < 0.1 or abs(tempo_ratio - 1.5) < 0.1 or abs(tempo_ratio - 1.33) < 0.1:
            tempo_score = 0.9  # High score for harmonic ratios
        else:
            tempo_diff = abs(tempo1 - tempo2)
            tempo_score = max(0, 1 - (tempo_diff / 30))  # Tighter tolerance
        
        # Enhanced key compatibility using circle of fifths
        key1, key2 = track1['key'], track2['key']
        key_distance = min(abs(key1 - key2), 12 - abs(key1 - key2))
        
        # Perfect match, fifth, or relative minor/major
        if key_distance == 0:
            key_score = 1.0
        elif key_distance == 7 or key_distance == 5:  # Fifth relationship
            key_score = 0.8
        elif key_distance == 3 or key_distance == 9:  # Relative minor/major
            key_score = 0.7
        else:
            key_score = max(0, 1 - (key_distance / 6))
        
        # Energy compatibility with variation consideration
        energy_diff = abs(track1['energy'] - track2['energy'])
        max_energy = max(track1['energy'], track2['energy'], 0.1)
        energy_score = max(0, 1 - (energy_diff / max_energy))
        
        # Energy variation compatibility (smoother transitions)
        energy_var_diff = abs(track1['energy_variation'] - track2['energy_variation'])
        energy_var_score = max(0, 1 - energy_var_diff)
        
        # Spectral compatibility (multiple features)
        spectral_centroid_diff = abs(track1['spectral_centroid'] - track2['spectral_centroid'])
        spectral_score = max(0, 1 - (spectral_centroid_diff / 2000))
        
        # Timbral compatibility using MFCC
        mfcc_distance = np.linalg.norm(track1['mfcc_mean'] - track2['mfcc_mean'])
        timbral_score = max(0, 1 - (mfcc_distance / 50))
        
        # Rhythm compatibility using ZCR
        zcr_diff = abs(track1['zcr'] - track2['zcr'])
        rhythm_score = max(0, 1 - (zcr_diff / 0.1))
        
        # Weighted average with Apple Music-style priorities
        compatibility = (
            tempo_score * 0.30 +      # Tempo is crucial
            key_score * 0.25 +        # Key harmony important
            energy_score * 0.20 +     # Energy flow
            spectral_score * 0.10 +   # Timbre matching
            timbral_score * 0.10 +    # MFCC timbral
            rhythm_score * 0.05       # Rhythm consistency
        )
        
        return min(1.0, compatibility)
    
    def create_crossfade(self, track1_audio: np.ndarray, track2_audio: np.ndarray, 
                        crossfade_samples: int, track1_tempo: float = None, track2_tempo: float = None) -> np.ndarray:
        """
        Create an Apple Music-style crossfade with tempo adjustment only during transition
        """
        # Ensure we don't exceed track lengths
        crossfade_samples = min(crossfade_samples, len(track1_audio), len(track2_audio))
        
        # Get crossfade sections
        track1_end = track1_audio[-crossfade_samples:].copy()
        track2_start = track2_audio[:crossfade_samples].copy()
        
        # Apply tempo adjustment only to crossfade sections if needed
        if track1_tempo and track2_tempo and abs(track1_tempo - track2_tempo) > 1.5:  # Higher threshold
            tempo_direction = "invisible" if abs(track1_tempo - track2_tempo) < 4 else ("increasing" if track2_tempo > track1_tempo else "decreasing")
            logger.info(f"  Creating {tempo_direction} tempo transition: {track1_tempo:.1f} → {track2_tempo:.1f} BPM")
            
            # Use invisible tempo sync for most differences, avoid gradual for small differences
            if abs(track1_tempo - track2_tempo) < 10:  # Use invisible sync for most cases
                adjusted_track1_end = self._apply_invisible_tempo_sync(
                    track1_end, track1_tempo, track2_tempo, is_outro=True
                )
                adjusted_track2_start = self._apply_invisible_tempo_sync(
                    track2_start, track2_tempo, track1_tempo, is_outro=False
                )
            else:
                # For very large differences, still use invisible but with more steps
                logger.info(f"  Large tempo difference, using multi-step invisible sync")
                # Split the tempo change into smaller steps
                mid_tempo = (track1_tempo + track2_tempo) / 2
                adjusted_track1_end = self._apply_invisible_tempo_sync(
                    track1_end, track1_tempo, mid_tempo, is_outro=True
                )
                adjusted_track2_start = self._apply_invisible_tempo_sync(
                    track2_start, track2_tempo, mid_tempo, is_outro=False
                )
        else:
            logger.info(f"  Tempo difference minimal ({abs(track1_tempo - track2_tempo):.1f} BPM), preserving natural flow")
            adjusted_track1_end = track1_end
            adjusted_track2_start = track2_start
        
        # Create ultra-smooth invisible fade curves with advanced smoothing
        fade_curve = np.linspace(0, 1, crossfade_samples)
        
        # Use advanced equal-power crossfading for natural sound
        # Apply gentle S-curve for more natural perception
        def invisible_s_curve(x):
            # Ultra-smooth S-curve that's nearly imperceptible
            return x * x * x * (x * (x * 6 - 15) + 10)  # Quintic smoothstep
        
        # Apply the invisible S-curve to create natural fades
        smooth_fade = np.array([invisible_s_curve(f) for f in fade_curve])
        
        # Create equal-power crossfade that preserves energy
        fade_out = np.cos(smooth_fade * np.pi / 2)  # Cosine fade out
        fade_in = np.sin(smooth_fade * np.pi / 2)   # Sine fade in
        
        # Apply additional smoothing passes for invisible transitions
        if crossfade_samples > 256:
            # Multi-pass smoothing with decreasing intensity
            for pass_num in range(3):
                window_size = max(5, min(41, crossfade_samples // (20 + pass_num * 10)))
                if window_size >= 5:
                    fade_out = signal.savgol_filter(fade_out, window_size | 1, 2)
                    fade_in = signal.savgol_filter(fade_in, window_size | 1, 2)
            
            # Ensure fade curves maintain proper bounds
            fade_out = np.clip(fade_out, 0, 1)
            fade_in = np.clip(fade_in, 0, 1)
            
            # Force exact start and end points
            fade_out[0] = 1.0
            fade_out[-1] = 0.0
            fade_in[0] = 0.0
            fade_in[-1] = 1.0
        
        # Apply micro-ramping only at the very edges to eliminate clicks
        ramp_samples = min(32, crossfade_samples // 32)  # Smaller ramps
        if ramp_samples > 0:
            # Ultra-gentle ramps using raised cosine
            ramp_curve = (1 - np.cos(np.linspace(0, np.pi, ramp_samples))) / 2
            
            # Apply to start
            fade_out[:ramp_samples] *= ramp_curve
            fade_in[:ramp_samples] *= ramp_curve
            
            # Apply to end (reverse curve)
            fade_out[-ramp_samples:] *= ramp_curve[::-1]
            fade_in[-ramp_samples:] *= ramp_curve[::-1]
        
        # Apply vocal-aware frequency crossfading
        crossfade_section = self._vocal_aware_crossfade(
            adjusted_track1_end, adjusted_track2_start, fade_out, fade_in
        )
        
        # Combine: track1 (original BPM, without crossfade section) + crossfade + track2 (original BPM, without crossfade section)
        result = np.concatenate([
            track1_audio[:-crossfade_samples],  # Keep original BPM
            crossfade_section,                  # Tempo-synced transition
            track2_audio[crossfade_samples:]    # Keep original BPM
        ])
        
        return result
    
    def _vocal_aware_crossfade(self, track1_end: np.ndarray, track2_start: np.ndarray,
                              fade_out: np.ndarray, fade_in: np.ndarray) -> np.ndarray:
        """
        Apply vocal-aware crossfading with intelligent frequency band processing
        """
        # Normalize audio levels before mixing for stable volume
        track1_rms_pre = np.sqrt(np.mean(track1_end**2)) if len(track1_end) > 0 else 0
        track2_rms_pre = np.sqrt(np.mean(track2_start**2)) if len(track2_start) > 0 else 0
        
        # Target stable volume level
        target_level = max(track1_rms_pre, track2_rms_pre) * 0.95
        
        # Normalize both tracks to similar levels before crossfading
        if track1_rms_pre > 0:
            track1_normalized = track1_end * (target_level / track1_rms_pre)
        else:
            track1_normalized = track1_end
            
        if track2_rms_pre > 0:
            track2_normalized = track2_start * (target_level / track2_rms_pre)
        else:
            track2_normalized = track2_start
        
        try:
            # Separate into vocal and non-vocal frequency bands for smart mixing
            # Vocal range: ~80 Hz - 15 kHz with emphasis on 200 Hz - 8 kHz
            
            # Low frequencies (below 200 Hz) - bass, kick, sub-bass
            sos_low = signal.butter(4, 200 / (self.sample_rate / 2), btype='low', output='sos')
            track1_low = signal.sosfilt(sos_low, track1_normalized)
            track2_low = signal.sosfilt(sos_low, track2_normalized)
            
            # Vocal frequencies (200 Hz - 4 kHz) - primary vocal range  
            sos_vocal = signal.butter(4, [200, 4000] / (self.sample_rate / 2), btype='band', output='sos')
            track1_vocal = signal.sosfilt(sos_vocal, track1_normalized)
            track2_vocal = signal.sosfilt(sos_vocal, track2_normalized)
            
            # Mid-high frequencies (4 kHz - 8 kHz) - vocal presence, instruments
            sos_mid_high = signal.butter(4, [4000, 8000] / (self.sample_rate / 2), btype='band', output='sos')
            track1_mid_high = signal.sosfilt(sos_mid_high, track1_normalized)
            track2_mid_high = signal.sosfilt(sos_mid_high, track2_normalized)
            
            # High frequencies (above 8 kHz) - air, cymbals, vocal harmonics
            sos_high = signal.butter(4, 8000 / (self.sample_rate / 2), btype='high', output='sos')
            track1_high = signal.sosfilt(sos_high, track1_normalized)
            track2_high = signal.sosfilt(sos_high, track2_normalized)
            
            # Apply different crossfade strategies per frequency band
            
            # Low frequencies: Standard equal-power crossfade
            low_mixed = track1_low * fade_out + track2_low * fade_in
            
            # Vocal frequencies: Intelligent vocal crossfade
            vocal_mixed = self._intelligent_vocal_crossfade(
                track1_vocal, track2_vocal, fade_out, fade_in
            )
            
            # Mid-high frequencies: Gentle crossfade with slight emphasis on incoming track
            mid_high_fade_in_emphasized = fade_in ** 0.8  # Slightly faster fade in
            mid_high_mixed = track1_mid_high * fade_out + track2_mid_high * mid_high_fade_in_emphasized
            
            # High frequencies: Quick transition to preserve clarity
            high_transition_point = len(fade_out) // 2
            high_fade_out = fade_out.copy()
            high_fade_in = fade_in.copy()
            high_fade_out[high_transition_point:] *= 0.5  # Faster fade out
            high_fade_in[:high_transition_point] *= 0.5   # Faster fade in
            high_mixed = track1_high * high_fade_out + track2_high * high_fade_in
            
            # Recombine all frequency bands
            crossfade_section = low_mixed + vocal_mixed + mid_high_mixed + high_mixed
            
        except Exception as e:
            logger.debug(f"Vocal-aware crossfade failed: {e}, using standard crossfade")
            # Fallback to standard crossfade
            track1_faded = track1_normalized * fade_out
            track2_faded = track2_normalized * fade_in
            crossfade_section = track1_faded + track2_faded
        
        # Apply dynamic volume stabilization throughout the crossfade
        window_size = min(4096, len(crossfade_section) // 8)
        if window_size > 512:
            # Analyze volume in overlapping windows
            hop_size = window_size // 4
            stable_crossfade = crossfade_section.copy()
            
            for i in range(0, len(crossfade_section) - window_size, hop_size):
                window = crossfade_section[i:i + window_size]
                window_rms = np.sqrt(np.mean(window**2))
                
                if window_rms > 0:
                    # Apply gentle volume correction only if needed
                    volume_diff = abs(window_rms - target_level) / target_level
                    if volume_diff > 0.1:  # Only correct significant differences
                        correction_factor = target_level / window_rms
                        # Limit correction to prevent artifacts
                        correction_factor = np.clip(correction_factor, 0.8, 1.2)
                        
                        # Apply gradual correction with smooth windowing
                        window_corrected = window * correction_factor
                        
                        # Blend correction smoothly
                        blend_window = np.hanning(window_size)
                        stable_crossfade[i:i + window_size] = (
                            stable_crossfade[i:i + window_size] * (1 - blend_window) +
                            window_corrected * blend_window
                        )
            
            crossfade_section = stable_crossfade
        
        # Final gentle limiting to prevent clipping (very conservative)
        peak = np.max(np.abs(crossfade_section))
        if peak > 0.9:
            # Very gentle soft limiting with smooth curve
            limit_ratio = 0.9 / peak
            crossfade_section = crossfade_section * limit_ratio
        
        return crossfade_section
    
    def _intelligent_vocal_crossfade(self, vocal1: np.ndarray, vocal2: np.ndarray,
                                   fade_out: np.ndarray, fade_in: np.ndarray) -> np.ndarray:
        """
        Intelligent crossfade specifically for vocal frequency ranges
        """
        # Detect vocal energy in both tracks
        vocal1_energy = np.sqrt(np.mean(vocal1**2))
        vocal2_energy = np.sqrt(np.mean(vocal2**2))
        
        # If one track has significantly more vocal energy, adjust crossfade
        energy_ratio = vocal2_energy / (vocal1_energy + 1e-8)
        
        if energy_ratio > 2.0:
            # Track 2 has much stronger vocals - fade in faster
            adjusted_fade_in = fade_in ** 0.7
            adjusted_fade_out = fade_out ** 1.3
        elif energy_ratio < 0.5:
            # Track 1 has much stronger vocals - fade out slower
            adjusted_fade_in = fade_in ** 1.3
            adjusted_fade_out = fade_out ** 0.7
        else:
            # Similar vocal energy - use equal-power crossfade
            adjusted_fade_in = fade_in
            adjusted_fade_out = fade_out
        
        # Apply crossfade with potential ducking in middle for clarity
        crossfaded = vocal1 * adjusted_fade_out + vocal2 * adjusted_fade_in
        
        # Apply gentle ducking in the middle of the crossfade to avoid vocal conflicts
        duck_start = len(crossfaded) // 3
        duck_end = 2 * len(crossfaded) // 3
        duck_curve = np.ones(len(crossfaded))
        
        # Create gentle ducking curve
        duck_amount = 0.85  # Reduce to 85% in the middle
        for i in range(duck_start, duck_end):
            position = (i - duck_start) / (duck_end - duck_start)
            # Bell curve for ducking
            duck_factor = 1.0 - (1.0 - duck_amount) * np.exp(-((position - 0.5) * 6)**2)
            duck_curve[i] = duck_factor
        
        crossfaded *= duck_curve
        
        return crossfaded
    
    def _gradual_tempo_sync(self, track1_end: np.ndarray, track2_start: np.ndarray,
                           track1_tempo: float, track2_tempo: float, crossfade_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ultra-smooth tempo curves with exponential easing for natural transitions
        """
        sr = self.sample_rate
        
        # Calculate the tempo change direction and amount
        tempo_change = track2_tempo - track1_tempo
        
        if abs(tempo_change) < 2:  # Even smaller threshold for smoother transitions
            return track1_end, track2_start
        
        logger.info(f"    Creating ultra-smooth tempo curves: {track1_tempo:.1f} → {track2_tempo:.1f} BPM")
        
        crossfade_duration = crossfade_samples / sr
        
        # Calculate beat periods for both tracks
        beat_period_1 = 60.0 / track1_tempo
        beat_period_2 = 60.0 / track2_tempo
        
        logger.info(f"    Beat periods: {beat_period_1:.3f}s → {beat_period_2:.3f}s")
        
        # Create time array for the crossfade duration
        time_points = np.linspace(0, crossfade_duration, crossfade_samples)
        progress = time_points / crossfade_duration
        
        # Use exponential ease-in-out curve for more natural feeling
        def exponential_ease_in_out(t):
            if t < 0.5:
                return 2 * t * t * t  # Slow start
            else:
                return 1 - 2 * (1 - t) ** 3  # Slow end
        
        # Apply the exponential curve
        smooth_curve = np.array([exponential_ease_in_out(p) for p in progress])
        
        # Add subtle musical timing modulation
        beats_in_crossfade = crossfade_duration / beat_period_1
        
        if beats_in_crossfade > 1:
            # Very subtle musical modulation (reduced from 5% to 2%)
            beat_phase = (time_points / beat_period_1) * 2 * np.pi
            musical_modulation = np.sin(beat_phase) * 0.02
            
            # Apply only in middle 60% of transition
            modulation_window = np.where(
                (progress >= 0.2) & (progress <= 0.8),
                np.sin((progress - 0.2) / 0.6 * np.pi) ** 2,
                0
            )
            smooth_curve += musical_modulation * modulation_window
        
        # Ensure curve stays within bounds
        smooth_curve = np.clip(smooth_curve, 0, 1)
        
        # Apply additional smoothing to eliminate any remaining artifacts
        if len(smooth_curve) > 32:
            window_size = min(15, len(smooth_curve) // 8)
            if window_size >= 3:
                smooth_curve = signal.savgol_filter(smooth_curve, window_size | 1, 2)
                smooth_curve = np.clip(smooth_curve, 0, 1)
        
        # Calculate instantaneous tempo with reduced intensity
        tempo_blend_factor = min(0.5, abs(tempo_change) / 40.0)  # Adaptive blending
        instantaneous_tempo = track1_tempo + tempo_change * smooth_curve * tempo_blend_factor
        
        # For track2: even gentler approach
        final_tempo_1 = track1_tempo + tempo_change * tempo_blend_factor
        track2_tempo_curve = final_tempo_1 + (track2_tempo - final_tempo_1) * smooth_curve * 0.6
        
        # Apply tempo stretching using the calculated curves
        adjusted_track1_end = self._apply_tempo_curve_gentle(track1_end, track1_tempo, instantaneous_tempo, sr)
        adjusted_track2_start = self._apply_tempo_curve_gentle(track2_start, track2_tempo, track2_tempo_curve, sr)
        
        # Ensure exact length match with high-quality resampling
        if len(adjusted_track1_end) != crossfade_samples:
            adjusted_track1_end = signal.resample(adjusted_track1_end, crossfade_samples)
        
        if len(adjusted_track2_start) != crossfade_samples:
            adjusted_track2_start = signal.resample(adjusted_track2_start, crossfade_samples)
        
        return adjusted_track1_end, adjusted_track2_start
    
    def _apply_tempo_curve_gentle(self, audio: np.ndarray, original_tempo: float, 
                                 tempo_curve: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply ultra-gentle tempo curve with overlap-add processing for smoothest results
        """
        try:
            # Calculate the average tempo ratio (gentler approach)
            tempo_ratios = tempo_curve / original_tempo
            avg_ratio = np.mean(tempo_ratios)
            
            # Only apply significant changes
            if abs(avg_ratio - 1.0) < 0.02:
                return audio
            
            # For very gentle transitions, use smaller window sizes
            if len(audio) > 2048:
                # Use overlap-add with smaller windows for smoother results
                window_size = min(2048, len(audio) // 16)  # Smaller windows
                hop_size = window_size // 4  # More overlap (75%)
                
                result = np.zeros(len(audio))
                window_func = np.hanning(window_size)
                normalization = np.zeros(len(audio))
                
                for i in range(0, len(audio) - window_size + 1, hop_size):
                    # Get current window
                    window_audio = audio[i:i + window_size] * window_func
                    
                    # Calculate local tempo ratio
                    ratio_start_idx = int((i / len(audio)) * len(tempo_ratios))
                    ratio_end_idx = int(((i + window_size) / len(audio)) * len(tempo_ratios))
                    ratio_end_idx = min(ratio_end_idx, len(tempo_ratios) - 1)
                    
                    if ratio_start_idx < len(tempo_ratios):
                        local_ratio = np.mean(tempo_ratios[ratio_start_idx:ratio_end_idx + 1])
                        
                        # Apply very gentle tempo stretching
                        if abs(local_ratio - 1.0) > 0.02:
                            # Use librosa's highest quality mode
                            stretched_window = librosa.effects.time_stretch(
                                window_audio, rate=1/local_ratio
                            )
                            
                            # High-quality resampling back to original size
                            if len(stretched_window) != window_size:
                                stretched_window = signal.resample(stretched_window, window_size)
                            
                            # Apply window function again and ensure same length
                            stretched_window = stretched_window[:window_size] * window_func
                        else:
                            stretched_window = window_audio
                        
                        # Overlap-add into result
                        end_idx = min(i + window_size, len(result))
                        actual_size = end_idx - i
                        result[i:end_idx] += stretched_window[:actual_size]
                        normalization[i:end_idx] += window_func[:actual_size]
                
                # Normalize overlapped regions
                nonzero_mask = normalization > 0.01
                result[nonzero_mask] /= normalization[nonzero_mask]
                
                return result
                
            else:
                # For shorter audio, use simple high-quality stretching
                if abs(avg_ratio - 1.0) > 0.02:
                    return librosa.effects.time_stretch(audio, rate=1/avg_ratio)
                else:
                    return audio
                    
        except Exception as e:
            logger.warning(f"    Gentle tempo curve failed: {e}, using original audio")
            return audio
    
    def _apply_tempo_curve(self, audio: np.ndarray, original_tempo: float, 
                          tempo_curve: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply a smooth tempo curve to audio using phase accumulation
        """
        try:
            # Calculate the cumulative tempo ratios
            tempo_ratios = tempo_curve / original_tempo
            
            # Use librosa's phase vocoder for smooth tempo changes
            # This preserves pitch while changing tempo smoothly
            
            # For very smooth transitions, we'll use a windowed approach
            if len(tempo_ratios) > 1024:  # For longer crossfades
                # Apply tempo stretching in overlapping windows for ultra-smooth results
                window_size = len(audio) // 8  # 8 overlapping windows
                hop_size = window_size // 2
                
                result = np.zeros_like(audio)
                window_func = np.hanning(window_size)
                
                for i in range(0, len(audio) - window_size + 1, hop_size):
                    # Get current window
                    window_audio = audio[i:i + window_size] * window_func
                    
                    # Calculate average tempo ratio for this window
                    ratio_start_idx = int((i / len(audio)) * len(tempo_ratios))
                    ratio_end_idx = int(((i + window_size) / len(audio)) * len(tempo_ratios))
                    ratio_end_idx = min(ratio_end_idx, len(tempo_ratios) - 1)
                    
                    if ratio_start_idx < len(tempo_ratios):
                        avg_ratio = np.mean(tempo_ratios[ratio_start_idx:ratio_end_idx + 1])
                        
                        # Apply tempo stretching to window
                        if abs(avg_ratio - 1.0) > 0.01:
                            stretched_window = librosa.effects.time_stretch(window_audio, rate=1/avg_ratio)
                            
                            # Resize back to original window size with quality resampling
                            if len(stretched_window) != window_size:
                                stretched_window = signal.resample(stretched_window, window_size)
                            
                            # Apply window function again after stretching
                            stretched_window *= window_func
                        else:
                            stretched_window = window_audio
                        
                        # Overlap-add into result
                        end_idx = min(i + window_size, len(result))
                        actual_size = end_idx - i
                        result[i:end_idx] += stretched_window[:actual_size]
                    
                return result
                
            else:
                # For shorter crossfades, use simpler approach
                avg_ratio = np.mean(tempo_ratios)
                if abs(avg_ratio - 1.0) > 0.01:
                    return librosa.effects.time_stretch(audio, rate=1/avg_ratio)
                else:
                    return audio
                    
        except Exception as e:
            logger.warning(f"    Tempo curve application failed: {e}, using original audio")
            return audio
    
    def _apply_invisible_tempo_sync(self, audio: np.ndarray, original_tempo: float, 
                                  target_tempo: float, is_outro: bool = True) -> np.ndarray:
        """
        Apply nearly invisible tempo synchronization with gradual pitch preservation
        """
        tempo_diff = abs(original_tempo - target_tempo)
        
        # Only apply very gentle adjustments for invisible sync
        if tempo_diff < 0.5:
            return audio
        
        # Use much smaller tempo adjustments to avoid pitch artifacts
        max_tempo_change = min(0.02, tempo_diff / original_tempo * 0.15)  # Max 2% change
        
        # Calculate gradual stretch factor
        if is_outro:
            # For outro: move very gradually toward target tempo
            if target_tempo > original_tempo:
                stretch_factor = 1.0 - max_tempo_change * 0.5  # Even gentler
            else:
                stretch_factor = 1.0 + max_tempo_change * 0.5
        else:
            # For intro: start much closer to original tempo
            if original_tempo > target_tempo:
                stretch_factor = 1.0 - max_tempo_change * 0.3
            else:
                stretch_factor = 1.0 + max_tempo_change * 0.3
        
        try:
            # Apply very gentle time stretching with enhanced quality
            # Use phase vocoder for better pitch preservation
            if abs(stretch_factor - 1.0) > 0.005:  # Only if meaningful change
                # Apply stretching in smaller chunks for smoother results
                chunk_size = len(audio) // 4  # Process in quarters
                adjusted_chunks = []
                
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    if len(chunk) > 1024:  # Only process significant chunks
                        # Apply gradual stretch factor that varies across the chunk
                        position_factor = i / len(audio)
                        local_stretch = 1.0 + (stretch_factor - 1.0) * position_factor
                        
                        adjusted_chunk = librosa.effects.time_stretch(chunk, rate=local_stretch)
                        
                        # Resample back to exact chunk size to maintain timing
                        if len(adjusted_chunk) != len(chunk):
                            adjusted_chunk = signal.resample(adjusted_chunk, len(chunk))
                        
                        adjusted_chunks.append(adjusted_chunk)
                    else:
                        adjusted_chunks.append(chunk)
                
                adjusted_audio = np.concatenate(adjusted_chunks)
                
                # Ensure exact length match
                if len(adjusted_audio) != len(audio):
                    adjusted_audio = signal.resample(adjusted_audio, len(audio))
                    
                logger.info(f"    Applied gradual invisible tempo sync: max {stretch_factor:.4f}x stretch")
                return adjusted_audio
            else:
                return audio
            
        except Exception as e:
            logger.warning(f"    Invisible tempo sync failed: {e}, using original")
            return audio
    
    def adjust_tempo(self, audio: np.ndarray, source_tempo: float, target_tempo: float) -> np.ndarray:
        """
        Adjust tempo with enhanced BPM synchronization like Apple Music
        """
        if abs(source_tempo - target_tempo) < 1:  # Very small difference, don't adjust
            return audio
        
        # Calculate stretch factor
        stretch_factor = source_tempo / target_tempo
        
        # Use librosa's high-quality time stretching (preserves pitch)
        adjusted_audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
        return adjusted_audio
    
    def align_beats(self, track1: Dict, track2: Dict, crossfade_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align beats between tracks for seamless transitions (without changing overall tempo)
        """
        audio1 = track1['audio_data']
        audio2 = track2['audio_data']
        beats1 = track1['beats']
        beats2 = track2['beats']
        sr = track1['sample_rate']
        
        # Find the best alignment point near the crossfade without tempo changes
        crossfade_time = crossfade_samples / sr
        track1_duration = len(audio1) / sr
        
        # Find beats near the transition point for better timing
        transition_point = track1_duration - crossfade_time
        
        # Find closest beats to the transition points
        if len(beats1) > 0 and len(beats2) > 0:
            beat1_idx = np.argmin(np.abs(beats1 - transition_point))
            beat2_idx = np.argmin(np.abs(beats2 - crossfade_time))
            
            if beat1_idx < len(beats1) and beat2_idx < len(beats2):
                # Calculate small timing adjustments (not tempo changes)
                beat1_time = beats1[beat1_idx]
                beat2_time = beats2[beat2_idx]
                
                # Minor adjustment to track1 ending to align with beat (max 0.5 seconds)
                target_adjustment = min(0.5, abs(beat1_time - transition_point))
                if beat1_time > transition_point:
                    target_end_time = track1_duration - target_adjustment
                else:
                    target_end_time = track1_duration + target_adjustment
                
                target_end_samples = int(target_end_time * sr)
                if 0 < target_end_samples <= len(audio1):
                    audio1 = audio1[:target_end_samples]
                
                # Minor adjustment to track2 start to align with beat (max 0.5 seconds)
                beat2_adjustment = min(0.5, beat2_time)
                beat2_samples = int(beat2_adjustment * sr)
                if beat2_samples < len(audio2):
                    audio2 = audio2[beat2_samples:]
        
        return audio1, audio2
    
    def smart_track_ordering(self, analyzed_tracks: List[Dict]) -> List[Dict]:
        """
        Reorder tracks for optimal transitions using a greedy approach
        """
        if len(analyzed_tracks) <= 1:
            return analyzed_tracks
        
        ordered_tracks = [analyzed_tracks[0]]  # Start with first track
        remaining_tracks = analyzed_tracks[1:].copy()
        
        while remaining_tracks:
            current_track = ordered_tracks[-1]
            best_track = None
            best_score = -1
            
            # Find the most compatible next track
            for track in remaining_tracks:
                compatibility = self.calculate_compatibility(current_track, track)
                if compatibility > best_score:
                    best_score = compatibility
                    best_track = track
            
            if best_track:
                ordered_tracks.append(best_track)
                remaining_tracks.remove(best_track)
                logger.info(f"Next track: {best_track['file_path'].name} (compatibility: {best_score:.2f})")
            else:
                # If no good match, take the first remaining
                ordered_tracks.append(remaining_tracks.pop(0))
        
        return ordered_tracks
    
    def create_mix(self) -> bool:
        """
        Create the automix from all tracks in the folder
        """
        try:
            # Get audio files
            audio_files = self.get_audio_files()
            if not audio_files:
                logger.error("No audio files found in the input folder")
                return False
            
            # Analyze all tracks
            logger.info("Analyzing tracks...")
            analyzed_tracks = []
            for file_path in audio_files:
                analysis = self.analyze_audio(file_path)
                if analysis:
                    analyzed_tracks.append(analysis)
            
            if len(analyzed_tracks) < 2:
                logger.error("Need at least 2 tracks to create a mix")
                return False
            
            # Smart ordering for better transitions
            logger.info("Optimizing track order...")
            ordered_tracks = self.smart_track_ordering(analyzed_tracks)
            
            # Create the mix with enhanced Apple Music-style transitions (keeping original BPMs)
            logger.info("Creating Apple Music-style mix with original BPMs and transition-only tempo sync...")
            crossfade_samples = int(self.crossfade_duration * self.sample_rate)
            
            # Start with the first track (original BPM)
            mixed_audio = ordered_tracks[0]['audio_data'].copy()
            
            # Add each subsequent track with enhanced crossfade
            for i in range(1, len(ordered_tracks)):
                current_track = ordered_tracks[i-1]
                next_track = ordered_tracks[i]
                
                logger.info(f"Mixing: {next_track['file_path'].name}")
                compatibility = self.calculate_compatibility(current_track, next_track)
                logger.info(f"  Compatibility score: {compatibility:.3f}")
                logger.info(f"  Current tempo: {current_track['tempo']:.1f} BPM (keeping original)")
                logger.info(f"  Next tempo: {next_track['tempo']:.1f} BPM (keeping original)")
                
                # Ensure smooth flow without cuts
                logger.info(f"  Analyzing flow points for seamless transition...")
                current_audio_for_mix = mixed_audio
                next_audio = next_track['audio_data'].copy()
                
                # Create flow-optimized versions
                if i == 1:  # First transition, use full current track
                    flow_current = current_audio_for_mix
                else:
                    # For subsequent tracks, we already have the flowing mix
                    flow_current = current_audio_for_mix
                
                # Optimize next track for natural flow entry
                flow_current, flow_next = self._ensure_smooth_flow(
                    {'audio_data': flow_current, 'sample_rate': self.sample_rate, 
                     'outro_start': current_track.get('outro_start', len(flow_current) / self.sample_rate * 0.85)},
                    {'audio_data': next_audio, 'sample_rate': self.sample_rate,
                     'intro_end': next_track.get('intro_end', 0)}
                )
                
                # Apply beat alignment for better transitions (without changing overall tempo)
                logger.info(f"  Aligning beats for seamless transition...")
                aligned_current, aligned_next = self.align_beats(
                    {'audio_data': flow_current, 'beats': current_track['beats'], 'sample_rate': self.sample_rate},
                    {'audio_data': flow_next, 'beats': next_track['beats'], 'sample_rate': self.sample_rate},
                    crossfade_samples
                )
                
                # Create enhanced crossfade with volume preservation
                mixed_audio = self.create_crossfade(
                    aligned_current, aligned_next, crossfade_samples,
                    current_track['tempo'], next_track['tempo']
                )
            
            # Normalize final mix to consistent level without changing relative volumes
            logger.info("Applying final volume normalization...")
            final_rms = np.sqrt(np.mean(mixed_audio**2))
            target_rms = 0.2  # Conservative target level
            
            if final_rms > 0:
                final_gain = target_rms / final_rms
                # Limit gain to prevent extreme changes
                if final_gain > 2.0:
                    final_gain = 2.0
                elif final_gain < 0.5:
                    final_gain = 0.5
                mixed_audio = mixed_audio * final_gain
            
            # Final gentle limiting to prevent any clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 0.95:
                mixed_audio = mixed_audio * (0.95 / max_val)
                logger.info(f"Applied gentle limiting: {max_val:.3f} → 0.95")
            
            # Save the result
            output_path = self.input_folder / self.output_file
            sf.write(str(output_path), mixed_audio, self.sample_rate)
            
            total_duration = len(mixed_audio) / self.sample_rate
            logger.info(f"Mix created successfully!")
            logger.info(f"Output: {output_path}")
            logger.info(f"Duration: {total_duration:.1f} seconds")
            logger.info(f"Tracks mixed: {len(ordered_tracks)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating mix: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Create Apple Music-style automix from folder of tracks")
    parser.add_argument("input_folder", help="Path to folder containing audio tracks")
    parser.add_argument("-o", "--output", default="automix_output.wav", 
                       help="Output filename (default: automix_output.wav)")
    parser.add_argument("-c", "--crossfade", type=float, default=8.0,
                       help="Crossfade duration in seconds (default: 8.0)")
    parser.add_argument("-s", "--sample-rate", type=int, default=44100,
                       help="Sample rate for processing (default: 44100)")
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        logger.error(f"Input folder does not exist: {args.input_folder}")
        return 1
    
    # Create automixer and run
    mixer = AutoMixer(
        input_folder=args.input_folder,
        output_file=args.output,
        crossfade_duration=args.crossfade,
        sample_rate=args.sample_rate
    )
    
    success = mixer.create_mix()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())