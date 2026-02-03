"""
Speaker Detection with Multi-Feature Fusion and Vector Database (ChromaDB)
===========================================================================
Uses ChromaDB for efficient similarity search and persistent storage, with
multi-feature fusion scoring (NN embeddings + FFT + MFCC) for higher accuracy.

Key features:
- Multi-feature fusion: combines neural network, FFT, and MFCC similarity scores
- Persistent storage: both embeddings (ChromaDB) and spectral features (disk)
- Audio quality gating: rejects low-SNR segments before identification
- Per-speaker temporal smoothing: avoids cross-speaker confidence bleed
- Enrollment quality checks: consistency validation and minimum sample count
- High-confidence centroid updates: prevents profile drift from misidentifications
"""

import numpy as np
import sounddevice as sd
import torch
from collections import deque
from datetime import datetime
import time
import warnings
import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import chromadb
from chromadb.config import Settings

warnings.filterwarnings('ignore')

# Windows fix: Monkey-patch SpeechBrain symlink issue
import speechbrain.utils.fetching as fetching_module

def patched_link_with_strategy(src, dst, local_strategy="copy"):
    src, dst = Path(src), Path(dst)
    if dst.exists(): return dst
    if not src.exists(): return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst

fetching_module.link_with_strategy = patched_link_with_strategy

from speechbrain.inference.speaker import EncoderClassifier


class VectorDBSpeakerDetector:
    """
    Speaker detection using ChromaDB vector database for efficient
    similarity search and persistent storage.
    """
    
    def __init__(
        self,
        sample_rate=16000,
        similarity_threshold=0.66,      # Lowered for better matching
        min_speech_duration=2.0,        # Larger samples for more stable detection
        vad_threshold=0.5,             # Slightly more sensitive
        device='cpu',
        db_path='./speaker_vectordb',
        enrollment_bonus=0.20,          # Higher bonus for enrolled speakers
        min_raw_similarity=0.20,        # Lower floor for matching
        confidence_smoothing=0.4,       # Temporal smoothing factor
        switch_confirmation_count=2,    # Require N consecutive detections before confirming switch
        use_fft=True,                   # Use FFT spectral features
        fft_weight=0.2,                 # Weight for FFT features in fusion scoring
        nn_weight=0.60,                 # Weight for neural network embedding score
        spectral_fft_weight=0.20,       # Weight for FFT spectral score in fusion
        spectral_mfcc_weight=0.20,      # Weight for MFCC spectral score in fusion
        min_audio_energy=0.005,         # Minimum RMS energy to accept audio segment
        min_snr_db=5.0,                 # Minimum signal-to-noise ratio in dB
        centroid_update_threshold=0.55, # Only update centroid if confidence above this
        min_enrollment_samples=3        # Minimum samples required for enrollment
    ):
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.enrollment_bonus = enrollment_bonus
        self.enrolled_threshold = similarity_threshold - enrollment_bonus
        self.min_speech_duration = min_speech_duration
        self.vad_threshold = vad_threshold
        self.device = device
        self.min_raw_similarity = min_raw_similarity
        self.confidence_smoothing = confidence_smoothing
        self.switch_confirmation_count = switch_confirmation_count
        self.use_fft = use_fft
        self.fft_weight = fft_weight

        # Multi-feature fusion weights (must sum to 1.0)
        self.nn_weight = nn_weight
        self.spectral_fft_weight = spectral_fft_weight
        self.spectral_mfcc_weight = spectral_mfcc_weight

        # Audio quality thresholds
        self.min_audio_energy = min_audio_energy
        self.min_snr_db = min_snr_db

        # Centroid update controls
        self.centroid_update_threshold = centroid_update_threshold

        # Enrollment quality
        self.min_enrollment_samples = min_enrollment_samples

        # Per-speaker temporal smoothing - track recent confidences per speaker
        self.per_speaker_confidences = {}  # {speaker_id: deque of recent confidences}
        self.max_recent = 5  # Number of recent detections to average per speaker

        # Speaker switch confirmation - prevents rapid flip-flopping
        self.pending_speaker = None      # Speaker waiting to be confirmed
        self.pending_count = 0           # How many times pending speaker detected
        self.confirmed_speaker = None    # Currently confirmed speaker
        
        # Initialize ChromaDB
        self.db_path = db_path
        print("Initializing ChromaDB vector database...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        # Create or get collection for speaker embeddings
        self.collection = self.chroma_client.get_or_create_collection(
            name="speaker_embeddings",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        print(f"[*] Loaded {self.collection.count()} embeddings from database")
        
        # Initialize VAD
        print("Initializing Voice Activity Detection (Silero VAD)...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.vad_model.to(device)
        
        # Initialize speaker embedding model
        print("Loading pretrained speaker embedding model (ECAPA-TDNN)...")
        savedir = Path("pretrained_models/spkrec-ecapa-voxceleb")
        if (savedir / "hyperparams.yaml").exists():
            self.embedding_model = EncoderClassifier.from_hparams(
                source=str(savedir),
                savedir=str(savedir),
                run_opts={"device": device}
            )
        else:
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(savedir),
                run_opts={"device": device}
            )
        print("Model loaded successfully!")
        
        # Runtime state
        self.speaker_count = self._get_max_speaker_id()
        self.last_speaker_id = None
        self.speech_buffer = deque(maxlen=3000)  # Larger buffer for more context
        self.is_speaking = False
        self._vad_buffer = np.array([], dtype=np.float32)
        
        # Recent detections for switch confirmation
        self.recent_detections = deque(maxlen=5)

        # Enrollment
        self.enrollment_mode = False
        self.enrollment_target = None
        self.enrollment_embeddings = []
        self.enrollment_fft_features = []  # Store FFT features during enrollment
        self.enrollment_mfcc_features = []  # Store MFCC features during enrollment
        self.enrollment_min_duration = 5.0  # Longer samples for enrollment (5 seconds)

        # Spectral features storage - persisted to disk alongside ChromaDB
        self.spectral_features_path = os.path.join(db_path, 'spectral_features.json')
        self.speaker_fft_features = {}   # {speaker_id: fft_centroid}
        self.speaker_mfcc_features = {}  # {speaker_id: mfcc_centroid}
        self._load_spectral_features()

        self.last_enrollment_time = 0       # Timestamp of last enrollment sample
    
    def _get_max_speaker_id(self) -> int:
        """Get the highest Speaker_N id from database."""
        if self.collection.count() == 0:
            return 0
        
        results = self.collection.get()
        max_id = 0
        for metadata in results['metadatas']:
            speaker_id = metadata.get('speaker_id', '')
            if speaker_id.startswith('Speaker_'):
                try:
                    num = int(speaker_id.split('_')[1])
                    max_id = max(max_id, num)
                except:
                    pass
        return max_id
    
    def _load_spectral_features(self):
        """Load persisted FFT/MFCC features from disk."""
        if os.path.exists(self.spectral_features_path):
            try:
                with open(self.spectral_features_path, 'r') as f:
                    data = json.load(f)
                for sid, feat in data.get('fft', {}).items():
                    self.speaker_fft_features[sid] = np.array(feat, dtype=np.float32)
                for sid, feat in data.get('mfcc', {}).items():
                    self.speaker_mfcc_features[sid] = np.array(feat, dtype=np.float32)
                print(f"[*] Loaded spectral features for {len(self.speaker_fft_features)} speakers")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[!] Could not load spectral features: {e}")

    def _save_spectral_features(self):
        """Persist FFT/MFCC features to disk."""
        data = {
            'fft': {sid: feat.tolist() for sid, feat in self.speaker_fft_features.items()},
            'mfcc': {sid: feat.tolist() for sid, feat in self.speaker_mfcc_features.items()}
        }
        os.makedirs(os.path.dirname(self.spectral_features_path), exist_ok=True)
        with open(self.spectral_features_path, 'w') as f:
            json.dump(data, f)

    def check_audio_quality(self, audio_data: np.ndarray) -> Tuple[bool, float, float]:
        """
        Check if audio segment has sufficient quality for reliable embedding extraction.
        Returns (is_acceptable, rms_energy, snr_db).
        """
        audio = audio_data.astype(np.float32)

        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))

        # Estimate SNR: compare top 90th percentile energy to bottom 10th percentile
        frame_size = 512
        num_frames = len(audio) // frame_size
        if num_frames < 2:
            return rms >= self.min_audio_energy, rms, 0.0

        frame_energies = np.array([
            np.sqrt(np.mean(audio[i * frame_size:(i + 1) * frame_size] ** 2))
            for i in range(num_frames)
        ])
        signal_energy = np.percentile(frame_energies, 90)
        noise_energy = np.percentile(frame_energies, 10) + 1e-10
        snr_db = 20 * np.log10(signal_energy / noise_energy)

        is_acceptable = rms >= self.min_audio_energy and snr_db >= self.min_snr_db
        return is_acceptable, rms, snr_db

    def _reset_session_state(self):
        """Reset temporal state between listening sessions to prevent stale data leakage."""
        self.per_speaker_confidences.clear()
        self.pending_speaker = None
        self.pending_count = 0
        self.confirmed_speaker = None
        self.recent_detections.clear()
        self.last_speaker_id = None

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for better embedding quality.
        - Remove DC offset
        - Normalize amplitude
        - Apply pre-emphasis filter (boosts high frequencies)
        - Apply noise gate
        """
        audio = audio_data.copy().astype(np.float32)
        
        # 1. Remove DC offset
        audio = audio - np.mean(audio)
        
        # 2. Noise gate - remove very quiet samples (likely noise)
        noise_threshold = 0.01
        noise_floor = np.percentile(np.abs(audio), 10)
        if noise_floor < noise_threshold:
            # Apply soft noise gate
            audio = np.where(np.abs(audio) < noise_threshold, audio * 0.1, audio)
        
        # 3. Pre-emphasis filter (y[n] = x[n] - alpha * x[n-1])
        # This boosts high frequencies which are important for speaker identity
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # 4. Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # 5. Apply slight compression to reduce dynamic range differences
        # This helps with varying microphone distances
        audio = np.sign(audio) * np.log1p(np.abs(audio) * 10) / np.log1p(10)
        
        return audio
    
    def extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract normalized speaker embedding with preprocessing."""
        # Preprocess audio first
        processed_audio = self.preprocess_audio(audio_data)
        
        audio_tensor = torch.FloatTensor(processed_audio).unsqueeze(0)
        with torch.no_grad():
            embedding = self.embedding_model.encode_batch(audio_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        return embedding / np.linalg.norm(embedding)
    
    def extract_fft_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract FFT-based spectral features for additional speaker characterization.
        
        Features extracted:
        - Spectral centroid (brightness)
        - Spectral bandwidth (spread)
        - Spectral flatness (noisiness)
        - Spectral rolloff (high-frequency energy)
        - Zero-crossing rate
        - Energy distribution across frequency bands
        - Short-term spectral statistics (mean, std, skew)
        
        Returns normalized feature vector of size 20.
        """
        audio = self.preprocess_audio(audio_data)
        
        # Parameters
        n_fft = 2048
        hop_length = 512
        
        features = []
        spectral_features_all = []
        
        # Process in frames
        for i in range(0, len(audio) - n_fft, hop_length):
            frame = audio[i:i + n_fft]
            
            # Apply Hann window
            window = np.hanning(n_fft)
            windowed = frame * window
            
            # FFT
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs = np.fft.rfftfreq(n_fft, 1/self.sample_rate)
            
            # Avoid division by zero
            spectrum_sum = np.sum(spectrum) + 1e-10
            
            # Spectral centroid - center of mass of spectrum
            centroid = np.sum(freqs * spectrum) / spectrum_sum
            
            # Spectral bandwidth - weighted std dev
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spectrum_sum)
            
            # Spectral flatness - geometric mean / arithmetic mean
            geo_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
            arith_mean = np.mean(spectrum) + 1e-10
            flatness = geo_mean / arith_mean
            
            # Spectral rolloff - frequency below which 85% of energy is contained
            cumsum = np.cumsum(spectrum)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
            
            spectral_features_all.append([centroid, bandwidth, flatness, rolloff])
        
        if len(spectral_features_all) == 0:
            return np.zeros(20)
        
        spectral_features_all = np.array(spectral_features_all)
        
        # Aggregate statistics across frames
        for i in range(4):
            features.extend([
                np.mean(spectral_features_all[:, i]),
                np.std(spectral_features_all[:, i]),
                np.percentile(spectral_features_all[:, i], 25),
                np.percentile(spectral_features_all[:, i], 75)
            ])
        
        # Zero-crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        features.append(zcr)
        
        # Energy distribution in bands (low, mid, high)
        full_spectrum = np.abs(np.fft.rfft(audio))
        full_freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        total_energy = np.sum(full_spectrum ** 2) + 1e-10
        
        low_mask = full_freqs < 500  # Low frequencies
        mid_mask = (full_freqs >= 500) & (full_freqs < 2000)  # Mid frequencies
        high_mask = full_freqs >= 2000  # High frequencies
        
        features.append(np.sum(full_spectrum[low_mask] ** 2) / total_energy)
        features.append(np.sum(full_spectrum[mid_mask] ** 2) / total_energy)
        features.append(np.sum(full_spectrum[high_mask] ** 2) / total_energy)
        
        features = np.array(features[:20])  # Cap at 20 features
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def extract_mfcc_features(self, audio_data: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features.
        MFCCs capture vocal tract characteristics unique to each speaker.
        
        Args:
            audio_data: Raw audio samples
            n_mfcc: Number of MFCC coefficients to extract (default 13)
            
        Returns:
            Normalized MFCC feature vector (39 dimensions: 13 MFCCs + 13 delta + 13 delta-delta stats)
        """
        audio = audio_data.flatten().astype(np.float32)
        
        # Parameters
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        n_fft = 512
        n_mels = 40
        
        # Pre-emphasis filter (enhances high frequencies)
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Frame the signal
        num_frames = 1 + (len(audio) - frame_length) // hop_length
        if num_frames < 1:
            return np.zeros(n_mfcc * 3)  # Return zeros if audio too short
        
        frames = np.zeros((num_frames, frame_length))
        for i in range(num_frames):
            start = i * hop_length
            frames[i] = audio[start:start + frame_length]
        
        # Apply Hamming window
        hamming = np.hamming(frame_length)
        frames *= hamming
        
        # Compute power spectrum
        power_spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
        
        # Create Mel filterbank
        mel_low = 0
        mel_high = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filterbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            for j in range(bin_points[i], bin_points[i + 1]):
                if j < filterbank.shape[1]:
                    filterbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i] + 1e-10)
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                if j < filterbank.shape[1]:
                    filterbank[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1] + 1e-10)
        
        # Apply filterbank to power spectrum
        mel_spectrum = np.dot(power_spectrum, filterbank.T)
        mel_spectrum = np.where(mel_spectrum == 0, 1e-10, mel_spectrum)
        log_mel_spectrum = np.log(mel_spectrum)
        
        # DCT to get MFCCs (Type-II DCT)
        n_coeff = log_mel_spectrum.shape[1]
        dct_matrix = np.zeros((n_mfcc, n_coeff))
        for k in range(n_mfcc):
            for n in range(n_coeff):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_coeff))
        dct_matrix *= np.sqrt(2 / n_coeff)
        dct_matrix[0] *= np.sqrt(0.5)
        
        mfccs = np.dot(log_mel_spectrum, dct_matrix.T)  # Shape: (num_frames, n_mfcc)
        
        # Compute delta (first derivative) and delta-delta (second derivative)
        def compute_delta(feat, N=2):
            delta = np.zeros_like(feat)
            for t in range(feat.shape[0]):
                numerator = sum(n * (feat[min(t + n, feat.shape[0] - 1)] - feat[max(t - n, 0)]) for n in range(1, N + 1))
                denominator = 2 * sum(n ** 2 for n in range(1, N + 1))
                delta[t] = numerator / (denominator + 1e-10)
            return delta
        
        delta_mfccs = compute_delta(mfccs)
        delta_delta_mfccs = compute_delta(delta_mfccs)
        
        # Aggregate statistics across frames
        features = []
        for mfcc_set in [mfccs, delta_mfccs, delta_delta_mfccs]:
            features.extend([
                np.mean(mfcc_set, axis=0),  # Mean across frames
            ])
        
        features = np.concatenate(features)  # 13 * 3 = 39 features
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def extract_combined_embedding(self, audio_data: np.ndarray, fft_weight: float = 0.3) -> np.ndarray:
        """
        Extract combined embedding using both neural network and FFT features.
        
        Args:
            audio_data: Raw audio samples
            fft_weight: Weight for FFT features (0-1). Neural weight = 1 - fft_weight
            
        Returns:
            Combined normalized embedding vector
        """
        # Get neural network embedding (192 dim for ECAPA-TDNN)
        nn_embedding = self.extract_embedding(audio_data)
        
        # Get FFT features (20 dim)
        fft_features = self.extract_fft_features(audio_data)
        
        # Weight and concatenate
        # Scale FFT features to match typical nn_embedding magnitude
        fft_scaled = fft_features * fft_weight
        nn_scaled = nn_embedding * (1 - fft_weight)
        
        # Concatenate
        combined = np.concatenate([nn_scaled, fft_scaled])
        
        # Normalize combined embedding
        combined = combined / np.linalg.norm(combined)
        
        return combined
    
    def get_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Get neural network embedding for database storage.
        Always returns 192-dim embedding (FFT is used separately for scoring).
        """
        # Always use pure NN embedding for database storage
        # FFT features are used as a separate scoring boost in identify_speaker
        return self.extract_embedding(audio_data)
    
    def add_speaker_embedding(self, speaker_id: str, embedding: np.ndarray, is_enrolled: bool = False):
        """Add a speaker embedding to the vector database."""
        embedding_id = f"{speaker_id}_{datetime.now().timestamp()}"
        
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[{
                "speaker_id": speaker_id,
                "is_enrolled": is_enrolled,
                "timestamp": datetime.now().isoformat(),
                "detection_count": 1
            }],
            ids=[embedding_id]
        )
    
    def find_similar_speaker(self, embedding: np.ndarray, n_results: int = 5) -> Tuple[Optional[str], float, bool]:
        """
        Find the most similar speaker using vector similarity search.
        Returns (speaker_id, similarity, is_enrolled)
        """
        if self.collection.count() == 0:
            return None, 0.0, False
        
        # Query the vector database
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(n_results, self.collection.count())
        )
        
        if not results['ids'][0]:
            return None, 0.0, False
        
        # ChromaDB returns distances, convert to similarity
        # For cosine distance: similarity = 1 - distance
        distance = results['distances'][0][0]
        similarity = 1.0 - distance
        
        metadata = results['metadatas'][0][0]
        speaker_id = metadata['speaker_id']
        is_enrolled = metadata.get('is_enrolled', False)
        
        return speaker_id, similarity, is_enrolled
    
    def find_all_speaker_similarities(self, embedding: np.ndarray) -> list:
        """
        Find similarities with ALL speakers in database.
        Returns list of (speaker_id, similarity, is_enrolled) sorted by similarity.
        """
        if self.collection.count() == 0:
            return []
        
        # Query for all speakers
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=self.collection.count()
        )
        
        if not results['ids'][0]:
            return []
        
        # Collect all similarities
        speakers_seen = {}
        for i, (dist, meta) in enumerate(zip(results['distances'][0], results['metadatas'][0])):
            speaker_id = meta['speaker_id']
            similarity = 1.0 - dist
            is_enrolled = meta.get('is_enrolled', False)
            
            # Keep highest similarity per speaker
            if speaker_id not in speakers_seen or similarity > speakers_seen[speaker_id][0]:
                speakers_seen[speaker_id] = (similarity, is_enrolled)
        
        # Sort by similarity descending
        result = [(sid, sim, enrolled) for sid, (sim, enrolled) in speakers_seen.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def update_speaker_centroid(self, speaker_id: str, new_embedding: np.ndarray):
        """
        Update speaker's centroid embedding using running average.
        This improves the speaker profile over time.
        """
        # Get all embeddings for this speaker
        results = self.collection.get(
            where={"speaker_id": speaker_id}
        )
        
        if not results['embeddings']:
            return
        
        # Calculate new centroid
        all_embeddings = [np.array(e) for e in results['embeddings']]
        all_embeddings.append(new_embedding)
        
        # Keep only last 20 embeddings for efficiency
        if len(all_embeddings) > 20:
            all_embeddings = all_embeddings[-20:]
        
        centroid = np.mean(all_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        # Update the main embedding (first one) with new centroid
        main_id = results['ids'][0]
        main_metadata = results['metadatas'][0]
        main_metadata['detection_count'] = main_metadata.get('detection_count', 0) + 1
        main_metadata['last_seen'] = datetime.now().isoformat()
        
        # Update in database
        self.collection.update(
            ids=[main_id],
            embeddings=[centroid.tolist()],
            metadatas=[main_metadata]
        )
    
    def identify_speaker(self, embedding: np.ndarray, audio_data: np.ndarray = None) -> Tuple[str, float]:
        """
        Identify speaker using multi-feature fusion scoring.

        Combines neural network embedding similarity with FFT and MFCC spectral
        similarity for higher accuracy. Uses per-speaker temporal smoothing to
        avoid cross-speaker confidence bleed.

        Args:
            embedding: Neural network embedding vector
            audio_data: Raw audio data (used for FFT/MFCC scoring when available)

        Returns:
            (speaker_id, fused_confidence) or ("Unknown", raw_nn_similarity)
        """
        # Single DB query - reuse for both top match and distinctiveness
        all_matches = self.find_all_speaker_similarities(embedding)

        if not all_matches:
            return "Unknown", 0.0

        speaker_id = all_matches[0][0]
        similarity = all_matches[0][1]
        is_enrolled = all_matches[0][2]

        # ===== STRICT UNKNOWN DETECTION =====
        min_absolute_similarity = 0.40  # Hard floor
        baseline_similarity = 0.25      # Expected random similarity
        required_gap = 0.10             # Required gap above baseline

        if similarity < min_absolute_similarity:
            return "Unknown", similarity

        if similarity < (baseline_similarity + required_gap):
            return "Unknown", similarity

        # ===== DISTINCTIVENESS CHECK =====
        if len(all_matches) >= 2:
            top_sim = all_matches[0][1]
            second_sim = all_matches[1][1]
            distinctiveness_gap = 0.05
            high_confidence_threshold = 0.50

            if top_sim < high_confidence_threshold and (top_sim - second_sim) < distinctiveness_gap:
                return "Unknown", similarity

        # ===== MULTI-FEATURE FUSION SCORING =====
        # Start with NN similarity as the base score
        nn_score = similarity

        # Compute spectral scores if audio_data provided and features available
        fft_score = 0.0
        mfcc_score = 0.0
        has_fft = self.use_fft and audio_data is not None and speaker_id in self.speaker_fft_features
        has_mfcc = self.use_fft and audio_data is not None and speaker_id in self.speaker_mfcc_features

        if has_fft:
            fft_score = self.compute_fft_similarity(audio_data, speaker_id)
        if has_mfcc:
            mfcc_score = self.compute_mfcc_similarity(audio_data, speaker_id)

        # Compute fused score with adaptive weighting
        if has_fft and has_mfcc:
            # Full fusion: NN + FFT + MFCC
            fused_score = (
                self.nn_weight * nn_score +
                self.spectral_fft_weight * fft_score +
                self.spectral_mfcc_weight * mfcc_score
            )
        elif has_fft:
            # Partial: NN + FFT only, redistribute MFCC weight
            fused_score = (
                (self.nn_weight + self.spectral_mfcc_weight * 0.5) * nn_score +
                (self.spectral_fft_weight + self.spectral_mfcc_weight * 0.5) * fft_score
            )
        elif has_mfcc:
            # Partial: NN + MFCC only, redistribute FFT weight
            fused_score = (
                (self.nn_weight + self.spectral_fft_weight * 0.5) * nn_score +
                (self.spectral_mfcc_weight + self.spectral_fft_weight * 0.5) * mfcc_score
            )
        else:
            # NN only
            fused_score = nn_score

        # Apply enrollment bonus
        effective_similarity = fused_score + (self.enrollment_bonus if is_enrolled else 0)
        threshold = self.enrolled_threshold if is_enrolled else self.similarity_threshold

        # ===== PER-SPEAKER TEMPORAL SMOOTHING =====
        # Only smooth with history from the SAME speaker to avoid cross-speaker bleed
        if speaker_id not in self.per_speaker_confidences:
            self.per_speaker_confidences[speaker_id] = deque(maxlen=self.max_recent)

        speaker_history = self.per_speaker_confidences[speaker_id]
        if speaker_history:
            smoothed_similarity = (
                effective_similarity * (1 - self.confidence_smoothing) +
                np.mean(speaker_history) * self.confidence_smoothing
            )
        else:
            smoothed_similarity = effective_similarity

        # Track this confidence for this speaker
        speaker_history.append(effective_similarity)

        # Must meet BOTH the threshold AND the min_raw_similarity
        if smoothed_similarity >= threshold and similarity >= self.min_raw_similarity:
            # Only update centroid when confidence is high enough to avoid drift
            if smoothed_similarity >= self.centroid_update_threshold:
                self.update_speaker_centroid(speaker_id, embedding)
            return speaker_id, smoothed_similarity
        else:
            return "Unknown", similarity
    
    def compute_fft_similarity(self, audio_data: np.ndarray, speaker_id: str) -> float:
        """
        Compute FFT feature similarity between audio and stored speaker features.
        Returns similarity score (0-1) or 0 if speaker has no FFT features stored.
        """
        if speaker_id not in self.speaker_fft_features:
            return 0.0
        
        # Extract FFT features from current audio
        current_fft = self.extract_fft_features(audio_data)
        stored_fft = self.speaker_fft_features[speaker_id]
        
        # Cosine similarity
        similarity = np.dot(current_fft, stored_fft)
        return max(0.0, similarity)  # Clamp to 0-1
    
    def compute_mfcc_similarity(self, audio_data: np.ndarray, speaker_id: str) -> float:
        """
        Compute MFCC feature similarity between audio and stored speaker features.
        Returns similarity score (0-1) or 0 if speaker has no MFCC features stored.
        """
        if speaker_id not in self.speaker_mfcc_features:
            return 0.0
        
        # Extract MFCC features from current audio
        current_mfcc = self.extract_mfcc_features(audio_data)
        stored_mfcc = self.speaker_mfcc_features[speaker_id]
        
        # Cosine similarity
        similarity = np.dot(current_mfcc, stored_mfcc)
        return max(0.0, similarity)  # Clamp to 0-1
    
    def start_enrollment(self, speaker_name: str):
        """Start enrollment mode."""
        print(f"\n{'='*70}")
        print(f"ENROLLMENT MODE: {speaker_name}")
        print(f"{'='*70}")
        print("Speak clearly for 10-15 seconds to enroll your voice.")
        
        self.enrollment_mode = True
        self.enrollment_target = speaker_name
        self.enrollment_embeddings = []
        self.enrollment_fft_features = []   # Reset FFT features collection
        self.enrollment_mfcc_features = []  # Reset MFCC features collection
    
    def complete_enrollment(self) -> bool:
        """Complete enrollment and save to vector database with quality checks."""
        if not self.enrollment_embeddings:
            print("[!] No embeddings collected. Enrollment failed.")
            self.enrollment_mode = False
            return False

        if len(self.enrollment_embeddings) < self.min_enrollment_samples:
            print(f"[!] Only {len(self.enrollment_embeddings)} samples collected, "
                  f"need at least {self.min_enrollment_samples}. Enrollment failed.")
            print("[!] Try speaking longer and more clearly.")
            self.enrollment_mode = False
            return False

        # Check embedding consistency - reject if samples are too spread out
        embeddings_array = np.array(self.enrollment_embeddings)
        centroid = np.mean(embeddings_array, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        similarities = np.dot(embeddings_array, centroid)
        avg_consistency = np.mean(similarities)

        if avg_consistency < 0.80:
            print(f"[!] Low enrollment consistency ({avg_consistency:.2f}). "
                  "Too much noise or multiple speakers detected.")
            print("[!] Try enrolling in a quieter environment with only one speaker.")
            self.enrollment_mode = False
            return False

        # Create centroid from collected embeddings
        mean_embedding = centroid

        # Add to database as enrolled speaker
        self.add_speaker_embedding(self.enrollment_target, mean_embedding, is_enrolled=True)

        # Save FFT features centroid if collected
        if self.enrollment_fft_features:
            fft_centroid = np.mean(self.enrollment_fft_features, axis=0)
            norm = np.linalg.norm(fft_centroid)
            if norm > 0:
                fft_centroid = fft_centroid / norm
            self.speaker_fft_features[self.enrollment_target] = fft_centroid
            print(f"    FFT features saved: {len(self.enrollment_fft_features)} samples")

        # Save MFCC features centroid if collected
        if self.enrollment_mfcc_features:
            mfcc_centroid = np.mean(self.enrollment_mfcc_features, axis=0)
            norm = np.linalg.norm(mfcc_centroid)
            if norm > 0:
                mfcc_centroid = mfcc_centroid / norm
            self.speaker_mfcc_features[self.enrollment_target] = mfcc_centroid
            print(f"    MFCC features saved: {len(self.enrollment_mfcc_features)} samples")

        # Persist spectral features to disk
        self._save_spectral_features()

        print(f"\n[+] Enrollment complete: {self.enrollment_target}")
        print(f"    Collected {len(self.enrollment_embeddings)} embeddings")
        print(f"    Consistency: {avg_consistency:.2f}")
        print(f"    Total embeddings in database: {self.collection.count()}")

        self.enrollment_mode = False
        self.enrollment_target = None
        self.enrollment_embeddings = []
        self.enrollment_fft_features = []
        self.enrollment_mfcc_features = []
        return True
    
    def _confirm_speaker_switch(self, detected_speaker: str, confidence: float) -> Tuple[str, float, bool]:
        """
        Confirm speaker switch using multiple consecutive detections.
        Returns (confirmed_speaker, confidence, is_switch_confirmed)
        
        This prevents rapid flip-flopping between speakers by requiring
        multiple consecutive detections of a new speaker before confirming.
        """
        # Track this detection (deque maxlen handles overflow automatically)
        self.recent_detections.append((detected_speaker, confidence))
        
        # First detection ever - confirm immediately
        if self.confirmed_speaker is None:
            self.confirmed_speaker = detected_speaker
            self.pending_speaker = None
            self.pending_count = 0
            return detected_speaker, confidence, False
        
        # Same as confirmed speaker - reset pending and confirm
        if detected_speaker == self.confirmed_speaker:
            self.pending_speaker = None
            self.pending_count = 0
            return detected_speaker, confidence, False
        
        # Different speaker detected
        if detected_speaker == self.pending_speaker:
            # Same pending speaker - increment count
            self.pending_count += 1
            
            # Check if we've reached confirmation threshold
            if self.pending_count >= self.switch_confirmation_count:
                # Confirm the switch!
                old_speaker = self.confirmed_speaker
                self.confirmed_speaker = detected_speaker
                self.pending_speaker = None
                self.pending_count = 0
                return detected_speaker, confidence, True
            else:
                # Not yet confirmed - return current confirmed speaker but show the detection
                # Return the detected speaker but mark as NOT a switch yet
                return detected_speaker, confidence, False
        else:
            # New pending speaker - start tracking
            self.pending_speaker = detected_speaker
            self.pending_count = 1
            
            # Return the detected speaker but NOT as a confirmed switch
            return detected_speaker, confidence, False
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[dict]:
        """
        Process audio through VAD and speaker detection.
        Uses speaker switch confirmation to prevent rapid flip-flopping.
        """
        self._vad_buffer = np.concatenate([self._vad_buffer, audio_chunk])
        
        result = None
        
        while len(self._vad_buffer) >= 512:
            vad_chunk = self._vad_buffer[:512]
            self._vad_buffer = self._vad_buffer[512:]
            
            audio_tensor = torch.FloatTensor(vad_chunk).to(self.device)
            
            with torch.no_grad():
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
            
            if speech_prob > self.vad_threshold:
                self.speech_buffer.append(vad_chunk)
                
                if not self.is_speaking:
                    self.is_speaking = True
                
                speech_duration = len(self.speech_buffer) * 512 / self.sample_rate
                if speech_duration >= self.min_speech_duration:
                    speech_audio = np.concatenate(list(self.speech_buffer))

                    try:
                        # Audio quality gate - reject low-quality segments
                        is_quality, rms, snr = self.check_audio_quality(speech_audio)
                        if not is_quality:
                            # Skip this segment silently - too noisy or too quiet
                            if not self.enrollment_mode:
                                self.speech_buffer.clear()
                            continue

                        embedding = self.get_embedding(speech_audio)

                        if self.enrollment_mode:
                            # Show enrollment progress
                            print(f"\r[Enrolling] Speech: {speech_duration:.1f}s / {self.enrollment_min_duration:.1f}s required | SNR: {snr:.1f}dB", end="", flush=True)

                            # Only collect if enough audio duration
                            if speech_duration >= self.enrollment_min_duration:
                                self.enrollment_embeddings.append(embedding)
                                # Always collect FFT and MFCC features during enrollment for persistence
                                fft_features = self.extract_fft_features(speech_audio)
                                self.enrollment_fft_features.append(fft_features)
                                mfcc_features = self.extract_mfcc_features(speech_audio)
                                self.enrollment_mfcc_features.append(mfcc_features)
                                result = {
                                    'mode': 'enrollment',
                                    'speaker_id': self.enrollment_target,
                                    'embeddings_collected': len(self.enrollment_embeddings),
                                    'timestamp': datetime.now()
                                }
                                print()  # New line after collecting
                                self.speech_buffer.clear()  # Clear buffer after collecting
                        else:
                            # Identify speaker with multi-feature fusion
                            raw_speaker_id, raw_confidence = self.identify_speaker(
                                embedding, audio_data=speech_audio
                            )

                            # Apply speaker switch confirmation
                            confirmed_speaker, confidence, is_confirmed_switch = self._confirm_speaker_switch(
                                raw_speaker_id, raw_confidence
                            )

                            # Only report as speaker_changed if it's a CONFIRMED switch
                            speaker_changed = is_confirmed_switch

                            # Use confirmed speaker for output (prevents showing rapid changes)
                            output_speaker = self.confirmed_speaker if self.confirmed_speaker else raw_speaker_id

                            # But if pending is building up, show the raw detection to indicate potential change
                            if self.pending_count > 0 and self.pending_speaker:
                                output_speaker = raw_speaker_id

                            # Get all speaker similarities for debug display
                            # (reuse embedding - no extra DB query needed since identify_speaker already queried)
                            all_similarities = self.find_all_speaker_similarities(embedding)

                            # Compute FFT and MFCC similarity for display
                            fft_similarity = 0.0
                            mfcc_similarity = 0.0
                            if self.use_fft and output_speaker != "Unknown":
                                fft_similarity = self.compute_fft_similarity(speech_audio, output_speaker)
                                mfcc_similarity = self.compute_mfcc_similarity(speech_audio, output_speaker)

                            result = {
                                'mode': 'detection',
                                'speaker_id': output_speaker,
                                'confidence': confidence,
                                'fft_similarity': fft_similarity,
                                'mfcc_similarity': mfcc_similarity,
                                'speaker_changed': speaker_changed,
                                'timestamp': datetime.now(),
                                'duration': len(speech_audio) / self.sample_rate,
                                'db_size': self.collection.count(),
                                'pending_switch': self.pending_speaker if self.pending_count > 0 else None,
                                'all_similarities': all_similarities
                            }

                            self.last_speaker_id = output_speaker

                            # Clear buffer after detection (not enrollment - that clears on success)
                            self.speech_buffer.clear()

                    except Exception as e:
                        print(f"[ERROR] {type(e).__name__}: {e}")
                        if not self.enrollment_mode:
                            self.speech_buffer.clear()
            else:
                if self.is_speaking:
                    self.is_speaking = False
                    # DON'T clear buffer during enrollment - allow pauses
                    if not self.enrollment_mode:
                        self.speech_buffer.clear()
        
        return result
    
    def start_listening(self, duration=None, enrollment_speaker=None):
        """Start real-time audio capture."""
        # Reset temporal state to prevent stale data from previous sessions
        self._reset_session_state()

        if enrollment_speaker:
            self.start_enrollment(enrollment_speaker)
        
        print("\n" + "="*70)
        print("SPEAKER DETECTION WITH VECTOR DATABASE")
        print("="*70)
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Similarity Threshold: {self.similarity_threshold}")
        print(f"Database Size: {self.collection.count()} embeddings")
        print("="*70)
        
        if self.enrollment_mode:
            print(f"\n[ENROLLMENT] Speak to enroll: {self.enrollment_target}")
            print("Press Ctrl+C after ~10 seconds to complete\n")
        else:
            print("\nListening... Press Ctrl+C to stop\n")
        
        def audio_callback(indata, frames, time_info, status):
            if status and "overflow" not in str(status).lower():
                print(f"[WARNING] {status}")
            
            result = self.process_audio_chunk(indata[:, 0])
            if result:
                self._print_detection(result)
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=4096,
                callback=audio_callback
            ):
                if duration:
                    time.sleep(duration)
                else:
                    while True:
                        time.sleep(0.1)
                        
        except KeyboardInterrupt:
            print("\n\nStopping...")
        
        finally:
            if self.enrollment_mode:
                self.complete_enrollment()
            self._print_summary()
    
    def _print_detection(self, result):
        """Print detection result with speaker switch confirmation status."""
        ts = result['timestamp'].strftime('%H:%M:%S')
        
        if result['mode'] == 'enrollment':
            print(f"[{ts}] Enrollment: Collected {result['embeddings_collected']} samples")
        else:
            speaker = result['speaker_id']
            conf = result['confidence'] * 100
            pending = result.get('pending_switch')
            all_sims = result.get('all_similarities', [])
            fft_sim = result.get('fft_similarity', 0) * 100
            mfcc_sim = result.get('mfcc_similarity', 0) * 100
            
            if result['speaker_changed']:
                print(f"\n{'>>> SPEAKER CHANGE CONFIRMED <<<':^60}")
            
            # Show detection with FFT and MFCC scores if available
            if fft_sim > 0 or mfcc_sim > 0:
                print(f"[{ts}] {speaker:12} | NN: {conf:5.1f}% | FFT: {fft_sim:5.1f}% | MFCC: {mfcc_sim:5.1f}%")
            else:
                print(f"[{ts}] {speaker:12} | Conf: {conf:5.1f}%")
            
            # Show all speaker similarities for debugging
            if all_sims:
                sims_str = " | ".join([f"{s[0]}:{s[1]*100:.0f}%" for s in all_sims[:5]])
                print(f"         Scores: {sims_str}")
    
    def _print_summary(self):
        """Print session summary with database stats."""
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"Total embeddings in database: {self.collection.count()}")
        
        # Get unique speakers
        results = self.collection.get()
        speakers = {}
        for metadata in results['metadatas']:
            sid = metadata['speaker_id']
            count = metadata.get('detection_count', 1)
            enrolled = metadata.get('is_enrolled', False)
            
            if sid not in speakers:
                speakers[sid] = {'count': count, 'enrolled': enrolled}
            else:
                speakers[sid]['count'] = max(speakers[sid]['count'], count)
        
        print(f"\nUnique speakers: {len(speakers)}")
        print("\nEnrolled speakers:")
        for sid, info in speakers.items():
            if info['enrolled']:
                print(f"  * {sid}: {info['count']} detections")
        
        print("\nAuto-detected speakers:")
        for sid, info in speakers.items():
            if not info['enrolled']:
                print(f"  - {sid}: {info['count']} detections")
        
        print("="*70)
    
    def record_and_identify(self, duration: float = 5.0):
        """
        Record audio for a fixed duration and identify the speaker.
        Uses multi-feature fusion for higher accuracy than continuous detection.
        """
        print(f"\n{'='*70}")
        print(f"RECORDING {duration} SECONDS OF AUDIO...")
        print(f"{'='*70}")
        print("Speak now!")

        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete

        print("\nRecording complete! Analyzing...")

        # Flatten to 1D
        audio_data = audio_data.flatten()

        # Check audio quality
        is_quality, rms, snr = self.check_audio_quality(audio_data)
        print(f"  Audio quality - RMS: {rms:.4f}, SNR: {snr:.1f}dB {'(OK)' if is_quality else '(LOW)'}")

        if not is_quality:
            print("[!] Audio quality too low for reliable identification.")
            print("    Try speaking louder or reducing background noise.")
            return

        # Extract embedding
        embedding = self.get_embedding(audio_data)

        # Get all speaker similarities
        all_matches = self.find_all_speaker_similarities(embedding)

        # Identify speaker with multi-feature fusion
        speaker_id, confidence = self.identify_speaker(embedding, audio_data=audio_data)

        # Display results
        print(f"\n{'='*70}")
        print("RESULT")
        print(f"{'='*70}")
        print(f"\n  Detected Speaker: {speaker_id}")
        print(f"  Fused Confidence: {confidence * 100:.1f}%")

        # Show spectral similarity breakdown
        if speaker_id != "Unknown" and self.use_fft:
            fft_sim = self.compute_fft_similarity(audio_data, speaker_id)
            mfcc_sim = self.compute_mfcc_similarity(audio_data, speaker_id)
            print(f"  NN Similarity:   {all_matches[0][1] * 100:.1f}%" if all_matches else "")
            print(f"  FFT Similarity:  {fft_sim * 100:.1f}%")
            print(f"  MFCC Similarity: {mfcc_sim * 100:.1f}%")

        print(f"\n  All NN Similarity Scores:")
        for i, (sid, sim, enrolled) in enumerate(all_matches):
            enrolled_tag = " [enrolled]" if enrolled else ""
            marker = " <-- BEST MATCH" if i == 0 else ""
            print(f"    {i+1}. {sid}: {sim*100:.1f}%{enrolled_tag}{marker}")

        print(f"\n{'='*70}")

    def reset_database(self):
        """Delete all speaker data from the database and persisted spectral features."""
        self.chroma_client.delete_collection("speaker_embeddings")
        self.collection = self.chroma_client.create_collection(
            name="speaker_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.speaker_count = 0
        self.speaker_fft_features.clear()
        self.speaker_mfcc_features.clear()
        if os.path.exists(self.spectral_features_path):
            os.remove(self.spectral_features_path)
        self._reset_session_state()
        print("[*] Database and spectral features reset complete")


def main():
    """Main entry point."""

    detector = VectorDBSpeakerDetector(
        similarity_threshold=0.7,       # Balanced threshold for fused scoring
        min_speech_duration=2.0,        # Faster detection
        vad_threshold=0.5,              # Sensitive VAD
        device='cpu',
        db_path='./speaker_vectordb',
        use_fft=True,                   # Enable spectral features
        fft_weight=0.20,                # Legacy FFT weight parameter
        nn_weight=0.60,                 # 60% neural network embedding
        spectral_fft_weight=0.20,       # 20% FFT spectral features
        spectral_mfcc_weight=0.20,      # 20% MFCC spectral features
        min_audio_energy=0.005,         # Audio quality gate: min RMS
        min_snr_db=5.0,                 # Audio quality gate: min SNR
        centroid_update_threshold=0.55, # Only update profile on high-confidence matches
        min_enrollment_samples=3        # Require at least 3 enrollment samples
    )

    while True:
        fft_status = "ON" if detector.use_fft else "OFF"
        print("\n" + "="*70)
        print("SPEAKER DETECTION WITH MULTI-FEATURE FUSION")
        print(f"Spectral Features: {fft_status} | Fusion: NN={detector.nn_weight:.0%} "
              f"FFT={detector.spectral_fft_weight:.0%} MFCC={detector.spectral_mfcc_weight:.0%}")
        print("="*70)
        print("\nOptions:")
        print("1. Start Continuous Detection")
        print("2. Record & Identify (5 sec)")
        print("3. Enroll Speaker")
        print("4. Reset Database")
        print("5. Show Database Stats")
        print("6. Toggle FFT Features")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            detector.start_listening()
        elif choice == "2":
            duration = input("Recording duration (seconds, default 5): ").strip()
            try:
                duration = float(duration) if duration else 5.0
            except:
                duration = 5.0
            detector.record_and_identify(duration)
        elif choice == "3":
            name = input("Enter speaker name: ").strip()
            if name:
                detector.start_listening(enrollment_speaker=name)
        elif choice == "4":
            confirm = input("Delete all data? (yes/no): ").strip().lower()
            if confirm == "yes":
                detector.reset_database()
        elif choice == "5":
            detector._print_summary()
        elif choice == "6":
            detector.use_fft = not detector.use_fft
            status = "ENABLED" if detector.use_fft else "DISABLED"
            print(f"\n[*] FFT Features {status}")
            if detector.use_fft:
                new_weight = input(f"FFT weight (current: {detector.fft_weight:.0%}, press Enter to keep): ").strip()
                if new_weight:
                    try:
                        detector.fft_weight = float(new_weight)
                        print(f"[*] FFT weight set to {detector.fft_weight:.0%}")
                    except:
                        print("[!] Invalid weight, keeping current value")
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-7.")


if __name__ == "__main__":
    main()
