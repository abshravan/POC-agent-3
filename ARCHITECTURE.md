# Speaker Identification System - Production Architecture

## Executive Summary

This document describes a production-ready speaker identification system designed to identify employees by voice biometrics and tag call transcripts automatically. The system uses **embedding-based recognition** with NVIDIA NeMo models, ensuring the core model remains static while embeddings are dynamically managed.

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SPEAKER IDENTIFICATION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │    Audio     │───▶│    Audio     │───▶│  Embedding   │───▶│   Vector   │ │
│  │   Ingestion  │    │ Preprocessing│    │  Extraction  │    │  Matching  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         FastAPI REST Layer                            │   │
│  │  ┌────────────┐  ┌────────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │  │ Enrollment │  │ Identification │  │    Admin    │  │  Health   │  │   │
│  │  │    API     │  │      API       │  │     API     │  │   Check   │  │   │
│  │  └────────────┘  └────────────────┘  └─────────────┘  └───────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         ▼                          ▼                          ▼             │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐        │
│  │   Qdrant     │         │  PostgreSQL  │         │  Prometheus  │        │
│  │ Vector Store │         │   Metadata   │         │   Metrics    │        │
│  └──────────────┘         └──────────────┘         └──────────────┘        │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Static Model, Dynamic Embeddings**: The NeMo speaker model is frozen; only speaker embeddings are added/removed
2. **Embedding-First Architecture**: All speaker matching is done via vector similarity search
3. **Horizontal Scalability**: Stateless API servers, scalable vector database
4. **Local-First Deployment**: Optimized for local machine with clear path to cloud scaling

---

## 2. Component Breakdown

### 2.1 Audio Ingestion Layer

**Purpose**: Accept audio from multiple sources and normalize format.

| Component | Description |
|-----------|-------------|
| **Stream Handler** | Accepts real-time audio streams (WebSocket, gRPC) |
| **File Handler** | Accepts uploaded audio files (WAV, MP3, FLAC, OGG) |
| **Format Normalizer** | Converts all audio to 16kHz mono PCM (NeMo requirement) |
| **Chunking Engine** | Splits long audio into processable segments (3-10 seconds optimal) |

```
Audio Input ──▶ Format Detection ──▶ Resampling ──▶ Mono Conversion ──▶ Chunking
     │                                                                      │
     └── 8kHz telephony ─▶ Upsample to 16kHz ─▶ Anti-aliasing filter ──────┘
```

**Key Considerations**:
- 8kHz telephony audio is upsampled using sinc interpolation
- Audio chunks overlap by 50% to avoid missing speaker boundaries
- Maximum chunk duration: 30 seconds (memory constraint)

### 2.2 Preprocessing Pipeline

**Purpose**: Clean and enhance audio for optimal embedding extraction.

```python
Pipeline:
1. DC Offset Removal          # Remove DC bias
2. Pre-emphasis (α=0.97)      # Boost high frequencies
3. Voice Activity Detection   # Extract speech segments only
4. Noise Reduction (optional) # Spectral subtraction for noisy environments
5. Amplitude Normalization    # Peak normalize to -3dB
6. Quality Assessment         # Reject low-SNR segments
```

| Step | Algorithm | Purpose |
|------|-----------|---------|
| VAD | Silero VAD / WebRTC VAD | Remove silence, detect speech regions |
| Noise Reduction | Spectral Gating | Office environment noise handling |
| Normalization | Peak + RMS | Consistent input levels |
| Quality Gate | SNR estimation | Reject segments < 10dB SNR |

### 2.3 Embedding Extraction (NVIDIA NeMo)

**Purpose**: Convert audio segments into fixed-dimensional speaker embeddings.

**Recommended Models** (ranked by accuracy/speed tradeoff):

| Model | Dimensions | Speed | Accuracy (VoxCeleb1-O) | Recommendation |
|-------|------------|-------|------------------------|----------------|
| **TitaNet-Large** | 192 | Medium | EER 0.68% | Best accuracy, production default |
| **TitaNet-Small** | 192 | Fast | EER 1.2% | High-throughput scenarios |
| **ECAPA-TDNN** | 192 | Medium | EER 0.87% | Good balance |

**Why NeMo TitaNet**:
1. State-of-the-art accuracy on speaker verification benchmarks
2. 192-dimensional embeddings (efficient storage and search)
3. Pre-trained on VoxCeleb (10,000+ speakers) - generalizes well
4. Apache 2.0 license - commercial use allowed
5. Excellent 8kHz degradation handling after resampling
6. Robust to noise and channel variations

```
Audio Segment ──▶ NeMo TitaNet ──▶ 192-dim Embedding ──▶ L2 Normalization
                       │
                       └── Static frozen model (no fine-tuning needed)
```

### 2.4 Vector Comparison & Matching

**Purpose**: Find the closest enrolled speaker given a query embedding.

**Algorithm**: Cosine Similarity Search with Confidence Scoring

```
Query Embedding
      │
      ▼
┌─────────────────┐
│  Vector Search  │──▶ Top-K similar embeddings (K=5)
│    (Qdrant)     │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Score Fusion    │──▶ Weighted average of top matches per speaker
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  Threshold      │──▶ Accept if confidence ≥ threshold
│  Decision       │──▶ Reject (Unknown) otherwise
└─────────────────┘
```

**Confidence Scoring Formula**:
```
confidence = (similarity - min_threshold) / (1.0 - min_threshold)
final_score = clamp(confidence, 0.0, 1.0)
```

**Threshold Strategy**:
- **Enrollment Verification Threshold**: 0.70 (strict - avoid false enrollments)
- **Identification Threshold**: 0.55 (balanced - allow some uncertainty)
- **High-Confidence Threshold**: 0.75 (auto-tag without review)

### 2.5 Database Layer

**Vector Store (Qdrant)**:
- Purpose: Store and search speaker embeddings
- Why Qdrant: Best local deployment, Rust-based (fast), excellent Python SDK
- Index Type: HNSW with cosine distance
- Capacity: 1600 users × 5 embeddings/user × 192 dims = ~6MB (trivial)

**Metadata Store (PostgreSQL/SQLite)**:
- Purpose: Store speaker metadata (name, department, enrollment date, etc.)
- Tables: `speakers`, `embeddings`, `identification_logs`, `thresholds`

```sql
-- Core schema
CREATE TABLE speakers (
    id UUID PRIMARY KEY,
    employee_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(100),
    enrolled_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    embedding_count INTEGER DEFAULT 0
);

CREATE TABLE identification_logs (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    audio_duration_sec FLOAT,
    identified_speaker_id UUID REFERENCES speakers(id),
    confidence FLOAT,
    threshold_used FLOAT,
    is_accepted BOOLEAN,
    processing_time_ms INTEGER
);
```

### 2.6 API Layer (FastAPI)

**Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/enroll` | POST | Enroll new speaker with audio samples |
| `/api/v1/enroll/{speaker_id}/add-sample` | POST | Add more samples to existing speaker |
| `/api/v1/identify` | POST | Identify speaker from audio |
| `/api/v1/identify/stream` | WebSocket | Real-time identification |
| `/api/v1/speakers` | GET | List all enrolled speakers |
| `/api/v1/speakers/{id}` | DELETE | Remove speaker (soft delete) |
| `/api/v1/speakers/{id}/embeddings` | DELETE | Clear and re-enroll speaker |
| `/api/v1/admin/thresholds` | GET/PUT | Manage identification thresholds |
| `/api/v1/health` | GET | Health check |
| `/api/v1/metrics` | GET | Prometheus metrics |

### 2.7 Monitoring Layer

**Metrics to Track**:

| Metric | Type | Description |
|--------|------|-------------|
| `speaker_id_requests_total` | Counter | Total identification requests |
| `speaker_id_latency_seconds` | Histogram | End-to-end identification latency |
| `speaker_id_confidence_score` | Histogram | Confidence score distribution |
| `speaker_id_acceptance_rate` | Gauge | % of identifications above threshold |
| `speaker_id_unknown_rate` | Gauge | % rejected as unknown |
| `enrollment_total` | Counter | Total enrollments |
| `embedding_extraction_seconds` | Histogram | Model inference time |
| `vector_search_seconds` | Histogram | Qdrant query time |
| `audio_quality_score` | Histogram | Input audio SNR distribution |

---

## 3. Recommended Models

### Primary: NVIDIA NeMo TitaNet-Large

```python
# Model loading
import nemo.collections.asr as nemo_asr
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    "nvidia/speakerverification_en_titanet_large"
)
```

**Justification**:
1. **Best-in-class accuracy**: 0.68% EER on VoxCeleb1-O (industry benchmark)
2. **Robust to noise**: Trained with data augmentation (noise, reverb, codec)
3. **Efficient**: 192-dim embeddings, ~23M parameters
4. **Proven at scale**: Used in NVIDIA Riva for enterprise deployments
5. **8kHz compatible**: Works well with upsampled telephony audio
6. **No fine-tuning needed**: Pre-trained model works out of the box

### Fallback: ECAPA-TDNN (SpeechBrain)

For systems without GPU or requiring faster inference:
```python
from speechbrain.inference.speaker import EncoderClassifier
model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")
```

---

## 4. Suggested Tech Stack

### Core Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Language** | Python 3.10+ | NeMo/PyTorch ecosystem |
| **API Framework** | FastAPI | Async, OpenAPI, fast |
| **ML Framework** | PyTorch + NeMo | NVIDIA speaker models |
| **Vector DB** | Qdrant | Local deployment, Rust performance |
| **Metadata DB** | PostgreSQL (prod) / SQLite (dev) | Reliable, scalable |
| **Task Queue** | Celery + Redis (optional) | Async processing for batch |
| **Caching** | Redis | Embedding cache, rate limiting |
| **Monitoring** | Prometheus + Grafana | Industry standard |
| **Logging** | Structlog + Loki | Structured, searchable |
| **Containerization** | Docker + Docker Compose | Reproducible deployment |

### Local Development Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | None (CPU inference) | NVIDIA GPU (4GB+ VRAM) |
| **Storage** | 10 GB | 50 GB (for models + logs) |

---

## 5. Scalability Strategy

### Phase 1: Local Single-Machine (10-50 users)

```
┌─────────────────────────────────┐
│      Docker Compose Stack       │
│  ┌───────┐ ┌───────┐ ┌───────┐ │
│  │  API  │ │Qdrant │ │SQLite │ │
│  │(1 CPU)│ │(local)│ │(file) │ │
│  └───────┘ └───────┘ └───────┘ │
└─────────────────────────────────┘
```

### Phase 2: Horizontal Scaling (100-500 users)

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │  API 1  │          │  API 2  │          │  API 3  │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
              ┌──────────────────────────┐
              │    Shared Qdrant Cloud   │
              │      + PostgreSQL        │
              └──────────────────────────┘
```

### Phase 3: Enterprise Scale (1000-1600+ users)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Ingress Controller                        ││
│  └────────────────────────────┬────────────────────────────────┘│
│                               │                                  │
│  ┌────────────────────────────┼────────────────────────────────┐│
│  │              API Deployment (3-10 replicas)                  ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        ││
│  │  │  Pod 1  │  │  Pod 2  │  │  Pod 3  │  │  Pod N  │        ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        ││
│  └─────────────────────────────────────────────────────────────┘│
│                               │                                  │
│  ┌────────────────────────────┼────────────────────────────────┐│
│  │    ┌─────────────┐    ┌─────────────┐    ┌────────────┐    ││
│  │    │   Qdrant    │    │ PostgreSQL  │    │   Redis    │    ││
│  │    │  (Cluster)  │    │  (Primary)  │    │  (Cache)   │    ││
│  │    └─────────────┘    └─────────────┘    └────────────┘    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Scaling Calculations

For 1600 users with 5 embeddings each:
- **Embeddings**: 1600 × 5 = 8,000 vectors
- **Storage**: 8,000 × 192 × 4 bytes = 6.1 MB (negligible)
- **Search latency**: <10ms with HNSW index
- **Throughput**: 100+ identifications/second per API instance

---

## 6. Cost Optimization Strategies

### Local Deployment

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| CPU inference | Eliminates GPU cost | Use ONNX Runtime optimized model |
| Embedding cache | Reduce redundant extractions | Redis cache with 1-hour TTL |
| Batch processing | Amortize overhead | Process multiple requests together |
| Model quantization | 2x faster inference | INT8 quantization with minimal accuracy loss |
| SQLite for small scale | No DB server needed | Single-file database |

### Cloud Deployment

| Optimization | Savings | Implementation |
|--------------|---------|----------------|
| Spot instances | 60-70% | API servers on spot/preemptible |
| Right-sizing | 30-40% | Start small, scale based on metrics |
| Qdrant Cloud | ~$50/month | Managed vector DB vs self-hosted |
| Reserved instances | 30-40% | For stable baseline load |

---

## 7. Accuracy Optimization Techniques

### Enrollment Phase

1. **Multiple Samples**: Require 3-5 diverse samples per speaker
2. **Quality Gating**: Reject samples with SNR < 15dB
3. **Consistency Check**: Ensure intra-speaker similarity > 0.80
4. **Augmented Enrollment**: Store embeddings from augmented audio (noise, reverb)

### Identification Phase

1. **Multi-Embedding Matching**: Compare against all speaker embeddings, not just centroid
2. **Score Aggregation**: Use weighted average of top-K matches per speaker
3. **Temporal Smoothing**: For continuous streams, smooth predictions over time
4. **Adaptive Thresholds**: Per-speaker thresholds based on enrollment quality

### Model-Level

1. **Domain Adaptation**: Fine-tune on similar audio domain if accuracy insufficient
2. **Ensemble**: Combine TitaNet + ECAPA-TDNN predictions
3. **Feature Fusion**: Combine speaker embeddings with prosodic features

---

## 8. Handling Similar-Sounding Speakers

### Problem

Some speakers may have very similar voice characteristics, leading to confusion.

### Solutions

| Strategy | Implementation |
|----------|----------------|
| **Distinctiveness Check** | At enrollment, warn if new speaker is too similar to existing (similarity > 0.70) |
| **Multi-Sample Enrollment** | More samples = better speaker discrimination |
| **Confusion Matrix Monitoring** | Track which speakers are frequently confused |
| **Speaker Pairs** | For known similar pairs, require higher confidence threshold |
| **Contextual Signals** | Use call metadata (caller ID, time) as additional signals |
| **Human-in-Loop** | For borderline cases, flag for human review |

### Detection Algorithm

```python
def check_similar_speakers(new_embedding, threshold=0.70):
    """Detect if new speaker is too similar to existing speakers."""
    all_embeddings = vector_store.get_all()
    similarities = cosine_similarity(new_embedding, all_embeddings)

    for speaker_id, sim in similarities:
        if sim > threshold:
            yield SimilarityWarning(
                existing_speaker=speaker_id,
                similarity=sim,
                recommendation="Collect more diverse samples"
            )
```

---

## 9. Data Privacy and Compliance

### Privacy Considerations

| Concern | Mitigation |
|---------|------------|
| **Biometric Data** | Voice embeddings are biometric; treat as PII |
| **Storage** | Encrypt embeddings at rest (AES-256) |
| **Transmission** | HTTPS/TLS only |
| **Retention** | Define retention policy; implement purge mechanism |
| **Consent** | Require explicit consent before enrollment |
| **Access Control** | Role-based access to speaker data |

### Compliance Checklist

- [ ] **GDPR**: Right to erasure implemented (delete speaker endpoint)
- [ ] **CCPA**: Data inventory and deletion requests
- [ ] **BIPA**: Biometric consent workflow (if applicable)
- [ ] **SOC 2**: Audit logging, access controls, encryption
- [ ] **Data Residency**: Keep data in required jurisdiction

### Security Measures

```python
# Embedding encryption at rest
from cryptography.fernet import Fernet

class SecureEmbeddingStore:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)

    def encrypt(self, embedding: np.ndarray) -> bytes:
        return self.cipher.encrypt(embedding.tobytes())

    def decrypt(self, encrypted: bytes) -> np.ndarray:
        return np.frombuffer(self.cipher.decrypt(encrypted), dtype=np.float32)
```

---

## 10. Failure Scenarios and Fallback Mechanisms

### Failure Matrix

| Scenario | Detection | Fallback | Recovery |
|----------|-----------|----------|----------|
| **Model loading fails** | Health check fails | Return 503, use cached model | Restart container |
| **Vector DB unavailable** | Connection timeout | Queue requests, retry | Reconnect with backoff |
| **Audio too short** | Duration < 1 second | Return error with guidance | N/A |
| **Audio quality poor** | SNR < threshold | Return low-confidence result | Request better sample |
| **No match found** | Confidence < threshold | Return "Unknown" with scores | N/A |
| **High latency** | P99 > SLA | Circuit breaker, shed load | Scale up |
| **OOM error** | Memory usage spike | Reduce batch size | Restart with limits |

### Circuit Breaker Pattern

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def identify_speaker(audio_data):
    """Identify speaker with circuit breaker protection."""
    embedding = extract_embedding(audio_data)
    results = vector_store.search(embedding, top_k=5)
    return score_and_threshold(results)
```

### Graceful Degradation

1. **Primary Path**: Full NeMo TitaNet inference
2. **Fallback 1**: Cached embeddings only (no new extractions)
3. **Fallback 2**: Return "Unknown" for all requests
4. **Fallback 3**: Queue requests for later processing

---

## 11. Folder/Project Structure

```
speaker-id-system/
├── README.md                           # Quick start guide
├── ARCHITECTURE.md                     # This document
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Modern Python project config
├── docker-compose.yml                  # Local development stack
├── docker-compose.prod.yml             # Production stack
├── Dockerfile                          # API container
├── Makefile                            # Common commands
├── .env.example                        # Environment template
├── .gitignore                          # Git ignore rules
│
├── config/
│   ├── __init__.py
│   ├── settings.py                     # Pydantic settings management
│   └── logging_config.py               # Structured logging setup
│
├── src/
│   ├── __init__.py
│   │
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── preprocessing.py            # Audio normalization pipeline
│   │   ├── vad.py                       # Voice activity detection
│   │   ├── quality.py                   # Audio quality assessment
│   │   └── formats.py                   # Format conversion utilities
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── extractor.py                 # NeMo embedding extraction
│   │   ├── cache.py                     # Embedding cache layer
│   │   └── models.py                    # Model loading/management
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── vector_store.py              # Vector DB abstraction
│   │   ├── qdrant_client.py             # Qdrant implementation
│   │   ├── speaker_repository.py        # Speaker CRUD
│   │   └── models.py                    # SQLAlchemy models
│   │
│   ├── identification/
│   │   ├── __init__.py
│   │   ├── engine.py                    # Core identification logic
│   │   ├── scorer.py                    # Confidence scoring
│   │   ├── threshold.py                 # Threshold management
│   │   └── similar_speakers.py          # Similar speaker detection
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI application
│   │   ├── dependencies.py              # Dependency injection
│   │   ├── schemas.py                   # Pydantic request/response models
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── enrollment.py            # Enrollment endpoints
│   │       ├── identification.py        # Identification endpoints
│   │       ├── speakers.py              # Speaker management
│   │       └── admin.py                 # Admin/config endpoints
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── metrics.py                   # Prometheus metrics
│       ├── health.py                    # Health checks
│       └── logging.py                   # Request logging
│
├── scripts/
│   ├── download_models.py               # Download NeMo models
│   ├── benchmark.py                     # Performance benchmarking
│   ├── evaluate_thresholds.py           # Threshold optimization
│   ├── migrate_db.py                    # Database migrations
│   └── generate_test_audio.py           # Generate test data
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Pytest fixtures
│   ├── test_audio/
│   │   ├── test_preprocessing.py
│   │   └── test_quality.py
│   ├── test_embeddings/
│   │   └── test_extractor.py
│   ├── test_identification/
│   │   ├── test_engine.py
│   │   └── test_scorer.py
│   └── test_api/
│       ├── test_enrollment.py
│       └── test_identification.py
│
├── models/                              # Downloaded model files (gitignored)
│   └── .gitkeep
│
└── data/                                # Local data storage (gitignored)
    ├── qdrant/                          # Vector DB storage
    ├── sqlite/                          # SQLite database
    └── audio_samples/                   # Test audio files
```

---

## 12. Sample Pseudo-Code

### Enrollment Flow

```python
async def enroll_speaker(
    employee_id: str,
    name: str,
    audio_samples: List[UploadFile],
    min_samples: int = 3
) -> EnrollmentResult:
    """
    Enroll a new speaker with audio samples.

    Steps:
    1. Validate audio samples (format, duration, quality)
    2. Extract embeddings from each sample
    3. Check embedding consistency
    4. Check for similar existing speakers
    5. Store embeddings in vector database
    6. Create speaker record in metadata database
    """
    # Step 1: Validate audio samples
    if len(audio_samples) < min_samples:
        raise ValidationError(f"Need at least {min_samples} audio samples")

    validated_audios = []
    for sample in audio_samples:
        audio = await load_and_preprocess(sample)
        quality = assess_quality(audio)

        if quality.snr_db < 15:
            raise AudioQualityError(f"Sample too noisy: SNR={quality.snr_db}dB")
        if quality.duration < 2.0:
            raise AudioQualityError(f"Sample too short: {quality.duration}s")

        validated_audios.append(audio)

    # Step 2: Extract embeddings
    embeddings = []
    for audio in validated_audios:
        embedding = embedding_extractor.extract(audio)
        embeddings.append(embedding)

    # Step 3: Check embedding consistency
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    similarities = [np.dot(e, centroid) for e in embeddings]
    consistency = np.mean(similarities)

    if consistency < 0.80:
        raise ConsistencyError(
            f"Samples inconsistent (score={consistency:.2f}). "
            "Ensure all samples are from the same speaker."
        )

    # Step 4: Check for similar existing speakers
    similar_speakers = await check_similar_speakers(centroid, threshold=0.70)
    if similar_speakers:
        warnings = [
            f"Similar to {s.name} (similarity={s.score:.2f})"
            for s in similar_speakers
        ]

    # Step 5: Store embeddings in vector database
    speaker_id = generate_uuid()
    for i, embedding in enumerate(embeddings):
        await vector_store.insert(
            id=f"{speaker_id}_{i}",
            vector=embedding,
            payload={"speaker_id": speaker_id, "sample_index": i}
        )

    # Step 6: Create speaker record
    speaker = await speaker_repository.create(
        id=speaker_id,
        employee_id=employee_id,
        name=name,
        embedding_count=len(embeddings),
        consistency_score=consistency
    )

    return EnrollmentResult(
        speaker_id=speaker_id,
        embeddings_stored=len(embeddings),
        consistency_score=consistency,
        similar_speakers=warnings if similar_speakers else None
    )
```

### Identification Flow

```python
async def identify_speaker(
    audio: UploadFile,
    threshold: float = 0.55,
    top_k: int = 5
) -> IdentificationResult:
    """
    Identify speaker from audio.

    Steps:
    1. Preprocess and validate audio
    2. Extract embedding
    3. Search vector database for similar embeddings
    4. Aggregate scores by speaker
    5. Apply threshold and return result
    """
    start_time = time.perf_counter()

    # Step 1: Preprocess audio
    audio_data = await load_and_preprocess(audio)
    quality = assess_quality(audio_data)

    if quality.snr_db < 10:
        return IdentificationResult(
            speaker_id=None,
            speaker_name="Unknown",
            confidence=0.0,
            reason="Audio quality too low",
            quality_score=quality.snr_db
        )

    # Step 2: Extract embedding
    query_embedding = embedding_extractor.extract(audio_data)

    # Step 3: Search vector database
    search_results = await vector_store.search(
        vector=query_embedding,
        limit=top_k * 3  # Get more results for aggregation
    )

    # Step 4: Aggregate scores by speaker
    speaker_scores = defaultdict(list)
    for result in search_results:
        speaker_id = result.payload["speaker_id"]
        speaker_scores[speaker_id].append(result.score)

    # Compute aggregated score per speaker
    aggregated = []
    for speaker_id, scores in speaker_scores.items():
        # Use weighted average: higher scores have more weight
        weights = np.array(scores) ** 2
        avg_score = np.average(scores, weights=weights)
        aggregated.append((speaker_id, avg_score))

    aggregated.sort(key=lambda x: x[1], reverse=True)

    # Step 5: Apply threshold
    if not aggregated or aggregated[0][1] < threshold:
        return IdentificationResult(
            speaker_id=None,
            speaker_name="Unknown",
            confidence=aggregated[0][1] if aggregated else 0.0,
            reason="Below confidence threshold",
            all_scores=aggregated[:5]
        )

    # Get speaker details
    best_speaker_id, best_score = aggregated[0]
    speaker = await speaker_repository.get(best_speaker_id)

    processing_time = (time.perf_counter() - start_time) * 1000

    # Log identification for monitoring
    await log_identification(
        speaker_id=best_speaker_id,
        confidence=best_score,
        threshold=threshold,
        accepted=True,
        processing_time_ms=processing_time
    )

    return IdentificationResult(
        speaker_id=best_speaker_id,
        speaker_name=speaker.name,
        employee_id=speaker.employee_id,
        confidence=best_score,
        processing_time_ms=processing_time,
        all_scores=aggregated[:5]
    )
```

### User Removal Flow

```python
async def remove_speaker(
    speaker_id: str,
    hard_delete: bool = False
) -> RemovalResult:
    """
    Remove a speaker from the system.

    Steps:
    1. Verify speaker exists
    2. Delete embeddings from vector database
    3. Soft-delete (or hard-delete) speaker record
    4. Log removal for audit
    """
    # Step 1: Verify speaker exists
    speaker = await speaker_repository.get(speaker_id)
    if not speaker:
        raise NotFoundError(f"Speaker {speaker_id} not found")

    # Step 2: Delete embeddings from vector database
    # Find all embeddings for this speaker
    deleted_count = await vector_store.delete(
        filter={"speaker_id": speaker_id}
    )

    # Step 3: Handle speaker record
    if hard_delete:
        # GDPR right to erasure - complete removal
        await speaker_repository.hard_delete(speaker_id)
        action = "hard_deleted"
    else:
        # Soft delete - keep record for audit
        await speaker_repository.update(
            speaker_id,
            is_active=False,
            deactivated_at=datetime.utcnow()
        )
        action = "soft_deleted"

    # Step 4: Audit log
    await audit_log.record(
        action="speaker_removed",
        speaker_id=speaker_id,
        speaker_name=speaker.name,
        embeddings_deleted=deleted_count,
        deletion_type=action
    )

    return RemovalResult(
        speaker_id=speaker_id,
        speaker_name=speaker.name,
        embeddings_deleted=deleted_count,
        action=action
    )
```

---

## 13. Monitoring Metrics

### Application Metrics (Prometheus)

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
IDENTIFICATION_REQUESTS = Counter(
    "speaker_id_requests_total",
    "Total identification requests",
    ["status"]  # success, unknown, error
)

IDENTIFICATION_LATENCY = Histogram(
    "speaker_id_latency_seconds",
    "Identification latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

CONFIDENCE_SCORES = Histogram(
    "speaker_id_confidence_score",
    "Confidence score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Enrollment metrics
ENROLLMENTS_TOTAL = Counter(
    "speaker_enrollments_total",
    "Total enrollment attempts",
    ["status"]  # success, failed_quality, failed_consistency
)

# Performance metrics
EMBEDDING_EXTRACTION_TIME = Histogram(
    "embedding_extraction_seconds",
    "Time to extract embedding",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0]
)

VECTOR_SEARCH_TIME = Histogram(
    "vector_search_seconds",
    "Time for vector similarity search",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
)

# System metrics
ENROLLED_SPEAKERS = Gauge(
    "speaker_enrolled_total",
    "Total number of enrolled speakers"
)

EMBEDDINGS_STORED = Gauge(
    "speaker_embeddings_total",
    "Total number of stored embeddings"
)

# Quality metrics
AUDIO_QUALITY_SNR = Histogram(
    "audio_quality_snr_db",
    "Audio quality (SNR in dB)",
    buckets=[5, 10, 15, 20, 25, 30, 40]
)
```

### Grafana Dashboard Panels

1. **Request Rate**: Identification requests per minute
2. **Latency Percentiles**: P50, P95, P99 latency over time
3. **Confidence Distribution**: Histogram of confidence scores
4. **Acceptance Rate**: % of identifications above threshold
5. **Unknown Rate**: % rejected as unknown
6. **Error Rate**: % of failed requests
7. **Enrollment Funnel**: Success vs failure breakdown
8. **Model Inference Time**: Embedding extraction latency
9. **Vector Search Time**: Database query latency
10. **Audio Quality**: SNR distribution of incoming audio

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: speaker-id-alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, speaker_id_latency_seconds) > 2.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High identification latency (P95 > 2s)"

      - alert: HighErrorRate
        expr: rate(speaker_id_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate (>5%)"

      - alert: LowAcceptanceRate
        expr: rate(speaker_id_requests_total{status="success"}[1h]) / rate(speaker_id_requests_total[1h]) < 0.7
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Low acceptance rate (<70%)"

      - alert: VectorDBDown
        expr: up{job="qdrant"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Vector database is down"
```

---

## Summary

This architecture provides a production-ready foundation for a speaker identification system that:

- Scales from 10 to 1600+ users without model retraining
- Uses state-of-the-art NVIDIA NeMo TitaNet for accuracy
- Supports local deployment with clear cloud migration path
- Handles poor audio quality and similar-sounding speakers
- Provides comprehensive monitoring and observability
- Meets privacy and compliance requirements
- Includes robust failure handling and fallback mechanisms

The embedding-based approach ensures that adding or removing employees is instantaneous and does not require any model changes, making it ideal for dynamic organizational environments.
