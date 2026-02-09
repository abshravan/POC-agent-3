# Speaker Identification System

Production-ready speaker identification system using NVIDIA NeMo embeddings.

Identify employees by voice biometrics and automatically tag call transcripts with speaker identity.

## Features

- **Embedding-based Recognition**: Uses NVIDIA NeMo TitaNet for state-of-the-art speaker embeddings
- **No Model Retraining**: Static model with dynamic embedding management
- **Scalable**: Designed for 10-1600+ enrolled users
- **Real-time Inference**: Sub-second identification latency
- **Quality Aware**: Automatic audio quality assessment and rejection
- **Privacy Compliant**: GDPR-ready with full deletion support

## Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM (16GB recommended)
- Optional: NVIDIA GPU with CUDA support

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd speaker-id-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NeMo models (optional, will download on first use)
# python scripts/download_models.py
```

### Running the API

```bash
# Development mode
python -m src.api.main

# Or with uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t speaker-id-system .
docker run -p 8000:8000 speaker-id-system
```

## API Usage

### Enroll a Speaker

```bash
curl -X POST "http://localhost:8000/api/v1/enroll" \
  -F "employee_id=EMP001" \
  -F "name=John Doe" \
  -F "department=Engineering" \
  -F "audio_files=@sample1.wav" \
  -F "audio_files=@sample2.wav" \
  -F "audio_files=@sample3.wav"
```

### Identify a Speaker

```bash
curl -X POST "http://localhost:8000/api/v1/identify" \
  -F "audio_file=@unknown_speaker.wav"
```

### List Enrolled Speakers

```bash
curl "http://localhost:8000/api/v1/speakers"
```

### Remove a Speaker

```bash
curl -X DELETE "http://localhost:8000/api/v1/speakers/{speaker_id}"
```

## Configuration

Configuration is managed through environment variables. See `.env.example` for all options.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | 8000 | API server port |
| `DEVICE` | cpu | Compute device (cpu, cuda) |
| `IDENTIFICATION_THRESHOLD` | 0.55 | Confidence threshold for identification |
| `MIN_ENROLLMENT_SAMPLES` | 3 | Minimum audio samples for enrollment |
| `QDRANT_PATH` | ./data/qdrant | Vector database storage path |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Speaker Identification System                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio ──▶ Preprocessing ──▶ NeMo TitaNet ──▶ Vector Search    │
│                                      │              │            │
│                                      ▼              ▼            │
│                              192-dim Embedding   Qdrant DB      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Audio Requirements

- **Format**: WAV, MP3, FLAC, OGG (auto-converted to 16kHz mono)
- **Duration**: 1-30 seconds per sample
- **Quality**: SNR > 10dB recommended
- **Enrollment**: Minimum 3 samples (5+ recommended)

## API Documentation

Interactive API documentation available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
speaker-id-system/
├── src/
│   ├── audio/          # Audio preprocessing
│   ├── embeddings/     # NeMo embedding extraction
│   ├── database/       # Vector store & speaker repository
│   ├── identification/ # Core identification engine
│   ├── api/            # FastAPI application
│   └── monitoring/     # Prometheus metrics
├── config/             # Configuration management
├── scripts/            # Utility scripts
├── tests/              # Test suite
├── data/               # Local data storage
└── models/             # Downloaded ML models
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Scaling

- **10-50 users**: Single Docker container
- **100-500 users**: Horizontal API scaling + managed vector DB
- **1000+ users**: Kubernetes deployment with Qdrant cluster

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed scaling strategies.

## License

MIT License - See LICENSE file for details.
