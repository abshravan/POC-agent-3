"""Monitoring module for speaker identification."""

from src.monitoring.metrics import setup_metrics, get_metrics, SpeakerIDMetrics

__all__ = ["setup_metrics", "get_metrics", "SpeakerIDMetrics"]
