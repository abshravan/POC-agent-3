#!/usr/bin/env python3
"""
Download and cache speaker embedding models.

This script pre-downloads the NeMo TitaNet model to avoid
delays during first API request.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def download_nemo_model():
    """Download NVIDIA NeMo TitaNet model."""
    console.print("\n[bold blue]Downloading NVIDIA NeMo TitaNet Model[/bold blue]\n")

    try:
        import nemo.collections.asr as nemo_asr

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading TitaNet-Large...", total=None)

            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                "nvidia/speakerverification_en_titanet_large"
            )

            progress.update(task, completed=True)

        console.print("[green]NeMo TitaNet model downloaded successfully![/green]")
        console.print(f"Model type: {type(model).__name__}")

        return True

    except ImportError:
        console.print("[yellow]NeMo not installed. Trying SpeechBrain fallback...[/yellow]")
        return download_speechbrain_model()

    except Exception as e:
        console.print(f"[red]Failed to download NeMo model: {e}[/red]")
        return False


def download_speechbrain_model():
    """Download SpeechBrain ECAPA-TDNN model as fallback."""
    console.print("\n[bold blue]Downloading SpeechBrain ECAPA-TDNN Model[/bold blue]\n")

    try:
        from speechbrain.inference.speaker import EncoderClassifier

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading ECAPA-TDNN...", total=None)

            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speechbrain-ecapa",
            )

            progress.update(task, completed=True)

        console.print("[green]SpeechBrain model downloaded successfully![/green]")
        return True

    except Exception as e:
        console.print(f"[red]Failed to download SpeechBrain model: {e}[/red]")
        return False


def download_silero_vad():
    """Download Silero VAD model."""
    console.print("\n[bold blue]Downloading Silero VAD Model[/bold blue]\n")

    try:
        import torch

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading Silero VAD...", total=None)

            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
            )

            progress.update(task, completed=True)

        console.print("[green]Silero VAD model downloaded successfully![/green]")
        return True

    except Exception as e:
        console.print(f"[red]Failed to download Silero VAD: {e}[/red]")
        return False


def main():
    """Main entry point."""
    console.print("[bold]Speaker Identification Model Downloader[/bold]")
    console.print("=" * 50)

    success = True

    # Download speaker embedding model
    if not download_nemo_model():
        success = False

    # Download VAD model (optional)
    if not download_silero_vad():
        console.print("[yellow]VAD model download failed (optional)[/yellow]")

    console.print("\n" + "=" * 50)

    if success:
        console.print("[bold green]All required models downloaded![/bold green]")
        console.print("\nYou can now start the API with:")
        console.print("  python -m src.api.main")
    else:
        console.print("[bold red]Some models failed to download.[/bold red]")
        console.print("Check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
