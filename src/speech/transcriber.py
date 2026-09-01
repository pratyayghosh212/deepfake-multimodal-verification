import json
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(
    audio_path: str,
    output_path: str,
    model_size: str = "base"
):
    """
    Transcribe audio using Faster-Whisper.

    The output contains:
    - detected language
    - language probability
    - timestamped transcript segments
    """

    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load Whisper model
    # -----------------------------------------------------

    print(
        f"Loading Whisper model: {model_size}"
    )

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )

    # -----------------------------------------------------
    # Transcribe
    # -----------------------------------------------------

    print("Transcribing audio...")

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5
    )

    # -----------------------------------------------------
    # Convert Whisper segments
    # into JSON-serializable objects
    # -----------------------------------------------------

    transcript_segments = []

    for segment in segments:

        text = segment.text.strip()

        # Ignore empty segments
        if not text:
            continue

        entry = {
            "segment_id": len(
                transcript_segments
            ),

            "start": round(
                segment.start,
                3
            ),

            "end": round(
                segment.end,
                3
            ),

            "text": text
        }

        transcript_segments.append(
            entry
        )

        print(
            f"[{entry['start']:.2f}s - "
            f"{entry['end']:.2f}s] "
            f"{entry['text']}"
        )

    # -----------------------------------------------------
    # Build final transcript object
    # -----------------------------------------------------

    result = {
        "audio_file": str(
            audio_path
        ),

        "model": model_size,

        "language": info.language,

        "language_probability": round(
            info.language_probability,
            4
        ),

        "segments": transcript_segments
    }

    # -----------------------------------------------------
    # Save transcript
    # -----------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print(
        "Transcription complete."
    )

    print(
        f"Language: "
        f"{info.language}"
    )

    print(
        f"Language probability: "
        f"{info.language_probability:.4f}"
    )

    print(
        f"Segments: "
        f"{len(transcript_segments)}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return result


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    transcribe_audio(
        audio_path=(
            "data/audio/audio.wav"
        ),

        output_path=(
            "data/transcripts/"
            "transcript.json"
        ),

        model_size="base"
    )