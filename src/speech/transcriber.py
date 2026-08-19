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
    - start timestamp
    - end timestamp
    - transcript text
    """

    audio_path = Path(audio_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading Whisper model...")

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )

    print("Transcribing audio...")

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5
    )

    transcript = []

    for segment in segments:

        

        entry = {
            "segment_id": len(transcript),
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip()
        }

        transcript.append(entry)

        print(
            f"[{entry['start']:.2f}s - "
            f"{entry['end']:.2f}s] "
            f"{entry['text']}"
        )

    result = {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": transcript
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nTranscription complete.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":

    transcribe_audio(
        audio_path="data/audio/audio.wav",
        output_path="data/transcripts/transcript.json",
        model_size="base"
    )