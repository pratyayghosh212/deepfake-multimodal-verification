import subprocess
from pathlib import Path

import librosa
import numpy as np


# =========================================================
# Audio extraction
# =========================================================

def extract_audio(
    video_path: str,
    output_path: str
):
    """
    Extract the audio stream from a video
    and save it as a WAV file.

    Returns basic information about
    the extracted audio.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # FFmpeg command
    # -----------------------------------------------------

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),

        # Mono
        "-ac",
        "1",

        # 16 kHz
        "-ar",
        "16000",

        # 16-bit PCM WAV
        "-acodec",
        "pcm_s16le",

        str(output_path)
    ]

    print("Extracting audio...")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # -----------------------------------------------------
    # Check FFmpeg
    # -----------------------------------------------------

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed while extracting audio:\n"
            f"{result.stderr}"
        )

    # -----------------------------------------------------
    # Verify output
    # -----------------------------------------------------

    if not output_path.exists():
        raise RuntimeError(
            "FFmpeg completed but the audio file "
            "was not created."
        )

    print(
        f"Audio saved to: {output_path}"
    )

    return {
        "audio_path": str(output_path),
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
        "codec": "pcm_s16le"
    }


# =========================================================
# Acoustic feature extraction
# =========================================================

def analyze_audio(
    audio_path: str,
    output_path: str,
    sample_rate: int = 16000,
    window_duration: float = 0.2
):
    """
    Extract timestamped acoustic features.

    Each analysis window contains:

        RMS energy
        Zero-crossing rate
        Spectral centroid
        Spectral bandwidth
        Spectral rolloff

    The default window duration is 0.2 seconds,
    matching the 5 FPS video analysis.
    """

    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # -----------------------------------------------------
    # Validate audio
    # -----------------------------------------------------

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # -----------------------------------------------------
    # Load audio
    # -----------------------------------------------------

    print("Loading audio...")

    audio, actual_sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True
    )

    if len(audio) == 0:
        raise RuntimeError(
            "Audio file contains no samples."
        )

    duration = len(audio) / actual_sr

    print(
        f"Sample rate: {actual_sr} Hz"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print(
        f"Analysis window: "
        f"{window_duration:.2f} seconds"
    )

    # -----------------------------------------------------
    # Convert time window to samples
    # -----------------------------------------------------

    window_samples = int(
        window_duration * actual_sr
    )

    results = []

    # -----------------------------------------------------
    # Analyze each window
    # -----------------------------------------------------

    start_sample = 0
    window_index = 0

    while start_sample < len(audio):

        end_sample = min(
            start_sample + window_samples,
            len(audio)
        )

        chunk = audio[
            start_sample:end_sample
        ]

        if len(chunk) == 0:
            break

        start_time = (
            start_sample / actual_sr
        )

        end_time = (
            end_sample / actual_sr
        )

        # -------------------------------------------------
        # RMS energy
        # -------------------------------------------------

        rms = librosa.feature.rms(
            y=chunk
        )

        rms_value = float(
            np.mean(rms)
        )

        # -------------------------------------------------
        # Zero crossing rate
        # -------------------------------------------------

        zcr = librosa.feature.zero_crossing_rate(
            chunk
        )

        zcr_value = float(
            np.mean(zcr)
        )

        # -------------------------------------------------
        # Spectral features
        # -------------------------------------------------

        centroid = librosa.feature.spectral_centroid(
            y=chunk,
            sr=actual_sr
        )

        centroid_value = float(
            np.mean(centroid)
        )

        bandwidth = librosa.feature.spectral_bandwidth(
            y=chunk,
            sr=actual_sr
        )

        bandwidth_value = float(
            np.mean(bandwidth)
        )

        rolloff = librosa.feature.spectral_rolloff(
            y=chunk,
            sr=actual_sr,
            roll_percent=0.85
        )

        rolloff_value = float(
            np.mean(rolloff)
        )

        # -------------------------------------------------
        # Store evidence
        # -------------------------------------------------

        result = {
            "id": (
                f"audio_"
                f"{window_index:04d}"
            ),

            "type": "audio",

            "start": round(
                start_time,
                3
            ),

            "end": round(
                end_time,
                3
            ),

            "duration": round(
                end_time - start_time,
                3
            ),

            "source": "librosa_acoustic_analysis",

            "features": {
                "rms_energy": round(
                    rms_value,
                    6
                ),

                "zero_crossing_rate": round(
                    zcr_value,
                    6
                ),

                "spectral_centroid": round(
                    centroid_value,
                    3
                ),

                "spectral_bandwidth": round(
                    bandwidth_value,
                    3
                ),

                "spectral_rolloff": round(
                    rolloff_value,
                    3
                )
            }
        }

        results.append(result)

        print(
            f"[{start_time:.2f}s - "
            f"{end_time:.2f}s] "
            f"RMS={rms_value:.4f} | "
            f"ZCR={zcr_value:.4f} | "
            f"Centroid={centroid_value:.1f}"
        )

        # -------------------------------------------------
        # Move to next window
        # -------------------------------------------------

        start_sample = end_sample
        window_index += 1

    # -----------------------------------------------------
    # Save analysis
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        import json

        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print(
        f"Created {len(results)} "
        f"audio analysis windows."
    )

    print(
        f"Saved audio analysis to: "
        f"{output_path}"
    )

    return results


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Step 1: Extract audio
    # -----------------------------------------------------

    audio_info = extract_audio(
        video_path="data/videos/test.mp4",
        output_path="data/audio/audio.wav"
    )

    print()
    print("Audio information:")
    print(audio_info)

    # -----------------------------------------------------
    # Step 2: Analyze audio
    # -----------------------------------------------------

    analyze_audio(
        audio_path="data/audio/audio.wav",
        output_path=(
            "data/audio/"
            "audio_analysis.json"
        ),

        # Match our 5 FPS visual timeline
        window_duration=0.2
    )