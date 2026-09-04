import json
import subprocess
from pathlib import Path

import librosa
import numpy as np


# =========================================================
# Configuration
# =========================================================

VIDEO_PATH = "data/videos/test.mp4"
AUDIO_PATH = "data/audio/audio.wav"
ANALYSIS_PATH = "data/audio/audio_analysis.json"

SAMPLE_RATE = 16000
WINDOW_DURATION = 0.2


# =========================================================
# Audio extraction
# =========================================================

def extract_audio(video_path: str, output_path: str):
    """
    Extract the audio stream from a video
    and save it as a WAV file.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    # Validate input video
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # Create output directory
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # FFmpeg command
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),

        # Convert to mono
        "-ac",
        "1",

        # Sample rate
        "-ar",
        str(SAMPLE_RATE),

        # PCM WAV
        "-acodec",
        "pcm_s16le",

        str(output_path)
    ]

    print("\nExtracting audio...")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Check FFmpeg result
    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed while extracting audio:\n\n"
            f"{result.stderr}"
        )

    # Verify output
    if not output_path.exists():
        raise RuntimeError(
            "Audio extraction completed, "
            "but the output file was not created."
        )

    print(
        f"Audio saved to: {output_path}"
    )

    return {
        "audio_path": str(output_path),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "format": "wav",
        "codec": "pcm_s16le"
    }


# =========================================================
# Check whether audio contains meaningful signal
# =========================================================

def check_audio_signal(audio):
    """
    Check whether the extracted audio contains
    meaningful non-silent signal.
    """

    max_amplitude = float(
        np.max(np.abs(audio))
    )

    rms_amplitude = float(
        np.sqrt(np.mean(audio ** 2))
    )

    return {
        "max_amplitude": max_amplitude,
        "rms_amplitude": rms_amplitude
    }


# =========================================================
# Acoustic feature extraction
# =========================================================

def analyze_audio(
    audio_path: str,
    output_path: str,
    sample_rate: int = SAMPLE_RATE,
    window_duration: float = WINDOW_DURATION
):
    """
    Extract timestamped acoustic features.

    Features:
        - RMS energy
        - Zero-crossing rate
        - Spectral centroid
        - Spectral bandwidth
        - Spectral rolloff

    Also detects silent audio windows.
    """

    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Validate audio file
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # Load audio
    print("\nLoading audio...")

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
    # Check overall audio signal
    # -----------------------------------------------------

    signal_info = check_audio_signal(audio)

    print("\nAudio signal check:")

    print(
        f"Maximum amplitude: "
        f"{signal_info['max_amplitude']:.6f}"
    )

    print(
        f"Overall RMS: "
        f"{signal_info['rms_amplitude']:.6f}"
    )

    if signal_info["max_amplitude"] < 0.0001:

        print(
            "\nWARNING: The extracted audio appears "
            "to be silent or nearly silent."
        )

    # -----------------------------------------------------
    # Window setup
    # -----------------------------------------------------

    window_samples = int(
        window_duration * actual_sr
    )

    results = []

    start_sample = 0
    window_index = 0

    print("\nAnalyzing audio windows...\n")

    # -----------------------------------------------------
    # Analyze windows
    # -----------------------------------------------------

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
        # Check silence
        # -------------------------------------------------

        chunk_max = float(
            np.max(np.abs(chunk))
        )

        chunk_rms = float(
            np.sqrt(np.mean(chunk ** 2))
        )

        is_silent = chunk_max < 0.0001

        # -------------------------------------------------
        # RMS Energy
        # -------------------------------------------------

        rms_value = float(
            np.sqrt(np.mean(chunk ** 2))
        )

        # -------------------------------------------------
        # Zero Crossing Rate
        # -------------------------------------------------

        zcr = librosa.feature.zero_crossing_rate(
            y=chunk
        )

        zcr_value = float(
            np.mean(zcr)
        )

        # -------------------------------------------------
        # Spectral features
        # -------------------------------------------------

        # Very short windows can cause FFT issues,
        # so choose a valid FFT size.

        n_fft = min(
            2048,
            len(chunk)
        )

        if n_fft < 256:
            n_fft = len(chunk)

        hop_length = max(
            1,
            n_fft // 4
        )

        centroid = librosa.feature.spectral_centroid(
            y=chunk,
            sr=actual_sr,
            n_fft=n_fft,
            hop_length=hop_length
        )

        centroid_value = float(
            np.mean(centroid)
        )

        bandwidth = librosa.feature.spectral_bandwidth(
            y=chunk,
            sr=actual_sr,
            n_fft=n_fft,
            hop_length=hop_length
        )

        bandwidth_value = float(
            np.mean(bandwidth)
        )

        rolloff = librosa.feature.spectral_rolloff(
            y=chunk,
            sr=actual_sr,
            roll_percent=0.85,
            n_fft=n_fft,
            hop_length=hop_length
        )

        rolloff_value = float(
            np.mean(rolloff)
        )

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        result = {
            "id": f"audio_{window_index:04d}",

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

            "audio_available": not is_silent,

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

        # -------------------------------------------------
        # Display result
        # -------------------------------------------------

        status = (
            "SILENT"
            if is_silent
            else "AUDIO"
        )

        print(
            f"[{start_time:.2f}s - "
            f"{end_time:.2f}s] "
            f"{status} | "
            f"RMS={rms_value:.4f} | "
            f"ZCR={zcr_value:.4f} | "
            f"Centroid={centroid_value:.1f} Hz"
        )

        # Move to next window
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
# Main
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AUDIO ANALYSIS PIPELINE")
    print("=" * 60)

    # -----------------------------------------------------
    # Step 1: Extract audio
    # -----------------------------------------------------

    audio_info = extract_audio(
        video_path=VIDEO_PATH,
        output_path=AUDIO_PATH
    )

    print("\nAudio information:")

    for key, value in audio_info.items():

        print(
            f"{key}: {value}"
        )

    # -----------------------------------------------------
    # Step 2: Analyze audio
    # -----------------------------------------------------

    analyze_audio(
        audio_path=AUDIO_PATH,
        output_path=ANALYSIS_PATH,
        sample_rate=SAMPLE_RATE,
        window_duration=WINDOW_DURATION
    )

    print("\nAudio analysis completed successfully.")