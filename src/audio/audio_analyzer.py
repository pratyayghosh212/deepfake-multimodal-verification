import subprocess
from pathlib import Path


def extract_audio(
    video_path: str,
    output_path: str
):
    """
    Extract the audio stream from a video
    and save it as a WAV file.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),

        # Convert to mono
        "-ac",
        "1",

        # 16 kHz sampling rate
        "-ar",
        "16000",

        # PCM WAV
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

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg error:\n{result.stderr}"
        )

    print(f"Audio saved to: {output_path}")


if __name__ == "__main__":

    extract_audio(
        video_path="data/videos/test.mp4",
        output_path="data/audio/audio.wav"
    )