# Deepfake-Aware Multimodal Verification

An AI-powered multimodal verification system that analyzes video, audio, and speech to identify potential manipulation and generate evidence-grounded explanations.

## Current Pipeline

```text
Video
  │
  ├──> Frame Extraction
  │       │
  │       └──> Timestamped Frames
  │
  ├──> Audio Extraction
  │       │
  │       └──> Audio Signal
  │
  └──> Speech Transcription
          │
          └──> Timestamped Transcript

Timestamped Frames
        │
        ↓
Face Detection
        │
        ↓
Facial Evidence