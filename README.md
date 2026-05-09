# Mobile AI Platform

Cluster 23 of the NextAura 500 SDKs / 25 Clusters project.

On-device AI for iOS and Android. CoreML, Apple MLX, ARKit, TFLite, ARCore, MediaPipe — all running locally on the device with no cloud required.

## Architecture

- Apple MLX for M-chip neural engine acceleration (iOS/macOS)
- CoreML for iOS model deployment
- TensorFlow Lite for Android on-device inference
- ARKit (iOS) + ARCore (Android) for augmented reality
- MediaPipe for real-time vision tasks (face, pose, hands)
- ONNX Runtime Mobile for cross-platform model serving
- React Native + Expo for cross-platform UI
- ElevenLabs + Deepgram + Whisper for voice AI
- Vision Camera SDK for high-performance camera access

## SDKs Used

CoreML, Metal, Apple MLX, ARKit, SwiftUI, TFLite, Android SDK, Jetpack Compose, ARCore, MediaPipe, ONNX Runtime, React Native, Expo, Vision Camera SDK, ElevenLabs, Deepgram, Whisper, FastAPI, Redis, Prometheus Client

## Quickstart

```bash
pip install -r requirements.txt  # Python backend

# Convert model to mobile formats
python main.py --mode convert --model ./model.pt --target coreml

# Run mobile inference benchmark
python main.py --mode benchmark --model mobilenet --device cpu

# Start mobile backend API
python main.py --mode serve --port 8000

# Test vision pipeline
python main.py --mode vision --task face_detection --source camera
```
