"""
mobile-ai-platform — Entry Point
"""
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Mobile AI Platform")
    p.add_argument("--mode", required=True, choices=["convert", "benchmark", "serve", "vision"])
    p.add_argument("--model", default="mobilenet")
    p.add_argument("--target", default="onnx", choices=["onnx", "coreml", "tflite"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--task", default="face_detection")
    p.add_argument("--source", default="camera")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--output", default="./mobile_models")
    return p.parse_args()

def main():
    args = parse_args()
    print("=" * 55)
    print("  Mobile AI Platform")
    print(f"  Mode: {args.mode.upper()} | Target: {args.target}")
    print("=" * 55)

    if args.mode == "convert":
        import torch
        import torchvision.models as models
        from conversion.model_converter import MobileModelConverter
        model = models.mobilenet_v3_small(pretrained=False)
        converter = MobileModelConverter(output_dir=args.output)
        result = converter.to_onnx(model, (1, 3, 224, 224))
        print(f"\nConversion: {'OK' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Output: {result.output_path} ({result.model_size_mb:.1f}MB)")

    elif args.mode == "benchmark":
        import torch, torchvision.models as models
        from conversion.model_converter import MobileModelConverter
        import tempfile
        model = models.mobilenet_v3_small(pretrained=False)
        converter = MobileModelConverter()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = f"{tmp}/model.onnx"
            converter.to_onnx(model, (1, 3, 224, 224), output_path=onnx_path)
            stats = converter.benchmark_onnx(onnx_path, (1, 3, 224, 224))
            print(f"\nONNX Benchmark:")
            for k, v in stats.items():
                print(f"  {k}: {v:.2f}")

    elif args.mode == "vision":
        from vision.mediapipe_vision import MediaPipeVision, MP_AVAILABLE
        if not MP_AVAILABLE:
            print("Install mediapipe: pip install mediapipe")
            return
        vision = MediaPipeVision()
        if args.source == "camera":
            vision.run_camera(task=args.task)
        else:
            import cv2, numpy as np
            frame = cv2.imread(args.source)
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = vision.detect_faces(frame)
            print(f"Detected {len(result.detections)} faces | {result.inference_ms:.1f}ms")

    elif args.mode == "serve":
        import uvicorn
        from fastapi import FastAPI
        app = FastAPI(title="Mobile AI Backend")
        @app.get("/health")
        def health():
            return {"status": "ok", "platform": "mobile-ai"}
        uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
