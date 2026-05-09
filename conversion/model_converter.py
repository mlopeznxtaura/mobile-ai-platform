"""
Convert PyTorch models to CoreML, TFLite, and ONNX for mobile deployment.
SDKs: coremltools, TensorFlow Lite, ONNX, PyTorch
"""
import os
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ConversionResult:
    source_path: str
    target_format: str
    output_path: str
    model_size_mb: float
    conversion_time_sec: float
    success: bool
    error: Optional[str] = None


class MobileModelConverter:
    """
    Convert PyTorch models to mobile-optimized formats.
    Target: CoreML (iOS), TFLite (Android), ONNX (cross-platform).
    """

    def __init__(self, output_dir: str = "./mobile_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_onnx(
        self, model: nn.Module, input_shape: Tuple,
        output_path: Optional[str] = None,
        opset: int = 17, dynamic_axes: bool = True,
    ) -> ConversionResult:
        """Export PyTorch model to ONNX."""
        output_path = output_path or str(self.output_dir / "model.onnx")
        t0 = time.time()
        try:
            model.eval()
            dummy = torch.randn(*input_shape)
            dynamic = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_axes else None
            torch.onnx.export(
                model, dummy, output_path,
                opset_version=opset,
                input_names=["input"], output_names=["output"],
                dynamic_axes=dynamic,
            )
            size_mb = os.path.getsize(output_path) / 1e6
            elapsed = time.time() - t0
            print(f"[Convert] ONNX: {output_path} ({size_mb:.1f}MB) in {elapsed:.2f}s")
            return ConversionResult("pytorch", "onnx", output_path, size_mb, elapsed, True)
        except Exception as e:
            return ConversionResult("pytorch", "onnx", output_path, 0, time.time()-t0, False, str(e))

    def to_coreml(
        self, model_path: str, input_shape: Tuple,
        output_path: Optional[str] = None,
        compute_units: str = "ALL",
    ) -> ConversionResult:
        """Convert ONNX model to CoreML (.mlpackage) for iOS/macOS."""
        output_path = output_path or str(self.output_dir / "model.mlpackage")
        t0 = time.time()
        try:
            import coremltools as ct
            mlmodel = ct.converters.onnx.convert(
                model=model_path,
                minimum_ios_deployment_target="16",
                compute_units=ct.ComputeUnit[compute_units],
            )
            mlmodel.save(output_path)
            size_mb = sum(
                os.path.getsize(os.path.join(dp, f)) / 1e6
                for dp, dn, fn in os.walk(output_path)
                for f in fn
            )
            elapsed = time.time() - t0
            print(f"[Convert] CoreML: {output_path} ({size_mb:.1f}MB) in {elapsed:.2f}s")
            return ConversionResult(model_path, "coreml", output_path, size_mb, elapsed, True)
        except ImportError:
            return ConversionResult(model_path, "coreml", "", 0, 0, False, "coremltools not installed")
        except Exception as e:
            return ConversionResult(model_path, "coreml", "", 0, time.time()-t0, False, str(e))

    def to_tflite(
        self, model_path: str, input_shape: Tuple,
        output_path: Optional[str] = None,
        quantize: bool = True,
    ) -> ConversionResult:
        """Convert to TFLite for Android. Optionally applies int8 quantization."""
        output_path = output_path or str(self.output_dir / "model.tflite")
        t0 = time.time()
        try:
            import tensorflow as tf
            # Load ONNX -> TF SavedModel -> TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()
            with open(output_path, "wb") as f:
                f.write(tflite_model)
            size_mb = os.path.getsize(output_path) / 1e6
            elapsed = time.time() - t0
            print(f"[Convert] TFLite: {output_path} ({size_mb:.1f}MB) in {elapsed:.2f}s")
            return ConversionResult(model_path, "tflite", output_path, size_mb, elapsed, True)
        except Exception as e:
            return ConversionResult(model_path, "tflite", "", 0, time.time()-t0, False, str(e))

    def benchmark_onnx(
        self, onnx_path: str, input_shape: Tuple, n_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark ONNX model inference latency."""
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inp_name = sess.get_inputs()[0].name
        dummy = np.random.randn(*input_shape).astype(np.float32)

        import time
        # Warmup
        for _ in range(10):
            sess.run(None, {inp_name: dummy})

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sess.run(None, {inp_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "model_size_mb": os.path.getsize(onnx_path) / 1e6,
        }
