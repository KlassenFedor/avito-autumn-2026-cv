"""Экспорт дообученной MobileNetV3-Small из PyTorch-чекпоинта в ONNX (нужны torch и onnx).

  python -m training.export_onnx --checkpoint weights/mobilenet_v3_small_orientation.pt \
      --out weights/mobilenet_v3_small_orientation.onnx
"""
import argparse

import numpy as np
import torch

from .model import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="weights/mobilenet_v3_small_orientation.pt")
    ap.add_argument("--out", default="weights/mobilenet_v3_small_orientation.onnx")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint, "cpu")
    x = torch.randn(4, 3, ckpt["height"], ckpt["width"])
    torch.onnx.export(model, x[:1], args.out, input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17, dynamo=False)

    # проверка: ONNX и PyTorch дают одинаковые логиты
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"saved {args.out} (onnxruntime не установлен — сверка пропущена)")
        return
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        ref = model(x).numpy()
    diff = np.abs(sess.run(None, {"image": x.numpy()})[0] - ref).max()
    print(f"saved {args.out}; max |logit diff| ONNX vs torch = {diff:.2e}")


if __name__ == "__main__":
    main()
