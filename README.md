# Решение задачи "Классификация поворота текста" в рамках Data Science Bootcamp от Авито

**Решение:** solution.ipynb - смесь двух маленьких CNN (~3.2M параметров, ~13 МБ весов), каждая с Test-Time Augmentation на 180° (далее - TTA):
`p_180 = 0.2 · MobileNetV3-Small + 0.8 · PP-LCNet_x1_0_textline_ori`.

| сабмит | 1 − Brier|
|---|---|
| MobileNetV3-Small, дообученная (лучшая одиночная без TTA) | 0.8922 |
| MobileNetV3-Small + TTA | 0.9081 |
| PP-LCNet x0.25 + TTA | 0.9363 |
| PP-LCNet x1.0 + TTA | 0.9595 |
| **итог: смесь 0.2 / 0.8** | **0.9636** |

## Начало работы

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python paddlepaddle-gpu==3.3.1 \
    --index https://www.paddlepaddle.org.cn/packages/stable/cu126/ --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace solution.ipynb
```

Для обучения — отдельное окружение `requirements-train.txt` (CUDA-библиотеки torch и paddle в одном venv конфликтуют;
поэтому в инференсе MobileNet работает через ONNX Runtime).

## Структура

```
solution.ipynb                   основной ноутбук
submission.csv                   итоговые предсказания
weights/                         MobileNet (.onnx для инференса, .pt из обучения, логи запуска), PP-LCNet x1.0
training/                        дообучение MobileNetV3-Small
  train_mobilenet.ipynb            ноутбук обучения
  data.py, model.py, metrics.py    датасет и препроцессинг, модель, метрики
  export_onnx.py                   .pt -> .onnx: python -m training.export_onnx
```

Использованный датасет - https://disk.yandex.ru/d/cTQTsN8ocO-RNg.

## Производительность

MobileNet (ONNX, CPU) — 2.6 мс/изображение, PP-LCNet x1.0 (GTX 1650) — 1.5 мс/изображение за проход; с TTA всё решение ~8 мс/изображение.

## Использованные open source инструменты

PyTorch / torchvision (BSD-3), PaddlePaddle / PaddleOCR — PP-LCNet textline_ori (Apache-2.0), ONNX / ONNX Runtime,
датасет TextOCR (CC BY 4.0), NumPy, pandas, OpenCV, Pillow, SciPy.
