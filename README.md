# Решение задачи "Классификация поворота текста" в рамках Data Science Bootcamp от Авито

**Решение:** смесь двух маленьких CNN (~3.2M параметров, ~13 МБ весов), каждая с Test-Time Augmentation на 180° (далее - TTA):
`p_180 = 0.2 · MobileNetV3-Small + 0.8 · PP-LCNet_x1_0_textline_ori`.

| сабмит | 1 − Brier|
|---|---|
| MobileNetV3-Small, дообученная (лучшая одиночная без TTA) | 0.8922 |
| MobileNetV3-Small + TTA | 0.9081 |
| PP-LCNet x0.25 + TTA | 0.9363 |
| PP-LCNet x1.0 + TTA | 0.9595 |
| **итог: смесь 0.2 / 0.8** | **0.9636** (расчёт; FINAL_LB) |

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
solution.ipynb                   инференс: test/images/ + weights/ -> submission.csv (самостоятельный)
submission.csv                   итоговые предсказания
weights/                         MobileNet (.onnx для инференса, .pt из обучения, логи запуска), PP-LCNet x1.0
training/                        дообучение MobileNetV3-Small
  train_mobilenet.ipynb            ноутбук обучения
  data.py, model.py, metrics.py    датасет и препроцессинг, модель, метрики
  export_onnx.py                   .pt -> .onnx: python -m training.export_onnx
```

Использованный датасет - https://disk.yandex.ru/d/cTQTsN8ocO-RNg.

## Подход

1. **Данные.** Для дообучения моделей был собран датасет на 500k изображений: 350k реальных слов из
   [TextOCR](https://textvqa.org/textocr/) и 150k синтетических строк (≈50% на русском языке). Класс 1 — то же изображение,
   повёрнутое на 180°. Разбиение train/val/test 70/15/15.
2. **Модели.** MobileNetV3-Small (ImageNet-инициализация), дообученная на этом датасете: вход 48×320, выбор эпохи по Brier на OCR-части val. PP-LCNet_x1_0_textline_ori из PaddleOCR — **без дообучения**.
3. **TTA.** Каждая модель предсказывает и по изображению, и по его повороту на 180°, а ответы
   объединяются с учётом того, что поворот меняет метку, — это убирает смещение модели к одному из классов.
4. **Составное решение.** итоговая модель - смесь MobileNet и PaddleOCR с коэффициентами 0.2 и 0.8 соответственно.


## Производительность

MobileNet (ONNX, CPU) — 2.6 мс/изображение, PP-LCNet x1.0 (GTX 1650) — 1.5 мс/изображение за проход; с TTA всё решение ~8 мс/изображение.

## Использованные open source инструменты

PyTorch / torchvision (BSD-3), PaddlePaddle / PaddleOCR — PP-LCNet textline_ori (Apache-2.0), ONNX / ONNX Runtime,
датасет TextOCR (CC BY 4.0), NumPy, pandas, OpenCV, Pillow, SciPy.
