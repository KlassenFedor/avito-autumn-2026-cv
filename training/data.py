"""Чтение и проверка разметки, препроцессинг кропов; работает и без PyTorch."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

SPLITS = ("train", "val", "test", "calibration", "holdout")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def read_annotations(csv_path, root, check_images=False):
    root = Path(root).resolve()
    rows, identifiers, paths = [], set(), set()
    owners = {"group_id": {}, "source_image_id": {}, "source_annotation_id": {}}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"image_id", "path", "group_id", "split", "label"}
        if not needed <= set(reader.fieldnames or []):
            raise ValueError(f"Нет колонок: {needed - set(reader.fieldnames or [])}")
        for line, raw in enumerate(reader, 2):
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError(f"Строка CSV {line}: неверное число колонок; текст с запятыми нужно брать в кавычки")
            row = {k: v.strip() for k, v in raw.items()}
            row["split"] = {"validation": "val", "calib": "calibration"}.get(row["split"], row["split"])
            if row["split"] not in SPLITS:
                raise ValueError(f"Строка {line}: неизвестный split {row['split']!r}")
            if not row["image_id"] or not row["group_id"] or not row["path"]:
                raise ValueError(f"Строка {line}: пустой image_id, path или group_id")
            label = float(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Строка {line}: label должен быть 0 или 1")
            row["label"] = int(label)
            if row.get("angle") not in (None, "", "**"):
                if float(row["angle"]) != 180 * label:
                    raise ValueError(f"Строка {line}: angle не соответствует label")
            row["source"] = row.get("source") or row.get("kind") or "unknown"
            path = (root / row["path"]).resolve()
            if row["image_id"] in identifiers or str(path) in paths:
                raise ValueError(f"Строка {line}: повтор image_id или пути к картинке")
            if not path.is_file():
                raise FileNotFoundError(f"Строка {line}: {path}")
            if check_images:
                with Image.open(path) as image:
                    size = image.size
                    image.verify()
                for name, actual in zip(("width", "height"), size):
                    if row.get(name) and int(row[name]) != actual:
                        raise ValueError(f"Строка {line}: {name} не совпадает с размером картинки")
            identifiers.add(row["image_id"])
            paths.add(str(path))
            for field in owners:
                value = row.get(field, "")
                if value.lower() in ("", "**", "nan", "none", "null"):
                    continue
                key = value if field == "group_id" else (row["source"], value)
                previous = owners[field].setdefault(key, row["split"])
                if previous != row["split"]:
                    raise ValueError(f"Утечка данных: {field}={value!r} есть и в {previous}, и в {row['split']}")
            row["absolute_path"] = str(path)
            rows.append(row)
    if not rows:
        raise ValueError("Пустой CSV")
    counts = Counter((r["split"], r["source"], r["label"]) for r in rows)
    summary = [{"split": s, "source": src, "label": y, "count": n}
               for (s, src, y), n in sorted(counts.items())]
    return rows, summary


def letterbox(image, height, width):
    """Сохраняет все пиксели и пропорции: без центрального кропа и автоповорота."""
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = image.resize((w, h), Image.Resampling.BILINEAR)
    # симметричные поля: положение паддинга не подсказывает ориентацию
    canvas = Image.new("RGB", (width, height), (127, 127, 127))
    canvas.paste(resized, ((width - w) // 2, (height - h) // 2))
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray((arr - MEAN) / STD)


class CropDataset:
    def __init__(self, rows, height=48, width=192, training=False, rotate_prob=0.5, seed=42):
        self.rows = rows
        self.height, self.width = height, width
        self.training, self.rotate_prob, self.seed = training, rotate_prob, seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["absolute_path"]) as source:
            image = source.convert("RGB")
        label = row.get("label", 0)
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            if rng.random() < self.rotate_prob:
                image = image.transpose(Image.Transpose.ROTATE_180)
                label = 1 - label  # в исходном CSV уже есть оба класса
            image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.9, 1.1)))
            image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.9, 1.1)))
        return letterbox(image, self.height, self.width), np.float32(label)
