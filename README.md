#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR benchmark + автоматическое разбиение датасета + YOLO labels.

Ожидаемая исходная структура:

dataset/
├── 01.01/
│   ├── images/
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   └── text/
│       ├── 001.txt
│       └── 002.txt
├── 02.01/
│   ├── images/
│   └── text/
└── ...

После обработки:

dataset/
├── 01.01/
│   ├── images/
│   │   ├── oneline/
│   │   ├── twolines/
│   │   ├── vertical/
│   │   └── unknown/
│   ├── text/
│   │   ├── oneline/
│   │   ├── twolines/
│   │   ├── vertical/
│   │   └── unknown/
│   └── labels/
│       ├── oneline/
│       │   └── classes.txt
│       ├── twolines/
│       │   └── classes.txt
│       ├── vertical/
│       │   └── classes.txt
│       └── unknown/
│           └── classes.txt
└── ...

Классы YOLO:
0 = container  (берётся напрямую из нового поля container_bbox)
1 = text       (каждый bbox из ocr_debug.segmentation_overlay.items)

container_bbox имеет формат:
[x_min, y_min, x_max, y_max]
и считается координатами исходного изображения.

Для class 1:
- если bbox уже лежат внутри container_bbox, они считаются source-coordinates;
- иначе, если segmentation_overlay содержит размер layout, bbox масштабируются внутрь container_bbox;
- если размер layout отсутствует, используется fallback scaled_inferred
  по максимальным x2/y2 локальных bbox. Этот режим отмечается в results.csv.
"""

import argparse
import csv
import math
import mimetypes
import re
import shutil
import statistics
import time
from pathlib import Path

import requests

try:
    from PIL import Image
except ImportError:
    raise SystemExit(
        "Не установлен Pillow. Установите:\n"
        "pip install pillow requests"
    )


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NUMBER_TYPES = ("oneline", "twolines", "vertical")
ALL_TYPES = ("oneline", "twolines", "vertical", "unknown")

CLASS_NAMES = ("container", "text")
CLASS_CONTAINER = 0
CLASS_TEXT = 1


def normalize_code(value):
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def read_expected(txt_path: Path):
    if not txt_path or not txt_path.is_file():
        return ""

    text = txt_path.read_text(encoding="utf-8-sig", errors="replace")

    for line in text.splitlines():
        line = line.strip()
        if line:
            return normalize_code(line)

    return ""


def percentile(values, p):
    if not values:
        return None

    values = sorted(float(v) for v in values)

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * p / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def timing_stats(values):
    values = [float(v) for v in values if v is not None]

    if not values:
        return {
            "avg": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }

    return {
        "avg": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 95),
    }


def pct(num, den):
    if not den:
        return None
    return 100.0 * num / den


def fmt(value, digits=2):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def discover_batches(data_root: Path):
    """
    Ищет папки вида:
      dataset/01.01/images + dataset/01.01/text
      dataset/02.01/images + dataset/02.01/text

    Также поддерживает случай, когда data_root сам является одной такой папкой.
    """
    if (data_root / "images").is_dir() and (data_root / "text").is_dir():
        return [(data_root.name, data_root)]

    batches = []

    for folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if (folder / "images").is_dir() and (folder / "text").is_dir():
            batches.append((folder.name, folder))

    return batches


def collect_images(batch_root: Path):
    """
    Берёт:
      1) неразобранные изображения непосредственно из images/
      2) изображения из images/oneline, images/twolines, images/vertical, images/unknown

    Это позволяет безопасно работать и с частично уже разобранным датасетом.
    """
    images_root = batch_root / "images"
    result = []

    # Файлы непосредственно в images/
    for path in sorted(images_root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result.append(path)

    # Уже разложенные по типам.
    for type_name in ALL_TYPES:
        type_dir = images_root / type_name

        if not type_dir.is_dir():
            continue

        for path in sorted(type_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                result.append(path)

    return result


def folder_type_for_image(image_path: Path, batch_root: Path):
    images_root = batch_root / "images"

    try:
        rel = image_path.relative_to(images_root)
    except ValueError:
        return "unknown"

    if len(rel.parts) >= 2:
        first = rel.parts[0].lower()
        if first in ALL_TYPES:
            return first

    return "unknown"


def find_text_for_image(image_path: Path, batch_root: Path):
    """
    Для:
      images/123.jpg -> text/123.txt

    Для уже разбитого:
      images/oneline/123.jpg -> text/oneline/123.txt

    Есть fallback по имени в text/.
    """
    images_root = batch_root / "images"
    text_root = batch_root / "text"

    rel = image_path.relative_to(images_root)
    mirrored = (text_root / rel).with_suffix(".txt")

    if mirrored.is_file():
        return mirrored

    flat = text_root / f"{image_path.stem}.txt"

    if flat.is_file():
        return flat

    # Если файл уже в одной из папок типов, но text почему-то ещё плоский.
    for type_name in ALL_TYPES:
        candidate = text_root / type_name / f"{image_path.stem}.txt"
        if candidate.is_file():
            return candidate

    return None


def send_image(session, url, image_path, timeout):
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"

    started = time.perf_counter()

    with image_path.open("rb") as f:
        response = session.post(
            url,
            files={
                "files": (
                    image_path.name,
                    f,
                    mime,
                )
            },
            timeout=timeout,
        )

    request_ms = (time.perf_counter() - started) * 1000.0

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError("Endpoint вернул JSON, но верхний уровень не является объектом")

    return data, request_ms


def resolve_type(folder_type, selected_route):
    """
    Если файл уже вручную разложен по типам — сохраняем этот тип.
    Если файл лежит прямо в images/ — используем selected_route сервиса.
    """
    if folder_type in NUMBER_TYPES:
        return folder_type, "folder"

    route = str(selected_route or "").strip().lower()

    if route in NUMBER_TYPES:
        return route, "service"

    return "unknown", "service"


def parse_box(value):
    """Парсит [x_min, y_min, x_max, y_max]."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def parse_container_bbox(data):
    """
    Новый bbox контейнерной зоны.

    Поддерживаем оба варианта:
      data["container_bbox"]
      data["ocr_debug"]["container_bbox"]
    """
    box = parse_box(data.get("container_bbox"))
    if box is not None:
        return box

    ocr_debug = data.get("ocr_debug") or {}
    if isinstance(ocr_debug, dict):
        return parse_box(ocr_debug.get("container_bbox"))

    return None


def get_overlay(data):
    ocr_debug = data.get("ocr_debug") or {}
    if not isinstance(ocr_debug, dict):
        return {}

    overlay = ocr_debug.get("segmentation_overlay") or {}
    return overlay if isinstance(overlay, dict) else {}


def parse_service_boxes(data):
    """
    Возвращает bbox класса 1 из:
      ocr_debug.segmentation_overlay.items[*].box
    """
    overlay = get_overlay(data)
    items = overlay.get("items") or []

    if not isinstance(items, list):
        return []

    boxes = []

    for item in items:
        if not isinstance(item, dict):
            continue

        box = parse_box(item.get("box"))
        if box is not None:
            boxes.append(box)

    return boxes


def parse_overlay_size(data):
    """
    Пытается получить размер координатного пространства segmentation_overlay.

    Поддерживает:
      "image_size": [w, h]
      "size": [w, h]
      "width": w, "height": h
      "image_width": w, "image_height": h
    """
    overlay = get_overlay(data)

    for key in ("image_size", "size"):
        value = overlay.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                w, h = float(value[0]), float(value[1])
                if w > 0 and h > 0:
                    return w, h
            except (TypeError, ValueError):
                pass

    for wk, hk in (
        ("width", "height"),
        ("image_width", "image_height"),
    ):
        try:
            w = float(overlay.get(wk))
            h = float(overlay.get(hk))
            if w > 0 and h > 0:
                return w, h
        except (TypeError, ValueError):
            pass

    return None


def box_inside(inner, outer, tolerance=1.0):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer

    return (
        ix1 >= ox1 - tolerance
        and iy1 >= oy1 - tolerance
        and ix2 <= ox2 + tolerance
        and iy2 <= oy2 + tolerance
    )


def map_text_boxes_to_source(data, container_box, image_width, image_height):
    """
    Переводит bbox класса 1 в координаты исходного изображения.

    Режимы:
      source
        text-box уже внутри container_bbox -> считаем source coords.

      scaled_overlay
        segmentation_overlay явно содержит размер layout.

      scaled_inferred
        размер layout отсутствует; используем max(x2), max(y2)
        как размер локального пространства.
    """
    local_boxes = parse_service_boxes(data)

    if not local_boxes:
        return [], "no_text_boxes"

    if container_box is None:
        return [], "no_container"

    container_box = clamp_box(container_box, image_width, image_height)
    if container_box is None:
        return [], "invalid_container"

    if all(box_inside(box, container_box) for box in local_boxes):
        result = []
        for box in local_boxes:
            clamped = clamp_box(box, image_width, image_height)
            if clamped is not None:
                result.append(clamped)
        return result, "source"

    overlay_size = parse_overlay_size(data)

    if overlay_size is not None:
        overlay_w, overlay_h = overlay_size
        mapping_mode = "scaled_overlay"
    else:
        overlay_w = max(box[2] for box in local_boxes)
        overlay_h = max(box[3] for box in local_boxes)
        mapping_mode = "scaled_inferred"

    if overlay_w <= 0 or overlay_h <= 0:
        return [], "invalid_overlay_size"

    cx1, cy1, cx2, cy2 = container_box
    cw = cx2 - cx1
    ch = cy2 - cy1

    mapped = []

    for x1, y1, x2, y2 in local_boxes:
        sx1 = cx1 + (x1 / overlay_w) * cw
        sy1 = cy1 + (y1 / overlay_h) * ch
        sx2 = cx1 + (x2 / overlay_w) * cw
        sy2 = cy1 + (y2 / overlay_h) * ch

        source_box = clamp_box(
            (sx1, sy1, sx2, sy2),
            image_width,
            image_height,
        )

        if source_box is not None:
            mapped.append(source_box)

    return mapped, mapping_mode


def clamp_box(box, image_width, image_height):
    x1, y1, x2, y2 = box

    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def union_boxes(boxes):
    if not boxes:
        return None

    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def box_to_yolo(box, image_width, image_height):
    x1, y1, x2, y2 = box

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return x_center, y_center, width, height


def build_yolo_lines(image_path: Path, data):
    """
    Формирует YOLO label:

      class 0 = container_bbox из ответа сервера
      class 1 = segmentation_overlay.items[*].box,
                переведённые в координаты исходного изображения
    """
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Некорректный размер изображения: {image_path}")

    raw_container = parse_container_bbox(data)
    container_box = None

    if raw_container is not None:
        container_box = clamp_box(
            raw_container,
            image_width,
            image_height,
        )

    text_boxes, mapping_mode = map_text_boxes_to_source(
        data=data,
        container_box=container_box,
        image_width=image_width,
        image_height=image_height,
    )

    lines = []

    # class 0: напрямую из container_bbox.
    if container_box is not None:
        x, y, w, h = box_to_yolo(
            container_box,
            image_width,
            image_height,
        )

        lines.append(
            f"{CLASS_CONTAINER} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    # class 1: зоны текста/символов.
    for box in text_boxes:
        x, y, w, h = box_to_yolo(
            box,
            image_width,
            image_height,
        )

        lines.append(
            f"{CLASS_TEXT} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    return (
        lines,
        len(text_boxes),
        container_box is not None,
        image_width,
        image_height,
        mapping_mode,
        container_box,
    )


def write_classes_file(batch_root: Path):
    """
    Создаёт отдельный classes.txt в каждой папке типа:

      labels/oneline/classes.txt
      labels/twolines/classes.txt
      labels/vertical/classes.txt
      labels/unknown/classes.txt

    Содержимое везде одинаковое:
      0 = container
      1 = text
    """
    labels_root = batch_root / "labels"

    for type_name in ALL_TYPES:
        type_dir = labels_root / type_name
        type_dir.mkdir(parents=True, exist_ok=True)

        classes_path = type_dir / "classes.txt"
        classes_path.write_text(
            "\n".join(CLASS_NAMES) + "\n",
            encoding="utf-8",
        )


def choose_destination_stem(
    batch_root: Path,
    type_name: str,
    original_stem: str,
    image_suffix: str,
):
    """
    Не перезаписывает существующие файлы.
    Если имя занято во всех целевых папках, создаёт suffix _1, _2...
    """
    images_dir = batch_root / "images" / type_name
    text_dir = batch_root / "text" / type_name
    labels_dir = batch_root / "labels" / type_name

    index = 0

    while True:
        suffix = "" if index == 0 else f"_{index}"
        stem = f"{original_stem}{suffix}"

        image_dst = images_dir / f"{stem}{image_suffix}"
        text_dst = text_dir / f"{stem}.txt"
        label_dst = labels_dir / f"{stem}.txt"

        if (
            not image_dst.exists()
            and not text_dst.exists()
            and not label_dst.exists()
        ):
            return stem, image_dst, text_dst, label_dst

        index += 1


def same_path(a: Path, b: Path):
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a.absolute() == b.absolute()


def save_and_move(
    batch_root: Path,
    image_path: Path,
    text_path: Path | None,
    type_name: str,
    yolo_lines,
    move_files: bool,
):
    """
    Создаёт label и при необходимости перемещает image + text.
    Если image уже лежит в правильной type-папке, не переименовывает его.
    """
    images_root = batch_root / "images"
    rel_to_images = image_path.relative_to(images_root)

    # Считаем изображение уже лежащим в нужной папке ТОЛЬКО если
    # путь реально начинается с images/<type_name>/...
    #
    # Важно: файл images/123.jpg раньше тоже получал folder_type="unknown",
    # из-за чего при type_name="unknown" ошибочно считалось, что он уже
    # находится в images/unknown/. Теперь это исправлено.
    already_in_target_folder = (
        len(rel_to_images.parts) >= 2
        and rel_to_images.parts[0].lower() == type_name
    )

    if already_in_target_folder:
        stem = image_path.stem
        image_dst = image_path
        text_dst = batch_root / "text" / type_name / f"{stem}.txt"
        label_dst = batch_root / "labels" / type_name / f"{stem}.txt"
    else:
        stem, image_dst, text_dst, label_dst = choose_destination_stem(
            batch_root=batch_root,
            type_name=type_name,
            original_stem=image_path.stem,
            image_suffix=image_path.suffix,
        )

    label_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.write_text(
        ("\n".join(yolo_lines) + "\n") if yolo_lines else "",
        encoding="utf-8",
    )

    if not move_files:
        return image_dst, text_dst, label_dst

    image_dst.parent.mkdir(parents=True, exist_ok=True)

    if not same_path(image_path, image_dst):
        shutil.move(str(image_path), str(image_dst))

    if text_path and text_path.is_file():
        text_dst.parent.mkdir(parents=True, exist_ok=True)

        if not same_path(text_path, text_dst):
            shutil.move(str(text_path), str(text_dst))

    return image_dst, text_dst, label_dst


def process_one(
    session,
    url,
    batch_name,
    batch_root,
    image_path,
    timeout,
    move_files,
):
    folder_type = folder_type_for_image(image_path, batch_root)
    text_path = find_text_for_image(image_path, batch_root)

    original_rel = image_path.relative_to(batch_root).as_posix()
    expected = read_expected(text_path)

    row = {
        "batch": batch_name,
        "original_image": original_rel,
        "final_image": "",
        "final_text": "",
        "label_file": "",
        "type": "unknown",
        "type_source": "",
        "selected_route": "",
        "expected": expected,
        "predicted": "",
        "processed": False,
        "full_correct": False,
        "base10_correct": False,
        "expected_last_digit": expected[-1] if len(expected) >= 11 else "",
        "last_digit_predicted": "",
        "last_digit_status": "",
        "last_digit_confidence": "",
        "last_digit_correct": False,
        "endpoint_confidence": "",
        "check_digit": "",
        "check_digit_source": "",
        "guessed_check_digit": "",
        "is_valid_iso": "",
        "request_ms": None,
        "server_pipeline_ms": None,
        "last_digit_elapsed_ms": None,
        "server_total_ms": None,
        "text_boxes": 0,
        "container_box_created": False,
        "container_bbox": "",
        "text_box_mapping": "",
        "image_width": "",
        "image_height": "",
        "failure_reason": "",
        "ocr_reason": "",
        "ocr_status": "",
        "error": "",
    }

    try:
        data, request_ms = send_image(
            session=session,
            url=url,
            image_path=image_path,
            timeout=timeout,
        )

        row["request_ms"] = request_ms

        raw_result = data.get("result")
        not_found = (
            raw_result is None
            or str(raw_result).strip().upper() in {"", "NOT_FOUND"}
        )

        predicted = "" if not_found else normalize_code(raw_result)

        row["predicted"] = predicted
        row["processed"] = not not_found
        row["full_correct"] = bool(expected) and predicted == expected
        row["base10_correct"] = (
            len(expected) >= 10
            and len(predicted) >= 10
            and predicted[:10] == expected[:10]
        )

        row["endpoint_confidence"] = data.get("confidence", "")
        row["failure_reason"] = data.get("failure_reason", "")

        ocr_debug = data.get("ocr_debug") or {}

        if isinstance(ocr_debug, dict):
            selected_route = str(
                ocr_debug.get("selected_route", "") or ""
            ).strip().lower()

            row["selected_route"] = selected_route
            row["check_digit"] = ocr_debug.get("check_digit", "")
            row["check_digit_source"] = ocr_debug.get("check_digit_source", "")
            row["guessed_check_digit"] = ocr_debug.get("guessed_check_digit", "")
            row["is_valid_iso"] = ocr_debug.get("is_valid_iso", "")
            row["ocr_reason"] = ocr_debug.get("reason", "")
            row["ocr_status"] = ocr_debug.get("status", "")

        final_type, type_source = resolve_type(
            folder_type=folder_type,
            selected_route=row["selected_route"],
        )

        row["type"] = final_type
        row["type_source"] = type_source

        # last_digit_ocr остаётся только тестовой метрикой.
        last_digit = data.get("last_digit_ocr") or {}

        if isinstance(last_digit, dict):
            value = last_digit.get("value")
            value = "" if value is None else str(value).strip()

            row["last_digit_predicted"] = value
            row["last_digit_status"] = str(
                last_digit.get("status", "") or ""
            )
            row["last_digit_confidence"] = last_digit.get("confidence", "")

            elapsed = last_digit.get("elapsed_ms")

            if isinstance(elapsed, (int, float)):
                row["last_digit_elapsed_ms"] = float(elapsed)

            row["last_digit_correct"] = (
                bool(row["expected_last_digit"])
                and row["last_digit_status"] == "recognized"
                and value == row["expected_last_digit"]
            )

        timings = data.get("timings_ms") or {}

        if isinstance(timings, dict):
            stage_keys = (
                "read_ms",
                "decode_ms",
                "bbox_ms",
                "ocr_ms",
                "validation_ms",
            )

            stage_values = [
                float(timings[key])
                for key in stage_keys
                if isinstance(timings.get(key), (int, float))
            ]

            if stage_values:
                row["server_pipeline_ms"] = sum(stage_values)

        if row["server_pipeline_ms"] is not None:
            row["server_total_ms"] = row["server_pipeline_ms"]

            if row["last_digit_elapsed_ms"] is not None:
                row["server_total_ms"] += row["last_digit_elapsed_ms"]

        # ---- YOLO labels ----
        (
            yolo_lines,
            text_box_count,
            container_box_created,
            image_width,
            image_height,
            text_box_mapping,
            container_box,
        ) = build_yolo_lines(
            image_path=image_path,
            data=data,
        )

        row["text_boxes"] = text_box_count
        row["container_box_created"] = container_box_created
        row["container_bbox"] = (
            ",".join(f"{v:.3f}" for v in container_box)
            if container_box is not None
            else ""
        )
        row["text_box_mapping"] = text_box_mapping
        row["image_width"] = image_width
        row["image_height"] = image_height

        image_dst, text_dst, label_dst = save_and_move(
            batch_root=batch_root,
            image_path=image_path,
            text_path=text_path,
            type_name=final_type,
            yolo_lines=yolo_lines,
            move_files=move_files,
        )

        row["final_image"] = image_dst.relative_to(batch_root).as_posix()
        row["final_text"] = (
            text_dst.relative_to(batch_root).as_posix()
            if text_path
            else ""
        )
        row["label_file"] = label_dst.relative_to(batch_root).as_posix()

    except requests.RequestException as exc:
        row["error"] = f"REQUEST_ERROR: {exc}"

    except Exception as exc:
        row["error"] = f"ERROR: {type(exc).__name__}: {exc}"

        # Даже при ошибке endpoint стараемся отправить файл в unknown
        # и создать пустой label.
        try:
            row["type"] = "unknown"
            row["type_source"] = "error"

            image_dst, text_dst, label_dst = save_and_move(
                batch_root=batch_root,
                image_path=image_path,
                text_path=text_path,
                type_name="unknown",
                yolo_lines=[],
                move_files=move_files,
            )

            row["final_image"] = image_dst.relative_to(batch_root).as_posix()
            row["final_text"] = (
                text_dst.relative_to(batch_root).as_posix()
                if text_path
                else ""
            )
            row["label_file"] = label_dst.relative_to(batch_root).as_posix()

        except Exception as move_exc:
            row["error"] += f" | MOVE_ERROR: {move_exc}"

    return row


def summarize(rows, scope, batch="", number_type=""):
    total = len(rows)
    processed = sum(bool(r["processed"]) for r in rows)
    full_correct = sum(bool(r["full_correct"]) for r in rows)
    base10_correct = sum(bool(r["base10_correct"]) for r in rows)

    with_check_digit = sum(bool(r["expected_last_digit"]) for r in rows)

    recognized = sum(
        bool(r["expected_last_digit"])
        and r["last_digit_status"] == "recognized"
        and str(r["last_digit_predicted"]) != ""
        for r in rows
    )

    last_digit_correct = sum(bool(r["last_digit_correct"]) for r in rows)

    request_times = [
        r["request_ms"]
        for r in rows
        if r["request_ms"] is not None
    ]

    server_times = [
        r["server_total_ms"]
        for r in rows
        if r["server_total_ms"] is not None
    ]

    req = timing_stats(request_times)
    srv = timing_stats(server_times)

    request_sum_ms = sum(request_times)

    images_per_min = (
        60000.0 * len(request_times) / request_sum_ms
        if request_sum_ms > 0
        else None
    )

    return {
        "scope": scope,
        "batch": batch,
        "type": number_type,
        "total_images": total,
        "processed": processed,
        "processed_pct": pct(processed, total),
        "full_correct": full_correct,
        "full_accuracy_all_pct": pct(full_correct, total),
        "full_accuracy_processed_pct": pct(full_correct, processed),
        "base10_correct": base10_correct,
        "base10_accuracy_all_pct": pct(base10_correct, total),
        "with_check_digit": with_check_digit,
        "last_digit_recognized": recognized,
        "last_digit_coverage_pct": pct(recognized, with_check_digit),
        "last_digit_correct": last_digit_correct,
        "last_digit_accuracy_all_pct": pct(last_digit_correct, with_check_digit),
        "last_digit_accuracy_recognized_pct": pct(last_digit_correct, recognized),
        "request_avg_ms": req["avg"],
        "request_median_ms": req["median"],
        "request_min_ms": req["min"],
        "request_max_ms": req["max"],
        "request_p95_ms": req["p95"],
        "server_avg_ms": srv["avg"],
        "server_median_ms": srv["median"],
        "server_min_ms": srv["min"],
        "server_max_ms": srv["max"],
        "server_p95_ms": srv["p95"],
        "images_per_min": images_per_min,
        "images_with_text_boxes": sum(r["text_boxes"] > 0 for r in rows),
        "total_text_boxes": sum(int(r["text_boxes"]) for r in rows),
    }


def summary_title(summary):
    if summary["scope"] == "all":
        return "ALL"

    if summary["scope"] == "type":
        return f"ALL / {summary['type']}"

    if summary["scope"] == "batch":
        return summary["batch"]

    if summary["scope"] == "batch_type":
        return f"{summary['batch']} / {summary['type']}"

    return summary["scope"]


def format_summary_block(summary):
    title = summary_title(summary)

    return "\n".join([
        f"==================== {title.upper()} ====================",
        "",
        f"Всего фотографий:                 {summary['total_images']}",
        f"Обработано:                        {summary['processed']} / {summary['total_images']}",
        f"Успешность обработки:              {fmt(summary['processed_pct'])}%",
        "",
        f"Полный номер правильно:            {summary['full_correct']} / {summary['total_images']}",
        f"Full accuracy по всем фото:         {fmt(summary['full_accuracy_all_pct'])}%",
        f"Full accuracy среди обработанных:   {fmt(summary['full_accuracy_processed_pct'])}%",
        "",
        f"Base10 правильно:                  {summary['base10_correct']} / {summary['total_images']}",
        f"Base10 accuracy:                   {fmt(summary['base10_accuracy_all_pct'])}%",
        "",
        "Дополнительный OCR контрольной цифры:",
        f"Распознано (recognized):            {summary['last_digit_recognized']} / {summary['with_check_digit']}",
        f"Coverage:                           {fmt(summary['last_digit_coverage_pct'])}%",
        f"Правильно по всем фото:             {summary['last_digit_correct']} / {summary['with_check_digit']}",
        f"Accuracy по всем фото:              {fmt(summary['last_digit_accuracy_all_pct'])}%",
        f"Accuracy среди recognized:          {fmt(summary['last_digit_accuracy_recognized_pct'])}%",
        "",
        "Разметка:",
        f"Фото с bbox текста:                 {summary['images_with_text_boxes']} / {summary['total_images']}",
        f"Всего bbox класса 1:                {summary['total_text_boxes']}",
        "",
        "End-to-end время запроса:",
        f"Среднее:                            {fmt(summary['request_avg_ms'])} ms",
        f"Медиана:                            {fmt(summary['request_median_ms'])} ms",
        f"Минимум:                            {fmt(summary['request_min_ms'])} ms",
        f"Максимум:                           {fmt(summary['request_max_ms'])} ms",
        f"P95:                                {fmt(summary['request_p95_ms'])} ms",
        "",
        "Серверное время:",
        f"Среднее:                            {fmt(summary['server_avg_ms'])} ms",
        f"Медиана:                            {fmt(summary['server_median_ms'])} ms",
        f"Минимум:                            {fmt(summary['server_min_ms'])} ms",
        f"Максимум:                           {fmt(summary['server_max_ms'])} ms",
        f"P95:                                {fmt(summary['server_p95_ms'])} ms",
        "",
        f"Производительность:                 {fmt(summary['images_per_min'])} фото/мин",
        "",
    ])


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OCR benchmark + разбиение dataset по selected_route + "
            "создание YOLO labels с 2 классами."
        )
    )

    parser.add_argument(
        "--url",
        required=True,
        help="URL OCR endpoint",
    )

    parser.add_argument(
        "--data-root",
        default="dataset",
        help="Корень датасета: 01.01/images + 01.01/text, 02.01/...",
    )

    parser.add_argument(
        "--out",
        default="benchmark_results",
        help="Папка для results.csv / summary.csv / summary.txt",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout одного запроса, сек",
    )

    parser.add_argument(
        "--no-move",
        action="store_true",
        help=(
            "Не перемещать images/text по oneline/twolines/vertical. "
            "Labels всё равно будут создаваться."
        ),
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out).resolve()

    if not data_root.is_dir():
        raise SystemExit(f"Не найдена папка датасета: {data_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    batches = discover_batches(data_root)

    if not batches:
        raise SystemExit(
            "Не найдены папки вида DATE/images + DATE/text"
        )

    batch_images = []
    total_images = 0

    for batch_name, batch_root in batches:
        write_classes_file(batch_root)

        images = collect_images(batch_root)

        batch_images.append(
            (batch_name, batch_root, images)
        )

        total_images += len(images)

    print(f"Найдено наборов: {len(batches)}")
    print(f"Всего изображений: {total_images}")
    print(f"Endpoint: {args.url}")
    print("Перемещение файлов:", "НЕТ" if args.no_move else "ДА")
    print()
    print("Классы:")
    print("  0 = container (напрямую из container_bbox)")
    print("  1 = text (segmentation_overlay.items -> source coords)")
    print()

    session = requests.Session()
    rows = []
    counter = 0

    benchmark_started = time.perf_counter()

    for batch_name, batch_root, images in batch_images:
        print(f"\n========== {batch_name}: {len(images)} изображений ==========")

        # images заранее собраны до начала перемещений, поэтому move безопасен.
        for image_path in images:
            counter += 1

            if not image_path.is_file():
                # Например, если один и тот же файл случайно попал в список дважды.
                print(
                    f"[{counter:>5}/{total_images}] "
                    f"SKIP: файл уже отсутствует: {image_path}"
                )
                continue

            row = process_one(
                session=session,
                url=args.url,
                batch_name=batch_name,
                batch_root=batch_root,
                image_path=image_path,
                timeout=args.timeout,
                move_files=not args.no_move,
            )

            rows.append(row)

            result_text = row["predicted"] or "NOT_FOUND"

            if row["error"]:
                status = "ERROR"
            elif row["full_correct"]:
                status = "OK"
            else:
                status = "FAIL"

            print(
                f"[{counter:>5}/{total_images}] "
                f"{row['original_image']} | "
                f"type={row['type']:<8} ({row['type_source']}) | "
                f"boxes={row['text_boxes']:<2} | "
                f"GT={row['expected'] or '-':<12} | "
                f"RESULT={result_text:<12} | "
                f"{status}"
            )

    benchmark_elapsed_s = time.perf_counter() - benchmark_started

    result_fields = [
        "batch",
        "original_image",
        "final_image",
        "final_text",
        "label_file",
        "type",
        "type_source",
        "selected_route",
        "expected",
        "predicted",
        "processed",
        "full_correct",
        "base10_correct",
        "expected_last_digit",
        "last_digit_predicted",
        "last_digit_status",
        "last_digit_confidence",
        "last_digit_correct",
        "endpoint_confidence",
        "check_digit",
        "check_digit_source",
        "guessed_check_digit",
        "is_valid_iso",
        "text_boxes",
        "container_box_created",
        "container_bbox",
        "text_box_mapping",
        "image_width",
        "image_height",
        "request_ms",
        "server_pipeline_ms",
        "last_digit_elapsed_ms",
        "server_total_ms",
        "failure_reason",
        "ocr_reason",
        "ocr_status",
        "error",
    ]

    write_csv(
        out_dir / "results.csv",
        rows,
        result_fields,
    )

    summaries = []

    # Все данные.
    summaries.append(
        summarize(
            rows,
            scope="all",
        )
    )

    # Все данные по типам.
    for type_name in ALL_TYPES:
        selected = [
            row
            for row in rows
            if row["type"] == type_name
        ]

        if selected:
            summaries.append(
                summarize(
                    selected,
                    scope="type",
                    number_type=type_name,
                )
            )

    # Каждая дата и дата/type.
    for batch_name, _, _ in batch_images:
        batch_rows = [
            row
            for row in rows
            if row["batch"] == batch_name
        ]

        summaries.append(
            summarize(
                batch_rows,
                scope="batch",
                batch=batch_name,
            )
        )

        for type_name in ALL_TYPES:
            selected = [
                row
                for row in batch_rows
                if row["type"] == type_name
            ]

            if selected:
                summaries.append(
                    summarize(
                        selected,
                        scope="batch_type",
                        batch=batch_name,
                        number_type=type_name,
                    )
                )

    summary_fields = [
        "scope",
        "batch",
        "type",
        "total_images",
        "processed",
        "processed_pct",
        "full_correct",
        "full_accuracy_all_pct",
        "full_accuracy_processed_pct",
        "base10_correct",
        "base10_accuracy_all_pct",
        "with_check_digit",
        "last_digit_recognized",
        "last_digit_coverage_pct",
        "last_digit_correct",
        "last_digit_accuracy_all_pct",
        "last_digit_accuracy_recognized_pct",
        "images_with_text_boxes",
        "total_text_boxes",
        "request_avg_ms",
        "request_median_ms",
        "request_min_ms",
        "request_max_ms",
        "request_p95_ms",
        "server_avg_ms",
        "server_median_ms",
        "server_min_ms",
        "server_max_ms",
        "server_p95_ms",
        "images_per_min",
    ]

    write_csv(
        out_dir / "summary.csv",
        summaries,
        summary_fields,
    )

    report_text = "\n".join(
        format_summary_block(summary)
        for summary in summaries
    )

    actual_rate = (
        len(rows) / (benchmark_elapsed_s / 60.0)
        if benchmark_elapsed_s > 0
        else 0.0
    )

    report_text += (
        "\n==================== BENCHMARK ====================\n\n"
        f"Полное время прогона:               {benchmark_elapsed_s:.2f} сек\n"
        f"Реальная скорость всего прогона:    {actual_rate:.2f} фото/мин\n"
    )

    (out_dir / "summary.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    print()
    print(report_text)
    print()
    print("Готово.")
    print(f"Результаты: {out_dir / 'results.csv'}")
    print(f"Сводка CSV: {out_dir / 'summary.csv'}")
    print(f"Сводка TXT: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
