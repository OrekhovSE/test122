#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def normalize_rel_path(value: str) -> str:
    value = str(value or "").strip().strip('"').strip("'")
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.lstrip("/")


def read_fail_from_csv(log_path: Path):
    fails = []

    with log_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        fields = {name.lower(): name for name in reader.fieldnames}

        for row in reader:
            if "status" in fields:
                is_fail = str(row.get(fields["status"], "")).strip().upper() == "FAIL"
            elif "full_correct" in fields:
                value = str(row.get(fields["full_correct"], "")).strip().lower()
                is_fail = value not in {"true", "1", "yes", "y"}
            else:
                is_fail = False

            if not is_fail:
                continue

            batch = normalize_rel_path(row.get(fields["batch"], "")) if "batch" in fields else ""

            if "filename" in fields:
                filename = normalize_rel_path(row.get(fields["filename"], ""))
            elif "file" in fields:
                filename = normalize_rel_path(row.get(fields["file"], ""))
            elif "path" in fields:
                filename = normalize_rel_path(row.get(fields["path"], ""))
            else:
                filename = ""

            if not filename:
                continue

            if batch:
                rel = Path(batch) / "images" / Path(filename)
            else:
                rel = Path(filename)

            fails.append(rel)

    return fails


def read_fail_from_text(log_path: Path):
    fails = []
    line_pattern = re.compile(r"^\s*\[\s*\d+\s*/\s*\d+\s*\]\s+(.+?)\s+\|")

    with log_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            parts = [part.strip() for part in line.split("|")]

            if not parts or parts[-1].upper() != "FAIL":
                continue

            match = line_pattern.search(line)
            if not match:
                continue

            rel_text = normalize_rel_path(match.group(1))
            if not rel_text:
                continue

            rel = Path(rel_text)
            parts_lower = [p.lower() for p in rel.parts]

            if "images" not in parts_lower and len(rel.parts) >= 2:
                rel = Path(rel.parts[0]) / "images" / Path(*rel.parts[1:])

            fails.append(rel)

    return fails


def read_fail_entries(log_path: Path):
    if log_path.suffix.lower() == ".csv":
        fails = read_fail_from_csv(log_path)
        if fails:
            return fails, "csv"

    return read_fail_from_text(log_path), "text"


def find_source_image(dataset_root: Path, rel_path: Path):
    candidate = dataset_root / rel_path
    if candidate.is_file():
        return candidate, rel_path

    parts = rel_path.parts

    if len(parts) >= 2 and "images" not in [p.lower() for p in parts]:
        alt_rel = Path(parts[0]) / "images" / Path(*parts[1:])
        alt = dataset_root / alt_rel
        if alt.is_file():
            return alt, alt_rel

    matches = [
        p for p in dataset_root.rglob(rel_path.name)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(matches) == 1:
        found = matches[0]
        return found, found.relative_to(dataset_root)

    return None, rel_path


def unique_destination(dst: Path):
    if not dst.exists():
        return dst

    index = 1
    while True:
        candidate = dst.with_name(f"{dst.stem}_{index}{dst.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Читает лог OCR benchmark, находит FAIL и переносит "
            "не прошедшие проверку фотографии в отдельную папку, "
            "сохраняя структуру датасета."
        )
    )

    parser.add_argument("--log", required=True, help="Лог benchmark (.txt/.log) или results.csv")
    parser.add_argument("--dataset", required=True, help="Корневая папка исходного датасета")
    parser.add_argument("--failed-dir", required=True, help="Папка для FAIL-фотографий")
    parser.add_argument("--dry-run", action="store_true", help="Показать переносы без изменения файлов")
    args = parser.parse_args()

    log_path = Path(args.log).resolve()
    dataset_root = Path(args.dataset).resolve()
