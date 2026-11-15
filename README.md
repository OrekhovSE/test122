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

    line_pattern = re.compile(
        r"^\s*\[\s*\d+\s*/\s*\d+\s*\]\s+(.+?)\s+\|"
    )

    with log_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
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

            if (
                "images" not in [p.lower() for p in rel.parts]
                and len(rel.parts) >= 2
            ):
                rel = (
                    Path(rel.parts[0])
                    / "images"
                    / Path(*rel.parts[1:])
                )

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

    if (
        len(parts) >= 2
        and "images" not in [p.lower() for p in parts]
    ):
        alt_rel = (
            Path(parts[0])
            / "images"
            / Path(*parts[1:])
        )

        alt = dataset_root / alt_rel

        if alt.is_file():
            return alt, alt_rel

    matches = [
        p
        for p in dataset_root.rglob(rel_path.name)
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(matches) == 1:
        found = matches[0]
        return found, found.relative_to(dataset_root)

    return None, rel_path


def image_rel_to_text_rel(image_rel: Path) -> Path:
    """
    01.01/images/oneline/123.jpg
        ->
    01.01/text/oneline/123.txt
    """
    parts = list(image_rel.parts)

    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "text"
            return Path(*parts).with_suffix(".txt")

    raise ValueError(
        f"В пути изображения нет папки images: {image_rel}"
    )


def unique_pair_destinations(
    failed_root: Path,
    image_rel: Path,
    text_rel: Path,
):
    """
    Подбирает одинаковый новый stem для image и text,
    если файл в целевой папке уже существует.
    """
    image_dst = failed_root / image_rel
    text_dst = failed_root / text_rel

    if not image_dst.exists() and not text_dst.exists():
        return image_dst, text_dst

    index = 1

    while True:
        image_candidate = image_dst.with_name(
            f"{image_dst.stem}_{index}{image_dst.suffix}"
        )
        text_candidate = text_dst.with_name(
            f"{text_dst.stem}_{index}{text_dst.suffix}"
        )

        if (
            not image_candidate.exists()
            and not text_candidate.exists()
        ):
            return image_candidate, text_candidate

        index += 1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Читает лог OCR benchmark, находит FAIL и переносит "
            "фотографию + соответствующий text-файл в отдельную папку, "
            "полностью сохраняя структуру датасета."
        )
    )

    parser.add_argument(
        "--log",
        required=True,
        help="Лог benchmark (.txt/.log) или results.csv",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Корневая папка исходного датасета",
    )

    parser.add_argument(
        "--failed-dir",
        required=True,
        help="Папка назначения для FAIL-файлов",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать, что будет перенесено, не меняя файлы",
    )

    args = parser.parse_args()

    log_path = Path(args.log).resolve()
    dataset_root = Path(args.dataset).resolve()
    failed_root = Path(args.failed_dir).resolve()

    if not log_path.is_file():
        raise SystemExit(f"Лог не найден: {log_path}")

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset не найден: {dataset_root}")

    fail_entries, log_type = read_fail_entries(log_path)

    unique_entries = []
    seen = set()

    for rel in fail_entries:
        key = rel.as_posix()

        if key not in seen:
            seen.add(key)
            unique_entries.append(rel)

    if not unique_entries:
        raise SystemExit(
            "FAIL-записи в логе не найдены. "
            "Поддерживается results.csv или текстовый лог "
            "со строками, заканчивающимися на '| FAIL'."
        )

    print(f"Лог: {log_path}")
    print(f"Тип лога: {log_type}")
    print(f"Dataset: {dataset_root}")
    print(f"Папка FAIL: {failed_root}")
    print(f"Найдено FAIL: {len(unique_entries)}")
    print(
        "Режим:",
        "DRY RUN — файлы не будут перемещены"
        if args.dry_run
        else "MOVE",
    )
    print()

    moved_pairs = 0
    moved_images_only = 0
    missing_images = 0
    missing_texts = 0
    errors = 0

    total = len(unique_entries)

    for index, logged_rel in enumerate(unique_entries, 1):
        source_image, actual_image_rel = find_source_image(
            dataset_root,
            logged_rel,
        )

        if source_image is None:
            missing_images += 1
            print(
                f"{index}/{total} НЕ НАЙДЕНА ФОТОГРАФИЯ: "
                f"{logged_rel.as_posix()}"
            )
            continue

        try:
            text_rel = image_rel_to_text_rel(actual_image_rel)
        except ValueError as exc:
            errors += 1
            print(
                f"{index}/{total} ОШИБКА ПУТИ: "
                f"{actual_image_rel.as_posix()} -> {exc}"
            )
            continue

        source_text = dataset_root / text_rel
        text_exists = source_text.is_file()

        if not text_exists:
            missing_texts += 1

        image_dst, text_dst = unique_pair_destinations(
            failed_root=failed_root,
            image_rel=actual_image_rel,
            text_rel=text_rel,
        )

        if args.dry_run:
            if text_exists:
                moved_pairs += 1
                print(
                    f"{index}/{total} ПЕРЕНЕС БЫ ПАРУ:"
                )
                print(
                    f"    IMG:  {actual_image_rel.as_posix()}"
                )
                print(
                    f"    TEXT: {text_rel.as_posix()}"
                )
            else:
                moved_images_only += 1
                print(
                    f"{index}/{total} ПЕРЕНЕС БЫ ФОТО, "
                    f"TEXT НЕ НАЙДЕН:"
                )
                print(
                    f"    IMG:  {actual_image_rel.as_posix()}"
                )
                print(
                    f"    TEXT: {text_rel.as_posix()}"
                )

            continue

        try:
            image_dst.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.move(
                str(source_image),
                str(image_dst),
            )

            if text_exists:
                text_dst.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.move(
                    str(source_text),
                    str(text_dst),
                )

                moved_pairs += 1

                print(
                    f"{index}/{total} ПЕРЕНЕС:"
                )
                print(
                    f"    IMG:  {actual_image_rel.as_posix()}"
                )
                print(
                    f"    TEXT: {text_rel.as_posix()}"
                )

            else:
                moved_images_only += 1

                print(
                    f"{index}/{total} ПЕРЕНЕС ФОТО, "
                    f"TEXT НЕ НАЙДЕН:"
                )
                print(
                    f"    IMG:  {actual_image_rel.as_posix()}"
                )
                print(
                    f"    TEXT: {text_rel.as_posix()}"
                )

        except Exception as exc:
            errors += 1

            print(
                f"{index}/{total} ОШИБКА: "
                f"{actual_image_rel.as_posix()} -> {exc}"
            )

    print()
    print("================ ИТОГ ================")
    print(f"FAIL в логе:                 {total}")
    print(f"Перенесено image + text:     {moved_pairs}")
    print(f"Перенесено только image:     {moved_images_only}")
    print(f"Фото не найдено:             {missing_images}")
    print(f"Text не найден:              {missing_texts}")
    print(f"Ошибок:                      {errors}")

    if args.dry_run:
        print()
        print(
            "Это был dry-run. "
            "Для реального переноса уберите --dry-run."
        )


if __name__ == "__main__":
    main()
