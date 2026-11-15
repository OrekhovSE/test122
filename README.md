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
