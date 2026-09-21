"""نسخهٔ هشت: وزن‌های ذخیره‌شده را برای تولید پیاپی نشانه‌ها مصرف می‌کنیم."""

from mini_gpt.generate import build_parser, main


if __name__ == "__main__":
    from ..console import configure_console
    configure_console()
    main(build_parser().parse_args())
