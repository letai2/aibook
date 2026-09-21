"""نسخهٔ هفت: معماری v6 ثابت است؛ چرخهٔ آموزش/ارزیابی و ذخیره تکمیل می‌شود."""

from mini_gpt.train import build_parser, train


if __name__ == "__main__":
    from ..console import configure_console
    configure_console()
    train(build_parser().parse_args())
