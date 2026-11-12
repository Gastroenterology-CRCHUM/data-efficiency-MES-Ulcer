"""Create a patient-level manifest for ulcer detection frames."""

from __future__ import annotations

import argparse

from src.data.create_manifest import (
    FIELDNAMES,
    create_manifest,
)
from src.data.create_manifest import (
    build_parser as _build_parser,
)

__all__ = ["FIELDNAMES", "build_parser", "main"]


def main(args: argparse.Namespace) -> None:
    create_manifest(args, class_description="0=NoUlcer, 1=Ulcer", dataset="ulcer")


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(
        description="Create ulcer detection manifest",
        input_default="data/ulcer/processed",
        output_default="data/ulcer/splits",
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
