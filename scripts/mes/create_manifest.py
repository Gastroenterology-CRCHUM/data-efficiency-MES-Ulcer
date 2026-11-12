"""Create a patient-level manifest for MES scoring frames."""

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
    create_manifest(args, class_description="Mayo scores", dataset="mes")


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(
        description="Create MES scoring manifest",
        input_default="data/mes/processed",
        output_default="data/mes/splits",
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
