from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.search_engine import SearchEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the semantic image index.")
    parser.add_argument("--images-dir", type=Path, default=settings.images_dir, help="Directory containing downloaded images.")
    parser.add_argument("--csv-path", type=Path, default=settings.photo_csv_path, help="CSV file with the source URLs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quicker smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = SearchEngine(settings)
    count = engine.build_index(images_dir=args.images_dir, csv_path=args.csv_path, limit=args.limit)
    print(f"Indexed {count} images using {engine.embedder.name}.")


if __name__ == "__main__":
    main()
