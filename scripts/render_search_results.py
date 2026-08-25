"""Render candidate CSV as an offline image-review grid."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def render_results(input_path: Path, output_path: Path, *, limit: int | None = None) -> None:
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if limit is not None:
        rows = rows[:limit]
    original_query = rows[0].get("query_text", "") if rows else ""
    raw_variants = rows[0].get("query_variants", "[]") if rows else "[]"
    variants = json.loads(raw_variants) if raw_variants else []
    cards = []
    for row in rows:
        escaped = {key: html.escape(str(value), quote=True) for key, value in row.items()}
        source = Path(row.get("image_path", ""))
        uri = html.escape(source.resolve().as_uri() if source.is_file() else "", quote=True)
        cards.append(
            f'<article><img src="{uri}" alt="{escaped.get("keyframe_id", "")}" loading="lazy">'
            f"<b>#{escaped.get('rank', '')}</b> RRF={escaped.get('rrf_score', '')} "
            f"cosine={escaped.get('score', '')}<br>"
            f"{escaped.get('video_id', '')} / keyframe {escaped.get('keyframe_id', '')} / "
            f"frame {escaped.get('frame_id', '')}<br>source ranks: "
            f"{escaped.get('source_ranks', '')}</article>"
        )
    title = html.escape(input_path.stem, quote=True)
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:14px system-ui;background:#111;color:#eee}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
article{{background:#222;padding:10px}}img{{width:100%;height:180px;object-fit:contain;background:#000}}</style></head>
<body><h1>{title}</h1><h2>Original Vietnamese query</h2><p>{html.escape(original_query)}</p>
<h2>English variants</h2><ol>{"".join(f"<li>{html.escape(str(item))}</li>" for item in variants)}</ol>
<main>{"".join(cards)}</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    render_results(args.input, args.output, limit=args.limit)


if __name__ == "__main__":
    main()
