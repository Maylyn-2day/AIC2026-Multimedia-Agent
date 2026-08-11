"""Command-line entry point for offline indexing and baseline search serving."""

import argparse
import json
from pathlib import Path

from .artifact_indexer import build_index
from .http_api import serve_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build_command = commands.add_parser("build", help="normalize organizer artifacts")
    build_command.add_argument("dataset", type=Path)
    build_command.add_argument("--output", type=Path, default=Path("data/offline_index"))
    build_command.add_argument("--video-prefix", help="only index video IDs starting with this value")

    serve_command = commands.add_parser("serve", help="serve the baseline search API")
    serve_command.add_argument("--index", type=Path, default=Path("data/offline_index"))
    serve_command.add_argument("--host", default="127.0.0.1")
    serve_command.add_argument("--port", type=int, default=8000)

    arguments = parser.parse_args()
    if arguments.command == "build":
        print(json.dumps(build_index(arguments.dataset, arguments.output, arguments.video_prefix), indent=2))
    else:
        serve_index(arguments.index, arguments.host, arguments.port)


if __name__ == "__main__":
    main()
