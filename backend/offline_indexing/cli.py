"""Command-line entry point for offline indexing and baseline search serving."""

import argparse
import json
from pathlib import Path

from .artifact_indexer import build_index
from .http_api import serve_index
from .keyframe_validator import validate_keyframes
from .object_normalizer import normalize_objects
from .video_preprocessor import preprocess_videos


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

    preprocess_command = commands.add_parser("preprocess", help="extract semantic keyframes from raw videos")
    preprocess_command.add_argument("video_directory", type=Path)
    preprocess_command.add_argument("video_ids", nargs="+")
    preprocess_command.add_argument("--output", type=Path, default=Path("data/processed/sample"))
    preprocess_command.add_argument("--scene-threshold", type=float, default=27.0)
    preprocess_command.add_argument("--dedup-threshold", type=float, default=0.04)
    preprocess_command.add_argument("--minimum-scene-frames", type=int, default=15)
    preprocess_command.add_argument("--skip-existing", action="store_true")
    preprocess_command.add_argument("--workers", type=int, default=1)

    validate_command = commands.add_parser("validate", help="validate generated keyframes and mappings")
    validate_command.add_argument("video_directory", type=Path)
    validate_command.add_argument("generated_directory", type=Path)
    validate_command.add_argument("reference_mapping_directory", type=Path)
    validate_command.add_argument("video_ids", nargs="+")

    features_command = commands.add_parser("features", help="extract SigLIP2 visual features")
    features_command.add_argument("keyframe_directory", type=Path)
    features_command.add_argument("video_ids", nargs="+")
    features_command.add_argument("--output", type=Path, default=Path("data/processed/siglip2"))
    features_command.add_argument("--batch-size", type=int, default=1)
    features_command.add_argument("--device")
    features_command.add_argument("--dense", action="store_true", help="also save final 24x24 patch features")
    features_command.add_argument("--weights", type=Path, help="local open_clip_model.safetensors file")

    objects_command = commands.add_parser("objects", help="normalize organizer object detections")
    objects_command.add_argument("object_directory", type=Path)
    objects_command.add_argument("mapping_directory", type=Path)
    objects_command.add_argument("video_ids", nargs="+")
    objects_command.add_argument("--output", type=Path, default=Path("data/processed/objects"))
    objects_command.add_argument("--minimum-score", type=float, default=0.1)

    arguments = parser.parse_args()
    if arguments.command == "build":
        print(json.dumps(build_index(arguments.dataset, arguments.output, arguments.video_prefix), indent=2))
    elif arguments.command == "serve":
        serve_index(arguments.index, arguments.host, arguments.port)
    elif arguments.command == "preprocess":
        reports = preprocess_videos(
            arguments.video_directory,
            arguments.output,
            arguments.video_ids,
            scene_threshold=arguments.scene_threshold,
            dedup_threshold=arguments.dedup_threshold,
            minimum_scene_frames=arguments.minimum_scene_frames,
            skip_existing=arguments.skip_existing,
            workers=arguments.workers,
        )
        print(json.dumps(reports, indent=2))
    elif arguments.command == "validate":
        reports = validate_keyframes(
            arguments.video_directory,
            arguments.generated_directory,
            arguments.reference_mapping_directory,
            arguments.video_ids,
        )
        print(json.dumps(reports, indent=2))
    elif arguments.command == "features":
        from .feature_extractor import extract_siglip2_features

        reports = extract_siglip2_features(
            arguments.keyframe_directory,
            arguments.output,
            arguments.video_ids,
            arguments.batch_size,
            arguments.device,
            arguments.dense,
            arguments.weights,
        )
        print(json.dumps(reports, indent=2))
    else:
        reports = normalize_objects(
            arguments.object_directory,
            arguments.mapping_directory,
            arguments.output,
            arguments.video_ids,
            arguments.minimum_score,
        )
        print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
