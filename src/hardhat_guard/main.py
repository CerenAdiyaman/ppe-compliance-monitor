"""Command line entry point.

Flags override configuration for a single run, which is what an operator wants
when trying a stream or tightening a threshold: nothing is written to `.env`,
so the next run is back to the configured behaviour.
"""

from __future__ import annotations

import argparse
import logging
import sys

from hardhat_guard.config import CAMERAS, settings

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hardhat-guard",
        description="Monitor a video stream for missing hard hats.",
        epilog="Press q or Esc to close the preview window.",
    )
    parser.add_argument(
        "--source",
        help="Webcam index, video file or stream URL (default: HG_VIDEO_SOURCE)",
    )
    parser.add_argument(
        "--camera-id",
        default="CAM_01",
        choices=sorted(CAMERAS) or None,
        help="Which entry of the camera table to monitor",
    )
    parser.add_argument("--device", help="cpu, cuda or cuda:<index>")
    parser.add_argument(
        "--conf",
        type=float,
        metavar="0.0-1.0",
        help="Minimum detection confidence",
    )
    parser.add_argument(
        "--confirmation-frames",
        type=int,
        help="Consecutive frames before a violation is confirmed",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        metavar="SECONDS",
        help="Minimum gap between records of the same violation",
    )
    parser.add_argument("--preview", action="store_true", help="Show an annotated window")
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Store identifiable faces - only with a lawful basis for doing so",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser


def apply_overrides(args: argparse.Namespace) -> None:
    """Fold command line flags into the settings singleton.

    Mutating the singleton rather than threading values through every
    constructor keeps the components reading from one place; pydantic still
    validates each assignment, so `--conf 45` is rejected here, not at the
    first frame.
    """
    if args.device is not None:
        settings.device = args.device
    if args.conf is not None:
        settings.confidence_threshold = args.conf
    if args.confirmation_frames is not None:
        settings.confirmation_frames = args.confirmation_frames
    if args.cooldown is not None:
        settings.cooldown_seconds = args.cooldown
    if args.no_anonymize:
        settings.anonymize_faces = False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=_LOG_FORMAT,
    )

    try:
        apply_overrides(args)
    except ValueError as error:
        # pydantic's message names the field and the constraint it broke.
        print(f"Invalid option: {error}", file=sys.stderr)
        return 2

    if not settings.anonymize_faces:
        logger.warning(
            "Face anonymisation is OFF. Stored snapshots will identify workers; "
            "use this only where you have a lawful basis for keeping them."
        )

    # Imported here, not at module scope: loading the detector pulls in torch,
    # which costs seconds. `--help` and a bad flag should not pay for that.
    from hardhat_guard.pipeline import build_pipeline

    try:
        pipeline = build_pipeline(
            source=args.source,
            camera_id=args.camera_id,
            preview=args.preview,
        )
    except (FileNotFoundError, RuntimeError) as error:
        # Missing weights, or a camera that will not open.
        print(f"Cannot start: {error}", file=sys.stderr)
        return 1

    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
