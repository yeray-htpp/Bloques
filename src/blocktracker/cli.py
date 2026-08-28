from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .capture import ScreenCapture
from .debug import DebugRenderer
from .detection import CandidateRegionDetector, DetectionConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detecta regiones que parecen historiales de ruleta")
    parser.add_argument("--image", type=Path, help="PNG/JPG existente; si se omite, captura la pantalla")
    parser.add_argument("--output", type=Path, default=Path("debug/candidatos.png"))
    parser.add_argument("--monitor", type=int, default=0, help="0=todos los monitores, 1=principal")
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--show-all", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    capture = ScreenCapture()
    image = capture.load(args.image) if args.image else capture.capture(args.monitor)
    detector = CandidateRegionDetector(DetectionConfig(min_score=args.min_score))
    candidates = detector.detect(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    debug = DebugRenderer.render(image, candidates, args.show_all)
    if not cv2.imwrite(str(args.output), debug):
        raise RuntimeError(f"No se pudo guardar: {args.output}")
    if candidates:
        best = candidates[0]
        print(f"Historial candidato: box={best.box}, score={best.score:.3f}, celdas={len(best.cells)}")
    else:
        print("No se encontraron regiones candidatas. Pruebe --min-score 0.2")
    print(f"Imagen debug: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

