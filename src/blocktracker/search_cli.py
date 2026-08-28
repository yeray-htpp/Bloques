from __future__ import annotations

import argparse
import re
import time

from .capture import ScreenCapture
from .live_session import LiveTrackingWorker
from .overlay import OverlayWindow, enable_dpi_awareness
from .search import ScreenSequenceSearcher, translate_box
from .tracking import LiveHistoryTracker


def _roulette_number(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("debe ser un entero") from error
    if not 0 <= number <= 36:
        raise argparse.ArgumentTypeError("debe estar entre 0 y 36")
    return number


def _html_color(value: str) -> str:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise argparse.ArgumentTypeError("use un color hexadecimal, por ejemplo #00E5FF")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Busca cuatro resultados consecutivos visibles en pantalla")
    parser.add_argument("numbers", nargs="*", type=_roulette_number, help="cuatro números entre 0 y 36")
    parser.add_argument("--monitor", type=int, default=0, help="0=todos los monitores, 1=principal")
    parser.add_argument("--interval", type=float, default=0.75, help="segundos entre análisis")
    parser.add_argument("--color", type=_html_color, default="#00E5FF", help="color del borde")
    return parser


def main() -> int:
    enable_dpi_awareness()
    args = build_parser().parse_args()
    numbers = args.numbers
    if not numbers:
        raw = input("Escribe los 4 números consecutivos (ej. 21 27 30 6): ")
        numbers = [_roulette_number(value) for value in raw.replace(",", " ").split()]
    if len(numbers) != 4:
        raise SystemExit("Error: se necesitan exactamente cuatro números.")

    sequence = tuple(numbers)
    capture = ScreenCapture()
    searcher = ScreenSequenceSearcher()
    print(f"Buscando {sequence} en pantalla. Pulsa Ctrl+C para cancelar.")
    scans = 0
    try:
        while True:
            image, origin = capture.capture_with_origin(args.monitor)
            result = searcher.search(image, sequence)
            scans += 1
            if result:
                screen_box = translate_box(result.match.enclosing_box, origin)
                print("\nHISTORIAL ENCONTRADO — BLOQUE 1: 4/10")
                print(f"Región fijada: x={screen_box[0]}, y={screen_box[1]}, "
                      f"ancho={screen_box[2]}, alto={screen_box[3]}")
                print("Celdas:")
                for cell in result.match.cells:
                    print(f"  {cell.value}: {translate_box(cell.box, origin)} confianza={cell.confidence:.2f}")
                print("Overlay activo. El programa seguirá abierto; pulsa Ctrl+C para cerrarlo.")

                tracker = LiveHistoryTracker(result)
                overlay = OverlayWindow(screen_box, tracker.current_count, tracker.target_count,
                                        active_color=args.color)
                worker = LiveTrackingWorker(capture, searcher, tracker, args.monitor, args.interval)
                worker.start()
                complete_announced = False

                def render_latest() -> None:
                    nonlocal complete_announced
                    event = worker.latest()
                    if event:
                        if event.error:
                            print(f"\rError temporal: {event.error}; reintentando...        ", end="", flush=True)
                        elif event.update:
                            update = event.update
                            if event.screen_box:
                                overlay.update(event.screen_box, update.count, tracker.target_count)
                            messages = {
                                "lost": "OCR perdió momentáneamente la fila; reintentando",
                                "uncertain": "lectura incierta; esperando confirmación",
                                "confirming": "nuevo giro detectado; confirmando",
                                "new_result": f"nuevo resultado {update.new_result} agregado",
                                "block_not_visible": "bloque completo ya no está totalmente visible",
                            }
                            detail = messages.get(update.status, "siguiendo y moviendo el bloque")
                            print(f"\rBLOQUE 1: {update.count}/{tracker.target_count} — {detail}        ",
                                  end="", flush=True)
                            if update.count >= tracker.target_count and not complete_announced:
                                print("\nBLOQUE 1 COMPLETO — seguirá moviéndose mientras permanezca visible.")
                                complete_announced = True
                    overlay.after(50, render_latest)

                overlay.after(50, render_latest)
                try:
                    overlay.run()
                except KeyboardInterrupt:
                    print("\nBlockTracker cerrado.")
                finally:
                    worker.stop()
                return 0

            print(f"\rEscaneos: {scans} — buscando...", end="", flush=True)
            time.sleep(max(0.15, args.interval))
    except KeyboardInterrupt:
        print("\nBúsqueda cancelada.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
