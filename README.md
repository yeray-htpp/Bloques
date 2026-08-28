# BlockTracker

BlockTracker observa únicamente los píxeles visibles de Windows, localiza cuatro
resultados consecutivos de ruleta y sigue el historial hasta formar un bloque de
10 resultados. No usa APIs del casino, DOM, clics, inicio de sesión ni permisos de
administrador.

## Funcionamiento

1. El usuario escribe cuatro resultados consecutivos entre 0 y 36.
2. RapidOCR busca la secuencia en la parte inferior y, si no aparece, completa el
   análisis del resto de la pantalla.
3. Al encontrarla, fija la fila del historial y deja de escanear globalmente.
4. Un worker captura únicamente una región ampliada alrededor de esa fila.
5. `LiveHistoryTracker` compara snapshots completos y detecta inserciones por la
   izquierda o la derecha. Un cambio debe repetirse dos veces para contar como giro.
6. Los cuatro valores iniciales y los seis posteriores forman el bloque 1.
7. Un overlay Qt transparente y click-through recalcula su posición en cada lectura.
8. Después de `10/10` continúa siguiendo los diez resultados mientras sean visibles.

El overlay está excluido de las capturas internas de Windows para no contaminar el
OCR. La captura, OCR y seguimiento funcionan en un hilo distinto del renderizado.

## Ejecución

Desde CMD:

```cmd
cd C:\Users\IK\Documents\GitHub\Bloques
python -m blocktracker.search_cli 21 27 30 6 --monitor 1
```

Para cambiar el color:

```cmd
python -m blocktracker.search_cli 21 27 30 6 --monitor 1 --color "#FF00FF"
```

También puede ejecutarse sin argumentos para que solicite los cuatro números:

```cmd
blocktracker-search
```

Para que un único rectángulo contenga solamente los resultados del bloque, se deben
introducir los cuatro resultados más recientes. Si la secuencia inicial ya está en
medio del historial, los resultados posteriores aparecen separados de ella por
resultados preexistentes; en ese caso el rectángulo exterior también abarcará ese
espacio intermedio.

## Componentes principales

- `ScreenCapture`: captura pasiva mediante `mss`.
- `RapidHistoryRecognizer`: OCR local de números y coordenadas.
- `ScreenSequenceSearcher`: adquisición inicial en toda la pantalla.
- `LiveHistoryTracker`: asociación de fila y comparación de snapshots.
- `LiveTrackingWorker`: captura/OCR fuera del hilo gráfico.
- `OverlayWindow`: overlay Qt transparente, always-on-top y click-through.

## Desarrollo

```powershell
py -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
python -m pytest -q
```
