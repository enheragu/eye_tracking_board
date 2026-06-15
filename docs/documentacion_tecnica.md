# Documentación técnica — *eyes_board_color*

> **▶ Lee primero la [guía de procesamiento](guia_procesamiento.md).** Este es el documento
> de **segunda lectura**, para desarrolladores o para entender el detalle del proceso. **Da
> por leída** la guía: usa sus conceptos (*trial*, fases, marcas, CSV, máquina de estados a
> nivel usuario) **sin repetirlos** — cuando aparece uno, se enlaza a la guía (`[guía §X]`) y
> aquí se entra en el **cómo** (algoritmos, implementación, hallazgos medidos). No necesitas
> tener la guía delante en paralelo; basta haberla leído.

Cubre la arquitectura, el flujo por fotograma, los algoritmos de cada módulo, la máquina de
estados en detalle, el contrato de entradas/salidas y los hallazgos de ingeniería medidos.
Los **números concretos** por participante están en el informe HTML y los CSV combinados.

> Convención: todo lo que aquí se afirma como **medido** se ha comprobado con un script
> sobre fotogramas reales del vídeo; lo que sea hipótesis se marca como tal.

**Índice**

1. [Arquitectura y flujo por fotograma](#1-arquitectura-y-flujo-por-fotograma)
2. [Entradas y salidas (contrato técnico)](#2-entradas-y-salidas-contrato-técnico)
3. [Corrección de imagen y detección de ArUcos](#3-corrección-de-imagen-y-detección-de-arucos)
4. [Localización del tablero (homografía + rejilla)](#4-localización-del-tablero-homografía--rejilla)
5. [Detección del panel de estímulo](#5-detección-del-panel-de-estímulo-panelhandler)
6. [Emparejamiento mirada–vídeo](#6-emparejamiento-mirada-vídeo-eyedatahandler)
7. [Máquina de estados en detalle](#7-máquina-de-estados-en-detalle)
8. [Derivación de marcas, fases y línea de tiempo](#8-derivación-de-marcas-fases-y-línea-de-tiempo-store_results)
9. [Detector de toque (best-effort)](#9-detector-de-toque-best-effort)
10. [Marcas motoras y la ambigüedad del contorno](#10-marcas-motoras-y-la-ambigüedad-del-contorno)
11. [Frecuencia de muestreo del *gaze*](#11-frecuencia-de-muestreo-del-gaze)
12. [Limitaciones del equipo de medida (mecanismo)](#12-limitaciones-del-equipo-de-medida-mecanismo)
13. [Parámetros clave](#13-parámetros-clave)

---

## 1. Arquitectura y flujo por fotograma

Una participante = un vídeo de *World* (**1280×720**, ~30 fps) + datos de *gaze* de Pupil Labs. El
punto de entrada es `src/process_video.py`, que instancia los *handlers* y recorre el vídeo
de forma **secuencial** (decodificación en streaming, no *seeks* por fotograma — de ahí la
mejora de ~6,5× respecto a 0.x). Para cada fotograma `capture_idx`:

1. **Desdistorsión** (`DistortionHandler.undistortImage`) con la calibración de cámara.
2. **Detección de ArUcos** una sola vez (`detectAllArucos`) sobre la imagen **original
   (distorsionada)** —para no perder los marcadores de los bordes que el undistort recorta—,
   desdistorsionando luego sus esquinas (ver apartado 3); se filtran los ids que no pertenecen
   al tablero ni a los paneles (`filterValidArucos`). La imagen desdistorsionada y las
   detecciones se **comparten** entre todos los handlers (una sola vez por fotograma).
3. **Emparejamiento de *gaze*** del fotograma (`EyeDataHandler.step`, apartado 6).
4. **Despacho a la máquina de estados** (`StateMachine.step`), que delega en `BoardHandler`,
   `PanelHandler` y `EyeDataHandler` según la fase.

### El bucle de despacho (`StateMachine.step`)

El corazón del procesamiento es un bucle que **reejecuta el estado actual mientras el estado
cambie dentro del mismo fotograma**:

```python
previous_state = None
while self.current_state != previous_state:
    previous_state = self.current_state
    self.state_info[self.current_state]['callback'](undistorted_image, capture_idx, gaze_list)
    self.frame_speed_multiplier = self.state_info[self.current_state]['frame_mult']
    if self.current_state != previous_state:
        self._recordTransition(previous_state, self.current_state, capture_idx)
```

Esto permite **encadenar varias transiciones en un mismo fotograma** (p. ej.
`test_finish_execution → init → get_test_name` cuando el panel siguiente ya está visible al
cerrar) sin perder pasos. Cada transición se registra con su *frame* exacto y el block/trial
activo (apartado 7 y 8).

- **Multiplicador de velocidad** (`frame_speed_multiplier`): cada estado declara cuántos
  fotogramas puede saltar el lector. Los estados de espera (`init`) avanzan más rápido; los
  de búsqueda y fase motora procesan fotograma a fotograma.
- **E/S de vídeo con hilos** (`ThreadVideoStream`): la lectura (y, en modo visualización, la
  escritura del `debug_<id>.mp4`) corre en hilos aparte para no bloquear el procesado.
- **Hilos de OpenCV**: limitados por `EEHA_CV_THREADS` (lo fija `run_all.py`) para no
  sobre-suscribir la máquina cuando se procesan varios participantes en paralelo. Con
  `cv_threads=1` el resultado es **determinista** entre ejecuciones.

### Mapa de módulos (`src/core/`)

| Módulo | Responsabilidad |
|---|---|
| `process_video.py` (raíz `src/`) | Punto de entrada: bucle de vídeo, instancia handlers, vuelca resultados. |
| `DistortionHandler` | Desdistorsión de imagen y de coordenadas (con homografía opcional). |
| `ArucoBoardHandler` | Detección/casado de ArUcos, pose, rotación 180°, homografía a vista cenital; corrección de color. |
| `BoardHandler` | Localiza y endereza el tablero, segmenta la rejilla, mide oclusión y toque. |
| `PanelHandler` | Detecta el panel de estímulo (identidad color/forma) y su polígono. |
| `EyeDataHandler` | Carga *gaze*/*fixations* `.pldata` y los empareja a *frames* de *World*. |
| `StateMachineHandler` | Máquina de estados del experimento; deriva marcas y vuelca CSV/PKL. |
| `ThreadVideoStream` | Lectura/escritura de vídeo con hilos. |
| `utils` | Máscaras de color (`getMaskHue`), proyecciones, logging por niveles. |

### Por qué este enfoque (ArUco + homografía) y no detección por aprendizaje

El problema concreto es **proyectar la mirada a la casilla exacta** de un tablero conocido y
detectar un **toque breve** sobre la pieza objetivo, todo desde el vídeo egocéntrico de las
gafas. Para eso se eligió localización por **marcadores fiduciales (ArUco) + homografía**:

- El tablero es un objeto **fijo y diseñado**; los marcadores fiduciales dan una geometría
  **conocida y calibrada**, de la que sale una homografía precisa que mapea cada muestra de
  *gaze* a su celda con error de píxeles. Es **determinista, transparente, reproducible** y
  ligero (sin GPU ni modelos).
- Para el **toque** se usa detección de **cambio/oclusión** sobre la casilla, no seguimiento de
  mano: en la vista egocéntrica la mano suele entrar **parcialmente ocluida por el borde** del
  tablero o salirse del encuadre al alcanzar, y el cambio de imagen funciona con manga de
  cualquier color y sin necesidad de ver la mano entera.

Se valoraron enfoques basados en **modelos aprendidos** (p. ej. **MediaPipe**, excelente para
seguimiento de manos/pose/objetos en general). Se descartaron porque resuelven un problema
**distinto**: no aportan la **geometría métrica por celda** del tablero —que es justo lo que
necesita la proyección de la mirada— y añadirían una dependencia más **pesada, menos
transparente y no determinista** sin mejorar la medida central (*gaze* → celda). No es que un
método sea "mejor"; es que la combinación fiducial + homografía encaja **directamente** con los
requisitos de esta tarea (precisión por celda, calibración y reproducibilidad).

---

## 2. Entradas y salidas (contrato técnico)

### Entradas

Por participante (`<data_root>/<id>/`): `world.mp4`, `world_timestamps.npy`,
`gaze.pldata` + `gaze_timestamps.npy` (o `fixations.*`), `camera_calib.json`. Configuración
común (`cfg/`): `game_config.yaml` (tablero 8×5: color/forma/ficha por casilla),
`game_aruco_board.yaml` (layout de los ArUcos del tablero), `default_trials_config.yaml`
(secuencia esperada; excepciones por participante en `cfg/trials_config_exceptions/`),
`sample_shape_cfg/` (un YAML por panel de estímulo). Detalle de uso en la [guía §2].

### Estructura de datos interna (`data_store`)

`store_results` serializa un único diccionario a `data_<id>.pkl` (carga rápida) y
`data_<id>.yaml` (legible):

```
sw_version, video_fps, gaze_sampling_rate, participant_id,
frames_info{estado: nº frames}, fixations_info{estado: nº muestras},
trials_data{(block,trial): {nombre_trial: board_metrics}},
state_transitions[ {frame, block, trial, trial_name, from_state, to_state} ]
```

Cada `board_metrics` de un *trial* contiene: `init_capture`, `end_capture`,
`early_init_capture`, `motor_onset_capture`, `target_touch_capture`, `hand_exit_capture`
(+ `hand_exit_source`), `target_cord` ([fila,col]), `target_norm_coord`, `status`,
`touch_diag`, y `sequence` (lista de muestras `{color, shape, slot, frame, phase,
board_coord, norm_board_coord}`).

### Salidas (ficheros)

Por participante, desde `store_results`: `trials_data_<id>.csv` (resumen por trial),
`trials_data_<id>_sequence.csv` (una fila por muestra de *gaze*),
`trials_data_<id>_transitions.csv` (línea de tiempo estados+marcas), `data_<id>.{pkl,yaml}`,
`result_log_<id>.txt`, y `debug_<id>.mp4` (solo con visualización). Por lote, desde
`src/tools/process_outputs.py`: `combined_{trials,sequence,transitions}_<topic>.csv`
(apilados con columna `participant`), `target_geometry.csv` (referencia por casilla),
`informe_comparativa.html` + `_frequencies.csv`. El contenido columna a columna está en la
[guía §5–§6]; aquí basta el contrato: **los CSV se derivan del `data_store`**, que es la
fuente de verdad (el PKL conserva campos que los CSV resumen no exponen).

---

## 3. Corrección de imagen y detección de ArUcos

### Desdistorsión (`DistortionHandler`)

De `camera_calib.json` se leen `camera_matrix` y `distortion_coefficients`. En el arranque se
precomputan los mapas de remapeo (`initUndistortRectifyMap` con `getOptimalNewCameraMatrix`,
`alpha=0`), y cada fotograma se corrige con `cv.remap` + recorte al ROI. Las **coordenadas de
mirada** se corrigen aparte (`correctCoordinates`): `undistortPoints` y, opcionalmente,
aplicación de una **homografía** para llevarlas a la vista cenital del tablero. Así un punto
de *gaze* puede proyectarse tanto a la imagen desdistorsionada (sin homografía) como a la
casilla del tablero (con la homografía del tablero).

| Original (con distorsión de lente) | Desdistorsionada |
|---|---|
| ![original](media/documentation/undistort_original.png) | ![desdistorsionada](media/documentation/undistort_corregida.png) |

*Las líneas rectas del tablero vuelven a verse rectas tras corregir la distorsión del objetivo.*

### Corrección de color (`ARUCOColorCorrection`)

Balance de blanco tipo *gray-world* en espacio **LAB** con peso por luminancia: se estima el
sesgo medio de los canales `a` y `b` (submuestreando 1 de cada 4 px por velocidad) y se resta
ponderado por la luma del píxel (`luma·1.1`), de modo que las zonas claras se corrigen más que
las sombras. Se aplica a **todo el fotograma desdistorsionado**. Mantiene los colores de las
fichas estables ante cambios de iluminación, lo que es clave para las máscaras de color (borde
del tablero, máscara de panel) y para la corrección de color del propio tablero.

| Original | Corregida (LAB gray-world) |
|---|---|
| ![original](media/documentation/color_original.png) | ![corregida](media/documentation/color_corregida.png) |

*Mismo fotograma antes (dominante cálida del entorno) y después de la corrección.*

### Detección y filtrado de ArUcos

`detectAllArucos` corre el detector una vez por fotograma. `filterValidArucos` descarta
cualquier id que no esté en la *whitelist* (ids del tablero ∪ ids de todos los paneles): el
detector dispara ocasionalmente el id 0 sobre piezas del tablero, y esas detecciones espurias
se eliminan antes de cualquier proceso o dibujo. **Nota medida:** se probó suavizar las
esquinas de los ArUcos entre fotogramas (`smoothArucos`) para estabilizar la homografía, pero
degradaba la detección de contorno y perdía *trials*; se revirtió.

**Detección sobre la imagen ORIGINAL (no la desdistorsionada).** Los ArUcos se detectan en la
imagen **original (distorsionada)** y sus esquinas se desdistorsionan después
(`correctCoordinates`). La desdistorsión (`alpha=0`) empuja los marcadores de los **bordes**
—sobre todo la fila superior— fuera del encuadre y los pierde, **aunque estaban completos en
la original** (un marcador cortado no lo detecta OpenCV). Medido: el participante 042 perdía
marcadores en **16/16** fotogramas (~5 por fotograma), debilitando la homografía y provocando
fallos `few_arucos`. Como `alpha=0` mantiene `newK == K`, las esquinas desdistorsionadas
coinciden con la imagen `alpha=0` a 0,15 px, así que **no cambian ni la resolución ni la
proyección de la mirada**: solo se recuperan marcadores.

| Detectados en la imagen original (todos) | En la desdistorsionada (se pierden los del borde) |
|---|---|
| ![aruco original](media/documentation/aruco_original.png) | ![aruco desdistorsionada](media/documentation/aruco_desdistorsionada.png) |

*Mismo fotograma de 042: 7 marcadores en la original vs 2 en la desdistorsionada (la fila
superior se pierde). Por eso se detecta sobre la original.*

---

## 4. Localización del tablero (homografía + rejilla)

El tablero se localiza con los **marcadores ArUco** que lo rodean: con sus esquinas se calcula
una **homografía** que proyecta el tablero a una **vista cenital** fija (`board_view`,
~1280×720). Sobre esa vista se reconstruye la rejilla de 8×5 casillas.

![tablero con ArUcos](media/documentation/tablero_aruco_completo.png)

`ArucoBoardHandler.getTransform` casa los ArUcos detectados con los configurados
(`game_aruco_board.yaml`), estima si el tablero se ve **girado 180°** (solo para el tablero,
no para los paneles: la rejilla se voltea con él) mediante `estimatePoseSingleMarkers` con
**histéresis** (`rotation_flip_threshold=3`, así un marcador suelto no fuerza el giro), y
calcula la homografía a la vista cenital.

Para estabilizar la rejilla cuando faltan marcadores, se guarda una **rejilla de referencia**
(mediana del rectángulo del tablero sobre ≥5 muestras estables, `reference_board_rect`); con
ella la `cell_matrix` se mantiene aunque el contorno del fotograma actual no se reconstruya.

El **contorno del tablero** (`detectContour`, [BoardHandler.py](../src/core/BoardHandler.py))
no es un borde genérico: muestrea el **color del marco** en los bordes de la vista cenital,
construye una máscara de ese color, aplica Canny + `findContours` y se queda con el
**rectángulo** grande. Es, en la práctica, el **borde blanco** del tablero.

| Tablero despejado | Mano dentro (alcance) | Máscara de blanco (alcance) |
|---|---|---|
| ![board clean](media/segmentation/01_board_warp_clean.png) | ![board reach](media/segmentation/02_board_warp_reach_handsin.png) | ![white mask](media/segmentation/05_white_mask_reach.png) |

Sobre esa vista cenital, la `cell_matrix` se construye **dividiendo el rectángulo del tablero
en una rejilla regular de 8×5** (`computeBoardMatrixFromRect`): el rectángulo viene de la
homografía de ArUcos (estabilizado por la mediana `reference_board_rect`) o, como respaldo, del
contorno detectado. Los tamaños de celda son **flotantes** (la división entera acumulaba hasta
`board_size−1` px de resto en los bordes derecho/inferior y clasificaba mal la mirada de las
últimas casillas como `not_board`). A cada casilla se le asigna su **propiedad** —color, forma
y si es ficha o hueco— desde `game_config.yaml` (rotada 180° si el tablero se ve girado).
`getCellIndex` localiza la casilla de una coordenada ya proyectada; `getPixelBoardNorm` la
normaliza contra el área del tablero (0,0 esquina superior izquierda; 1,1 inferior derecha).

La imagen siguiente es el procesamiento **real** sobre un fotograma: los **ArUcos detectados**
(marcas con punto), el **contorno del tablero** (recuadro verde), la **vista cenital
reproyectada** con la rejilla y las propiedades de cada casilla (recuadro inferior derecho,
*picture-in-picture*) y el **estado de la máquina** (texto superior). Es el mismo render que
produce el `debug_<id>.mp4`, donde además la mirada se dibuja como círculos de color según
dónde cae (verde = casilla del tablero, naranja = anticipada, rojo = sobre el panel y por tanto
**no contabilizada**, blanco = fuera del tablero).

![reprojección y rejilla del tablero](media/documentation/proceso_overview.png)

### Hallazgo medido: la homografía es estable ante pérdida de ArUcos

Midiendo la **deriva** de la vista cenital (correlación cruzada del perfil de blanco frente a
un fotograma limpio) a lo largo de varios trials, la vista se mantiene **estable (≈1 px)**
aunque los ArUcos bajen a 6–8 y el contorno se pierda; `board_view` **sigue presente** durante
todo el alcance (medido en 12 trials de 4 participantes: `board_view` presente el 100% del
tiempo). La rejilla de referencia estable mantiene la homografía anclada. **Consecuencia
práctica:** los fallos del detector de toque **no** vienen de que la proyección se desplace o
desaparezca (salvo cuando los ArUcos caen a 2–3 y la `cell_matrix` deja de poder calcularse).

---

## 5. Detección del panel de estímulo (`PanelHandler`)

El panel de muestra que se enseña antes de cada *trial* se detecta igual que el tablero, pero
con un `ArucoBoardHandler` **por cada panel configurado** (`sample_shape_cfg/`, un YAML por
combinación color×forma). El nombre del fichero codifica la identidad (`<forma>_<color>.yaml`),
así que **detectar el panel = identificar qué objeto se busca**; no hace falta leer la imagen
del panel. Los paneles se crean con `estimate_rotation=False`: su orientación es irrelevante
(solo importan presencia e identidad), lo que ahorra la estimación de pose por panel.

`PanelHandler.step` prueba cada handler de panel y se queda con el primero cuyos ArUcos casen.
`processPanel` en la máquina de estados aplica además una **confirmación de 2 fotogramas**
(`panel_detected_threshold=2`) para no disparar con una detección aislada.

**Panel como máscara de oclusión** (`getPanelPolygon`): mientras el panel se retira, su área
se proyecta a la imagen (homografía inversa de su vista, expandida 1,05× alrededor del centro)
y se **excluye** de tres cosas: de la clasificación de la mirada (gaze sobre el panel → fase
`on_panel`, ver guía §3), de la detección del borde del tablero y de la medida de oclusión de
mano. Así el panel —que barre sobre el tablero al retirarse— no rompe el contorno ni finge una
entrada de mano, pero la mano entrando por la zona ya libre **sí** se detecta.

El panel se identifica por sus ArUcos (recuadro = polígono detectado), de donde sale el
color/forma del objetivo a buscar:

![detección del panel](media/documentation/deteccion_panel.png)

---

## 6. Emparejamiento mirada–vídeo (`EyeDataHandler`)

La explicación a nivel de usuario (dos relojes, conteo→tiempo) está en la [guía §4 y §7]; aquí
el **cómo**. `EyeDataHandlerPLDATA` carga el `.pldata` (vía `msgpack`) y, en el arranque:

1. **Filtra por confianza**: descarta muestras con `confidence ≤ 0.6` (umbral sugerido por
   Pupil Labs). Las inválidas se conservan **solo** para medir la frecuencia (abajo).
2. **Mide la frecuencia real** `gaze_sampling_rate = 1 / mediana(Δt)` sobre **todas** las
   muestras (válidas e inválidas), porque cada muestra ocupa un intervalo de muestreo del
   aparato. `gaze_continuity = fracción de Δt dentro de ±20% de la mediana`. (Por qué *todas*
   y no solo las válidas: [guía §7].)
3. **Empareja cada muestra a un *frame* de *World*** comparando *timestamps* con
   `bisect_right` sobre `world_timestamps` ordenados (la muestra "cae" en el frame cuyo
   intervalo la contiene). El *gaze* **no** trae número de *frame*; se calcula aquí.
4. **Propaga la duración** (solo *fixations*): una fijación de `duration` ms se replica por
   `int(fps · duration/1000)` *frames* consecutivos (mínimo 1). El *gaze* crudo dura un frame.
5. **Voltea el eje vertical** (`Y → 1−Y`): las gafas usan origen abajo-izquierda; la imagen,
   arriba-izquierda.

El resultado es `fixation_start_world_frame{frame: [índices de muestra]}`. `step(frame)`
devuelve las coordenadas normalizadas de las muestras de ese *frame*; por eso a un *frame*
le corresponden varias muestras (la mirada va más rápido que el vídeo).

> Existe también `EyeDataHandlerCSV` (mismo contrato `step()` sobre un CSV con
> `start_frame_index`/`end_frame_index`), pero el procesado de referencia usa la vía `.pldata`.

### Clasificación de la mirada (qué cuenta y qué no)

Cada muestra proyectada se etiqueta según **dónde cae**, y eso decide si entra en los
contadores. El `debug_<id>.mp4` la dibuja con un marcador unificado (halo oscuro + núcleo de
color + anillo claro) para que se lea sobre cualquier fondo y no se confunda con las esquinas
de los ArUcos. Colores: verde = casilla (cuenta); naranja = anticipada sobre el tablero
(cuenta); magenta = sobre el panel (no cuenta); azul = fuera del tablero (`not_board`, no
cuenta); gris = casilla aún tapada por el panel (no cuenta). Cada fila muestra la **vista
general** (el recuadro amarillo marca la zona) y su **recorte** ampliado:

| Caso | Vista general | Recorte |
|---|---|---|
| **Cuenta** — sobre una casilla | ![](media/documentation/gaze_tablero.png) | ![](media/documentation/gaze_tablero_zoom.png) |
| **Cuenta** — anticipada (panel saliendo) | ![](media/documentation/gaze_anticipada.png) | ![](media/documentation/gaze_anticipada_zoom.png) |
| **No cuenta** — sobre el panel de muestra (mirando la señal) | ![](media/documentation/gaze_panel.png) | ![](media/documentation/gaze_panel_zoom.png) |
| **No cuenta** — junto al panel mientras se retira (`not_board`) | ![](media/documentation/gaze_panel_no.png) | ![](media/documentation/gaze_panel_no_zoom.png) |
| **No cuenta** — fuera del tablero | ![](media/documentation/gaze_fuera.png) | ![](media/documentation/gaze_fuera_zoom.png) |

---

## 7. Máquina de estados en detalle

Hay **una sola** máquina de estados; los "dos niveles" (detección vs trial) son dos formas de
leer su resultado, no dos máquinas (modelo conceptual y diagramas en la [guía §3]). Estados:

`init → get_test_name → test_start_execution → test_execution → test_motor_recovery → test_finish_execution → init`

Cada callback recibe `(undistorted_image, capture_idx, gaze_list)` y puede cambiar
`self.current_state`; el bucle de despacho (apartado 1) reejecuta el nuevo estado en el mismo
fotograma si hace falta.

```mermaid
stateDiagram-v2
    [*] --> init
    init --> get_test_name: panel de muestra detectado
    get_test_name --> test_start_execution: panel identificado (color/forma)
    test_start_execution --> test_execution: contorno del tablero confirmado
    test_execution --> test_motor_recovery: contorno perdido = motor_onset (la mano entra)
    test_motor_recovery --> test_finish_execution: hand_exit, o aparece el panel siguiente (by_next_panel)
    test_finish_execution --> init: trial cerrado y publicado
    note left of test_start_execution: search_start: empiezan a contar la búsqueda y la mirada anticipada
    note right of test_execution: target_found: primer gaze sobre la casilla objetivo
    note right of test_motor_recovery: marcas target_touch y hand_exit (mejor esfuerzo)
```

El ciclo se cierra en `init`, listo para el siguiente panel. Las **marcas** (`search_start`,
`target_found`, `motor_onset`, `target_touch`, `hand_exit`) no son estados: son **instantes**
dentro de estos estados, y delimitan las fases del trial (búsqueda → verificación → motora →
retirada). La derivación exacta está en el apartado 8.

Los tres fotogramas siguientes (del `debug_<id>.mp4`, que rotula el estado en la esquina
superior) muestran las transiciones clave de un *trial* real: el panel de muestra mostrado, su
retirada con el tablero apareciendo, y la búsqueda con la mirada ya proyectada sobre las
casillas y la vista cenital en el *picture-in-picture*.

| Estado | Qué dispara el cambio | Fotograma |
|---|---|---|
| `init` / `get_test_name` | aparece el **panel de muestra** (señal: hexágono verde) sobre el tablero | ![panel](media/documentation/estado_1_panel.png) |
| `test_start_execution` | el **experimentador retira el panel**; el tablero queda visible y la **búsqueda visual comienza** | ![retirada](media/documentation/estado_2_retirada.png) |
| `test_execution` | el **contorno del tablero** se confirma → búsqueda; la mirada se clasifica por casilla | ![búsqueda](media/documentation/estado_3_busqueda.png) |
| `test_motor_recovery` | la mano **cruza el borde** y alcanza la pieza (contorno perdido) | ![motora](media/documentation/estado_4_motora.png) |

Dentro de `test_motor_recovery` se resuelven tres **sub-marcas** (no son estados, son los
hitos que delimitan las fases motoras): la mano **entra** (cruza el borde, `motor_onset`),
**toca** la pieza objetivo (`target_touch`) y **sale** del tablero (`hand_exit`):

| Sub-marca | Qué señala | Fotograma |
|---|---|---|
| **entra** (`motor_onset`) | la mano cruza el borde del tablero hacia la pieza | ![entra](media/documentation/submarca_entra.png) |
| **toca** (`target_touch`) | la mano alcanza la pieza objetivo (mejor esfuerzo) | ![toca](media/documentation/submarca_toca.png) |
| **sale** (`hand_exit`) | la mano abandona el tablero tras responder | ![sale](media/documentation/submarca_sale.png) |

La segmentación de estas tres marcas en el tiempo se ve además en la señal de oclusión del
objetivo (apartado 9, `oclusion_temporal`): sube al entrar/tocar y baja al salir.

### `init` — espera y casado con la secuencia

Vigila la aparición de un panel (`processPanel`). Cuando aparece uno:

- **Casado con la secuencia esperada** (`default_trials_config.yaml`): si el panel detectado
  no coincide con el *trial* esperado, se avanzan e inscriben los *trials* intermedios como
  `missing_trial_error_<esperado>` (con `init/end = -1`) hasta encontrar el que casa. Así un
  fallo de detección no descuadra el resto.
- **Rechazo de paneles fuera de secuencia / espurios** (`_detectedInRemaining`): si el panel
  no aparece en lo que queda de secuencia, se **ignora** (sigue en `init`) en vez de consumir
  toda la secuencia como *missing* y terminar el run.
- Al casar, fija `block_id`/`trial_id` y pasa a `get_test_name`.

### `get_test_name` — panel visible (codificación)

El panel está presente y el objeto a buscar identificado. Espera a que el panel **desaparezca**
(se retira) para pasar a `test_start_execution`. Si en su lugar aparece un panel **distinto**
(salto), `_handleUnexpectedPanel` lo gestiona (abajo).

### `test_start_execution` — el tablero aparece

Mientras el panel se retira:

- **Mirada anticipada** (`processEarlyGaze`): la pose del tablero ya se conoce por los ArUcos,
  así que el *gaze* de esta ventana se registra etiquetado por dónde cae —`pre_start`,
  `on_panel`, `blank`, `not_board`— (ver guía §3 y §6.2). En v1.2 **ninguna** se descarta.
- **Cebado de la referencia de toque** (`_primeTouchReference`): captura una referencia limpia
  de la casilla objetivo en esta ventana permisiva, para objetivos de borde a los que la mano
  llega antes de poder capturarla más tarde.
- **Racha de confirmación de contorno**: requiere `board_contour_start_confirm_threshold=6`
  fotogramas con contorno estable antes de arrancar el *trial* (una detección aislada producía
  *trials* degenerados). El primer *frame* de la racha (`contour_streak_start_frame`) se guarda
  para **retrotraer** `init_capture` sobre él. Confirmada la racha → `test_execution`.

### `test_execution` — búsqueda

Al entrar (primera vez):

- **Retrodatado de `init_capture`** al primer *frame* de la racha; las muestras `pre_start`
  dentro de la racha se **relabelan a `execution`** y pasan a contar.
- **Inicio del seguimiento de toque** (`initTargetTracking`): umbral por color
  (`touch_threshold_by_color`, apartado 9 y 13), marca de objetivo cálido (rojo/amarillo),
  celdas de control. Se salta si `_primeTouchReference` ya cebó una referencia limpia.
- **Referencia de oclusión de tablero** (`board_occ_ref`): se captura el tablero limpio para,
  desde aquí, medir cada fotograma cuánto lo ocluye la mano (sirve a `hand_exit`).

En cada fotograma: registra el *gaze* (fase `execution`), vigila el toque (`_trackTargetTouch`)
y la oclusión de tablero (`_trackBoardOcclusion`). **Fin del *trial*** = pérdida sostenida del
contorno (la mano cruza el borde): tras `board_contour_switch_state_threshold=4` *frames* sin
contorno, `finishTrial` fija `end_capture` retrotraído al último *frame* con tablero visible y
publica `frame_motor_onset`. También sale por `_handleUnexpectedPanel` si aparece el panel
siguiente (cierre `test_finish_by_next_panel`, duración = cota superior).

### `test_motor_recovery` — fase motora (toque + salida de mano)

Tras decidir el fin, **no cierra aún**: sigue para anotar el toque y la **salida de la mano**.
Es el estado más delicado. En cada fotograma:

1. **Cede a cualquier panel confirmado** (mismo o distinto): el panel de muestra ya se retiró
   al inicio del *trial*, así que un panel confirmado aquí es la presentación siguiente → cierra
   y deja que `init` la recoloque. Sin esto se perdían *trials* con re-presentación del mismo
   panel (~3% de *trials* válidos, medido).
2. **Registra el *gaze*** como fase `motor` (alcance + retirada), que va a la secuencia pero no
   a los contadores.
3. **Vigila el toque** (`_trackTargetTouch`); si el toque se confirma justo ahora, **reinicia**
   la cuenta de `hand_exit` para que ésta sea el contorno que vuelve *después* del toque, no el
   de mitad de alcance (ver apartado 10).
4. **`hand_exit` por tres fuentes**, en orden de sensibilidad:
   - **`ft_return`**: la oclusión **local** del objetivo (`fT`), tras subir por encima de
     `ft_enter_level=0.20`, vuelve sostenidamente bajo `ft_exit_level=0.05`. La más sensible
     para alcances pequeños (un dedo sobre una casilla).
   - **`board_occ`**: la oclusión del **tablero completo** vuelve a reposo tras haber subido.
     A diferencia del contorno, se mantiene alta a mitad de alcance, así que su retorno es la
     mano saliendo de verdad aun sin toque detectado.
   - **`contour`**: el contorno reaparece de forma sostenida (`motor_recovery_confirm=3`) y
     **después** del toque, con tolerancia a parpadeo (`motor_recovery_miss_tolerance=2`).
   - **`deadline`**: a `motor_recovery_max_frames=75` (~2,5 s) cierra con la mejor estimación
     disponible.

### `test_finish_execution` — cierre

`trimTrialToFrame` recorta la secuencia al `end_capture` (conservando las muestras `motor`),
recalcula los contadores por color **hasta el toque**, deriva `touch_diag`, y guarda una copia
profunda en `board_metrics_store[(block,trial)]`. Vuelve a `init`.

### Gestión de saltos / errores (robustez)

- **`_handleUnexpectedPanel`** (usado por `test_start_execution`, `test_execution`,
  `get_test_name`): con datos ya recogidos cierra como `test_finish_by_next_panel` (válido,
  cota superior); sin datos, `transition_error_no_init`. Vuelve a `init` para re-emparejar.
- **Registro de transiciones** (`_recordTransition`): cada cambio de estado se anota con
  *frame*, origen/destino y block/trial; se fusiona luego con las marcas (apartado 8).
- **Inicio re-basado y registro completo de gaze**: el *trial* arranca en `frame_search_start`
  (primera mirada al tablero durante la retirada del panel), no en `frame_init`; por eso
  `trial_duration_s` **no es comparable** con 1.0/1.1. Detalle de fases en la [guía §3].

### Errores que publica

| Tag | Significado |
|---|---|
| `missing_trial_error_*` | un panel esperado no se detectó (suele ser fallo de detección, no del participante). |
| `transition_error_no_init_*` | apareció otro panel antes de que el *trial* recogiera datos. |
| `test_finish_by_next_panel` | *trial* **válido** cerrado al aparecer el panel siguiente (duración = cota superior). |
| `test_finish_by_end_of_video` | *trial* cerrado por fin de grabación. |

---

## 8. Derivación de marcas, fases y línea de tiempo (`store_results`)

A partir del `board_metrics` de cada *trial*, `store_results` calcula las columnas de los CSV:

- **Marcas crudas** (`frame_*`): `search_start = early_init_capture (o init_capture)`,
  `init`, `target_found` (primera muestra de la secuencia cuya casilla = objetivo; la mirada
  cae sobre la pieza objetivo, resaltada en la vista cenital), `motor_onset = motor_onset_capture`,
  `target_touch`, `hand_exit`, `end = end_capture`.

![target encontrado](media/documentation/marca_target_found.png)
- **Duraciones por fase**: `time_to_target`, `search` (inicio→motor_onset), `reach`
  (motor_onset→toque), `withdraw` (toque→hand_exit). Vacías si falta la marca que las acota.
- **Covariables**: `anticipatory_gaze` (nº de muestras `pre_start`), `anticipation_lead_s`,
  `target_row/col`. La distancia de alcance (mm) **no** se repite por trial: vive en
  `target_geometry.csv` (la calcula `writeTargetGeometry` desde `game_config.yaml`: centro de
  celda en mm y distancia recta 2D desde el lado de entrada del participante).
- **`phaseOf(frame, base_phase)`**: las muestras de la ventana previa conservan su etiqueta de
  localización (`pre_start`/`on_panel`/`blank`/`not_board`); el resto se parte por las marcas
  en `search`/`verification`/`motor`/`withdraw` (definición en la [guía §3]).

Las marcas son los **límites de las fases**; un trial real se segmenta así (en objetivos de
borde la fase motora puede quedar muy corta, como aquí):

![línea de tiempo de fases](media/documentation/timeline_fases.png)

### Línea de tiempo unificada (estados + marcas)

`state_transitions` se **fusiona** con las marcas conductuales (cada `frame_*` se convierte en
una fila-evento) en una sola tabla cronológica por *trial* (`..._transitions.csv`, columna
`event`). El **toque NO es un estado**: hacerlo estado acoplaría la segmentación a una señal
*best-effort* y desincronizaría la secuencia ante un toque perdido (~10%). La fusión da la
vista unificada sin ese acoplamiento. Formato y ejemplo en la [guía §6.3].

### Diagnóstico de toque (`touch_diag`)

`_deriveTouchDiagnostics` clasifica por qué un *trial* tiene o no toque, para depurar cobertura:
`confirmed`, `fT_below` (subió pero no llegó al umbral), `control_ge` (una celda de control
cambió tanto como el objetivo), `unconfirmed`, `few_arucos` (sin `cell_matrix`),
`never_activated`. No es una columna de los CSV de análisis; se usa en el informe HTML.

---

## 9. Detector de toque (best-effort)

El toque de la pieza objetivo (`frame_target_touch`) se detecta por **cambio de imagen** (no
por color) en el entorno de la casilla objetivo de la vista cenital, comparado con su
apariencia de referencia limpia (`getTargetOcclusionMeasure`). Funciona con manga de cualquier
color. Tres salvaguardas contra falsos positivos:

1. **Composición de color (histograma H-S)**: un desplazamiento del *warp* mantiene los
   colores; una mano mete un color nuevo. Si el histograma apenas cambia, es deriva → 0.
2. **Alineación por correlación de fase**: compensa el micro-temblor del *warp* (±10 px) antes
   de medir el cambio.
3. **Separación frente a celdas de control**: el cambio en el objetivo debe destacar sobre la
   **mediana de celdas de control** alejadas, para descartar cambios globales (la mano sobre
   todo el tablero).

La señal blanco/color de la casilla se mide sobre los **píxeles** (umbral de saturación), no
por geometría, así que un pequeño desfase de la celda no reasigna las zonas.

Sobre la vista cenital se marcan la **celda objetivo** (recuadro amarillo) y las **celdas de
control** (gris); abajo se lee `occl target/ctrl`. La oclusión solo se confirma cuando el
objetivo cambia y el control **no**:

| Tablero limpio (objetivo visible) | Mano alcanzando (objetivo ocluido) |
|---|---|
| ![oclusión limpia](media/documentation/reproyeccion_celdas.png) | ![oclusión toque](media/documentation/oclusion_areas.png) |

*`occl target/ctrl` pasa de **0.00/0.00** (limpio) a **0.50/0.00** (la mano oclúye el objetivo
mientras las celdas de control siguen a 0): esa separación es la que confirma el toque.*

A lo largo del trial, la oclusión del **objetivo** (`fT`) permanece a 0 durante la búsqueda,
sube al cruzar el umbral cuando la mano alcanza la pieza y vuelve a 0 al retirarse, mientras
la del **control** (`fC`) se mantiene plana — esa es la señal que se vigila:

![oclusión en el tiempo](media/documentation/oclusion_temporal.png)

**Refinamientos v1.2** (sobre lo anterior):

- **Umbral de toque por color** (`touch_threshold_by_color = {red:0.13, yellow:0.15, blue:0.20,
  green:0.20}`). Medido sobre la distribución de `fT` de los 20 participantes: los toques reales
  superan ~0.22 en todos los colores, pero las piezas **cálidas** (rojo/amarillo ≈ tono de
  piel) dan una señal genuinamente menor mucho más a menudo (una mano cálida sobre una pieza
  cálida apenas cambia el color), así que muchos toques reales caían en ~0.13–0.20 y se perdían
  (rojo: 68 fallos vs azul 11). Para esos colores se baja el umbral; la separación frente a
  control sigue protegiendo contra falsos positivos.
- **GATE 1 (color) se omite para objetivos cálidos**: el histograma H-S apenas cambia cuando
  mano y pieza comparten tono, así que para rojo/amarillo se confía en el cambio de gris/borde.
- **Componentes de cambio adicionales**: además del histograma, gradiente de **bordes (Sobel)**
  y **SSIM** (`structural_similarity`, `scikit-image`) entre la casilla y su referencia limpia,
  más sensibles a la estructura (un dedo) que al ruido de color.
- **Plantilla de sesión** (`session_template`): referencia limpia persistente del tablero,
  refrescada cada 15 fotogramas, usada como respaldo cuando no hay ventana limpia reciente para
  la casilla objetivo.

> **Coste:** estos componentes (SSIM + Sobel + oclusión per-frame + blend de plantilla)
> triplican el tiempo de procesado respecto a v1.1 (~25 min/participante vs ~7–8). Si la
> velocidad importara, el camino es calcular SSIM/Sobel **de forma perezosa** (solo cuando los
> gates baratos ya sugieren toque), no en todos los fotogramas.

### Hallazgo medido: el fallo del toque era de temporización, no de homografía

El toque ocurre en la **fase motora**, con la mano ya sobre el tablero; la señal de oclusión
del objetivo **sube con claridad** (medido: `fT` llega a ~1.0). El problema medido (7/7 fallos
recuperables) era que `test_motor_recovery` **cerraba el trial antes** de que el toque
culminara: el contorno reaparece a mitad de alcance (la mano ya cruzó el borde y está en el
centro) y se leía como "la mano salió". El toque ocurría 13–57 fotogramas **después**. Solución:
registrar `hand_exit` pero **no cerrar** con esa reaparición; el cierre y el `hand_exit` real se
toman con el contorno que vuelve **después** del toque, o con el panel siguiente, o por *timeout*.

Cobertura best-effort medida en los 20 participantes (1119 trials): **92.0%** en v1.2 (vs
**82.8%** en v1.1; rojo 72.5%→89.5%, fila 4 —cerca del participante— 70.2%→88.3%), con **0
toques implausibles** (ninguno antes de aparecer el tablero ni después de salir la mano).
Limitaciones residuales medidas: ArUcos a 2–3 (sin `cell_matrix`), geometría del alcance que no
oclúye la casilla, o celdas de control más ocluidas que el objetivo. El cierre del *trial* **no**
depende del toque (es el cruce del borde, `frame_end`), así que un toque perdido no descuadra la
segmentación.

---

## 10. Marcas motoras y la ambigüedad del contorno

| Marca | Señal que usa |
|---|---|
| `frame_motor_onset` / `frame_end` | **pérdida sostenida del contorno** (la mano cruza el borde hacia dentro) |
| `frame_target_touch` | **oclusión por cambio** del entorno de la casilla objetivo (mejor esfuerzo) |
| `frame_hand_exit` | **oclusión que vuelve a reposo** (local o de tablero), o contorno que vuelve tras el toque |

Orden temporal: borde-entra (`motor_onset`/`end`) → toque → mano-sale (`hand_exit`). Clave: el
**contorno (borde) solo se oculta mientras la mano lo cruza**, no mientras está realcanzando en
el centro; por eso reaparece a mitad de alcance y no sirve, por sí solo, para distinguir
*mano-dentro-realcanzando* de *mano-fuera*. De ahí que `hand_exit` se ancle a la **oclusión**
(que sí se mantiene alta a mitad de alcance) y, como respaldo, al contorno que vuelve **tras**
el toque. Es también la razón de que `target_touch` pueda caer un par de *frames* **después**
de `motor_onset`/`trial_end` (ver el ejemplo de la [guía §6.3]).

---

## 11. Frecuencia de muestreo del *gaze*

No se asume 200 Hz: se **mide** por participante como `1 / mediana(Δt entre muestras)`, usando
**todas** las muestras (válidas e inválidas), porque cada muestra representa un intervalo de
muestreo (implementación en el apartado 6; conteo→tiempo en la [guía §7]). `gaze_continuity` =
fracción de Δt dentro de ±20% de la mediana (avisa de muestreo irregular). Valores observados
≈124 Hz y ≈248 Hz según el participante. Mezclar participantes a frecuencias distintas sin
corregir introduciría un sesgo sistemático; por eso el tiempo de mirada se calcula siempre con
la frecuencia medida de cada uno.

---

## 12. Limitaciones del equipo de medida (mecanismo)

El dispositivo es el **Pupil Labs Core** (Kassner, Patera & Bulling, 2014): cámaras oculares a
200 Hz (192×192 px), cámara de escena (*World*) hasta 1080p/30 Hz (estas grabaciones a **1280×720**), exactitud 0,60° y precisión 0,02°
(con calibración, en laboratorio). Las consecuencias para estos datos:

- **Error angular → incertidumbre espacial.** La exactitud de 0,60° se degrada en uso
  naturalista (movimiento de cabeza, iluminación variable). Ese error angular se traduce en
  incertidumbre al asignar la mirada a una casilla: a mayor distancia participante–tablero y
  casillas más pequeñas, más probable que una muestra caiga en la casilla adyacente.
- **Detección de pupila y características individuales.** El algoritmo usa técnica de **pupila
  oscura** sobre la imagen **infrarroja** del ojo. El color del iris, la forma del párpado, la
  presencia de pliegue epicántico u otras características anatómicas influyen en la facilidad y
  precisión de esa detección. No están controladas en el diseño y explican parte de la variación
  entre participantes en muestras válidas y precisión (no es diferencia de hardware: el equipo es
  el mismo).
- **Deriva de la calibración.** Se calibra al inicio de la sesión; si las gafas se desplazan
  durante el experimento, la correspondencia mirada-estimada↔posición-real se degrada. El modelo
  3D del ojo incorpora compensación de deslizamiento (*slippage compensation*), que no elimina
  el efecto por completo.
- **Pérdida de señal.** Parpadeos, movimiento brusco o reflejo corneal desfavorable producen
  muestras ausentes o de baja confianza (≤0,6, descartadas). Esos fragmentos no se contabilizan;
  los totales previos al filtrado no se conservan (no se guardaron los ficheros de captura).

---

## 13. Parámetros clave

| Parámetro | Dónde | Valor | Qué controla |
|---|---|---|---|
| `board_contour_switch_state_threshold` | StateMachine | 4 | fotogramas sin contorno para cerrar el trial (motor_onset) |
| `board_contour_start_confirm_threshold` | StateMachine | 6 | fotogramas de contorno estable para iniciar el trial |
| `panel_detected_threshold` | StateMachine | 2 | fotogramas para confirmar un panel |
| `target_occlusion_threshold` / `_separation` | StateMachine | 0.20 / 0.10 | umbral de toque por defecto y separación frente a control |
| `touch_threshold_by_color` | StateMachine | red 0.13 / yellow 0.15 / blue 0.20 / green 0.20 | umbral de toque por color (cálidos más bajos, v1.2) |
| `occlusion_pixel_diff` / `_edge` / `_ssim` | BoardHandler | 60 / 50 / 0.55 | umbrales de los componentes de cambio (gris, Sobel, SSIM) |
| `target_occlusion_confirm_threshold` | StateMachine | 2 | fotogramas de oclusión sostenida para confirmar el toque |
| `target_warmup_frames` | StateMachine | 3 | espera mínima antes de vigilar el toque (panel barriendo) |
| `motor_recovery_max_frames` | StateMachine | 75 | ventana (~2,5 s) para vigilar toque + salida de mano |
| `motor_recovery_confirm` / `_miss_tolerance` | StateMachine | 3 / 2 | contorno sostenido para `hand_exit`, con tolerancia a parpadeo |
| `ft_enter_level` / `ft_exit_level` / `ft_exit_confirm` | StateMachine | 0.20 / 0.05 / 3 | entrada/salida/confirmación de `hand_exit` por oclusión local |
| `board_occ_enter_level` / `board_occ_exit_level` / `_confirm` | StateMachine | 0.12 / 0.05 / 3 | ídem por oclusión de tablero completo |
| `_template_blend_every` | BoardHandler | 15 | cada cuántos fotogramas se refresca la plantilla de sesión |
| `rotation_flip_threshold` | ArucoBoardHandler | 3 | histéresis del giro 180° del tablero |

Los valores se han ajustado **midiendo** sobre muestra; no son nominales.

---

## Referencias

Kassner, M., Patera, W., & Bulling, A. (2014). Pupil: An open source platform for pervasive eye
tracking and mobile gaze-based interaction. En *Adjunct Proceedings of UbiComp 2014* (pp.
1151–1160). ACM. https://doi.org/10.1145/2638728.2641695

(Referencias metodológicas sobre *gaze* vs *fixations* y *dwell time* en la
[guía de procesamiento](guia_procesamiento.md#referencias).)
