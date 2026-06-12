# Guía de comprensión del procesamiento de datos

## 1. ¿Qué problema resuelve?

En el experimento, una persona lleva unas **gafas de seguimiento ocular** (Pupil Labs) y tiene que **buscar un objeto** (definido por un **color** y una **forma**, p. ej. "hexágono rojo") en un **tablero físico** de fichas de colores. Antes de cada búsqueda se le enseña un panel con el objeto que debe encontrar. El experimento completo tiene **6 *blocks* de 10 *trials*** cada uno.

Las gafas graban dos cosas:

- Un **vídeo de lo que ve la persona** (la escena, con el tablero): es el vídeo de *World*.
- La **posición de la mirada** (*gaze*) en cada instante: a dónde está mirando dentro de ese vídeo.

El objetivo del software es, a partir de esos datos, reconstruir **a qué casilla del tablero estaba mirando la persona en cada momento** y resumir, para cada *trial*, **cuánto miró cada ficha y cada hueco**. Eso es lo que luego se analiza.

---

## 2. Entradas (lo que necesita el programa)

Por cada participante existe una carpeta `data/<id>/` con:

| Entrada | Qué es |
|---|---|
| `world.mp4` | El vídeo de la escena (*World*) grabado por las gafas. |
| `world_timestamps.npy` | El instante de tiempo de cada imagen (*frame*) del vídeo. |
| `gaze.pldata` + `gaze_timestamps.npy` | La posición de la mirada (*gaze*) y su instante de tiempo. (También puede usarse *fixations*, que son miradas ya agrupadas con una *duración*.) |
| `camera_calib.json` | La **calibración de la cámara**, para corregir la deformación del objetivo (las líneas rectas vuelven a verse rectas). |

Además, hay ficheros de **configuración** comunes a todos (carpeta `cfg/`):

| Configuración | Qué define |
|---|---|
| `game_config.yaml` | Cómo es el tablero: una rejilla de **8 columnas × 5 filas** (40 casillas), y para cada casilla su **color**, su **forma** y si contiene una **ficha** o es un **hueco vacío**. |
| `game_aruco_board.yaml` | Dónde están los **marcadores ArUco** (cuadros tipo código QR) que rodean el tablero y permiten localizarlo y enderezarlo. |
| `default_trials_config.yaml` | La **secuencia esperada** de búsquedas: qué objeto se busca en cada *block* y *trial*. (Algunos participantes tienen su propia versión en `cfg/trials_config_exceptions/`.) |
| `sample_shape_cfg/` | La descripción de los **paneles de estímulo** que se le muestran a la persona antes de cada *trial*. |

Colores posibles: **rojo, verde, azul, amarillo**. Formas posibles: **círculo, hexágono, triángulo, cuadrado, trapecio**.

---

## 3. Cómo funciona, paso a paso

El programa principal (`src/process_video.py`) recorre el vídeo de *World* **frame a frame** y, para cada uno, hace lo siguiente. Cada paso está implementado en un "módulo" (un fichero dentro de `src/core/`):

1. **Corrección de color e imagen** *(`ArucoBoardHandler.py`, `DistortionHandler.py`)*. Se ajusta el color y se corrige la deformación de la lente de la cámara, para que el tablero se vea con sus colores reales y sin curvaturas.

2. **Localizar el tablero y "enderezarlo"** *(`BoardHandler.py`)*. Usando los marcadores ArUco que rodean el tablero, el programa calcula la perspectiva y genera una **vista cenital** (como si mirásemos el tablero perfectamente desde arriba). Sobre esa vista enderezada, divide el tablero en su rejilla de 8×5 casillas.

3. **Identificar el panel de estímulo** *(`PanelHandler.py`)*. Cuando a la persona se le muestra el panel con el objeto a buscar, el programa lo detecta y deduce **qué color y forma** toca buscar en ese *trial*.

4. **Emparejar la mirada con el vídeo** *(`EyeDataHandler.py`)* — *ver apartado 4, es importante.*

5. **Saber en qué fase del experimento estamos** *(`StateMachineHandler.py`)*. Un "director de orquesta" (una *máquina de estados*) que va siguiendo el experimento:

   | Fase | Qué está pasando |
   |---|---|
   | `init` | En espera. Se vigila si aparece un panel de estímulo. |
   | `get_test_name` | El panel está visible: se ha identificado el objeto a buscar. |
   | `test_start_execution` | El panel desaparece. Se espera a que el tablero sea visible en su totalidad. |
   | `test_execution` | **La persona busca.** Se registra, en cada instante, a qué casilla mira. |
   | `test_finish_execution` | El tablero queda ocluido por la mano del participante: el *trial* ha terminado y se guardan sus resultados. |

   `StateMachineHandler` también compara lo que ocurre con la **secuencia esperada** de *trials*. Si un *trial* que debía aparecer no aparece, o si hay un salto raro entre paneles, lo marca como **error** (ver apartado 8) en lugar de perder el dato.

### Cómo se detecta el inicio y el fin de cada *trial*

El proceso es completamente **automático**: no hay anotación manual de ningún *trial*. El inicio y el fin se infieren del vídeo usando las únicas referencias que se pueden extraer de forma sistemática mediante análisis de imagen:

- **Inicio del *trial*:** el sistema detecta que el tablero es visible en su totalidad (todos los marcadores ArUco del tablero son reconocibles y su contorno puede reconstruirse). En la práctica, esto ocurre cuando el participante ha quitado el panel de estímulo y tiene el tablero delante sin obstáculos. La detección debe mantenerse varios *frames* consecutivos para confirmar el inicio (una detección aislada podía producir *trials* degenerados casi vacíos), y el `init_capture` se retrotrae al primer *frame* de esa racha confirmada.
- **Fin del *trial*:** el tablero deja de ser detectable, porque la **mano del participante entra en el campo de visión y lo oclúye** al coger la pieza. Es el único evento sistemático de fin de búsqueda que se puede detectar automáticamente en el vídeo. La pérdida del tablero debe confirmarse durante varios *frames* consecutivos para descartar fallos puntuales de detección, pero el `end_capture` registrado **se retrotrae al último *frame* en que el tablero fue realmente visible**, de modo que la duración del *trial* no incluye el tiempo de confirmación (~0,2 s que sí incluían versiones anteriores del software).
- **Caso especial:** si el panel del siguiente *trial* se detecta mientras un *trial* sigue técnicamente en ejecución, es que el detector de oclusión no llegó a dispararse durante la cogida (p. ej. una cogida limpia que no rompe el borde el tiempo suficiente). El *trial* se cierra como válido en el último *frame* con tablero visible y se marca con el estado `test_finish_by_next_panel`. **Atención al interpretarlo:** ese instante es cuando el participante aparta la vista del tablero (ya con la pieza cogida), no cuando la mano entra; la duración es por tanto una **cota superior** que incluye el tiempo entre la cogida y el giro hacia la mesa. Conviene decidir explícitamente en el análisis si estos *trials* se incluyen, se corrigen o se excluyen.

Estas son las **únicas marcas temporales fiables disponibles** en el diseño experimental a partir de la información de imagen; no existen otras referencias que se puedan extraer de forma sistemática para todos los participantes.

**Limitación importante para la interpretación:** el período registrado como *trial* no coincide exactamente con el proceso cognitivo de búsqueda visual. Por un lado, puede haber un pequeño retardo al inicio (el tablero se ve pero el participante aún no ha empezado a buscar activamente). Por otro, el *trial* termina en el **momento de la respuesta motora** (la mano alcanza la pieza), no en el momento en que se toma la decisión. Esta imprecisión es inherente al diseño; **su ventaja es que es consistente y repetible entre todos los participantes**, lo que permite comparaciones válidas.

6. **Proyectar la mirada sobre el tablero**. Para cada punto de *gaze* se calcula su posición en la vista cenital del tablero y, con eso, **en qué casilla cae**: de qué color y forma es esa casilla, y si es una **ficha** o un **hueco vacío**.

---

## 4. Un detalle clave: el vídeo y la mirada van a velocidades distintas

El **vídeo de *World*** se graba a **30 imágenes por segundo** (1080p/30 Hz, confirmado en todos los participantes), pero la **mirada (*gaze*) se registra mucho más rápido: el Pupil Labs Core muestrea a 200 Hz**, es decir, unas 6–7 muestras de *gaze* por cada imagen de *World*. Esto **se tiene en cuenta** al proyectar (`EyeDataHandler.py`):

- Cada medida de *gaze* se asigna **al *frame* del vídeo de *World* que le corresponde**. Esa correspondencia se decide **comparando los instantes de tiempo** (*timestamps*): se mira en qué *frame* de *World* "cae" el instante de cada mirada. (El dato de *gaze* no trae un número de *frame* ya puesto; se calcula a partir de los tiempos.)
- En el caso de las *fixations* (miradas con duración), esa mirada se reparte por **todos los *frames* que dura**.
- Como la mirada va más rápida que el vídeo, **a un mismo *frame* del vídeo le suelen corresponder varias muestras de *gaze***.

Estas frecuencias (30 fps en *World* y 200 Hz en *gaze*) corresponden a las especificaciones del Pupil Labs Core (Kassner et al., 2014). Conviene verificarlas si se incorporan nuevos participantes con una configuración de grabación diferente.

**Consecuencia práctica para el análisis:** los conteos de los CSV cuentan **muestras de *gaze*, no imágenes de vídeo**. Por eso un *trial* puede tener más muestras que *frames*. Cómo convertir esos conteos en tiempo se explica en el apartado 7.

Aparte, se aplican dos ajustes: solo se usan las miradas con **confianza suficiente** (se descartan las que tienen una fiabilidad de **0,6 o menos**, en una escala de 0 a 1; este filtrado se hace al cargar los datos) y se **voltea el eje vertical**, porque las gafas usan el origen abajo-izquierda y las imágenes lo usan arriba-izquierda. El umbral de 0,6 es el que sugiere la propia Pupil Labs: *«In our experience useful data carries a confidence value greater than ~0.6»* ([documentación de Pupil Player](https://docs.pupil-labs.com/core/software/pupil-player/)).

### ¿Por qué se usa *gaze* y no *fixations*?

Las gafas pueden ofrecer dos tipos de dato: la **mirada cruda** (*gaze*: la posición del ojo muestreada a intervalos fijos) o las **fijaciones** (*fixations*: agrupaciones de mirada que un algoritmo interpreta como "una parada" sobre un punto). En este proyecto se trabaja con *gaze*, por dos motivos:

- Las *fixations* dependen de **cómo se parametrice el detector** (umbrales de dispersión, duración mínima, etc.). Como señalan Ehinger et al. (2019), *«it is difficult to establish what an eye movement is, as the definition typically depends on the used algorithm, reference frame, and individual researcher»*; en muchos sistemas las fijaciones se definen de forma residual como todo lo que no es sacada ni parpadeo. Sin una parametrización clara y estándar los resultados no son replicables y, además, la agrupación en fijaciones descarta información entre eventos. En nuestro caso, añadido a eso, no había suficientes *fixations* para el análisis.
- *Gaze* se usa como **medida absoluta y replicable**: al muestrearse a una frecuencia conocida y constante, cada muestra equivale a una fracción fija de tiempo, sin depender de ningún detector ni de sus parámetros. En la literatura técnica este enfoque se denomina análisis de ***dwell time*** (tiempo de permanencia de la mirada) mediante **análisis *frame-by-frame***: en lugar de detectar fijaciones, se clasifica directamente cada muestra de *gaze* según el área de interés en la que cae, y el tiempo acumulado en cada área es la medida de resultado (Vansteenkiste et al., 2015). Este enfoque es especialmente adecuado en contextos naturalistas con eye-trackers portátiles, donde los algoritmos de detección de fijaciones son menos robustos (Land & Hayhoe, 2001). Los conteos resultantes son **comparables entre participantes y reproducibles**.

---

## 5. Salidas (lo que se obtiene)

Todo se guarda en `<output_root>/<topic>/<id>/` (por ejemplo `OutputData/gaze/044/`), según el nombre del participante. Por defecto tanto los datos de entrada como la salida viven en el disco externo (`/media/quique/EXTERNAL_USB1/BusquedaVisualAnalysis/`, carpetas `InputData/` y `OutputData/`); ambas rutas pueden cambiarse con los argumentos `--data_root`/`--output_root` o las variables de entorno `EEHA_DATA_ROOT`/`EEHA_OUTPUT_ROOT`. Para procesar varios participantes en paralelo existe `run_all.py` (descubre los participantes del directorio de datos y limita los procesos simultáneos y los hilos de OpenCV de cada uno). Se generan varios ficheros con la **misma información en distintos formatos**:

| Fichero | Para qué sirve |
|---|---|
| **`trials_data_<id>.csv`** | **Resumen por *trial*.** *(base de los análisis)* |
| **`trials_data_<id>_sequence.csv`** | **Recorrido temporal de la mirada.** *(base de los análisis)* |
| `data_<id>.yaml` | Todos los datos en formato legible por humanos. |
| `data_<id>.pkl` | Los mismos datos, en formato para volver a cargarlos rápido desde Python. |
| `result_log_<id>.txt` | Un informe de texto con tablas-resumen (reparto de tiempo y de miradas, y una tabla por *trial*). |
| `debug_<id>.mp4` | (Solo si se activa la visualización) Vídeo de comprobación con todo dibujado encima. |

### ¿Qué hay en el YAML/PKL que no esté en los CSV?

Los CSV bastan para casi todo, pero el `data_<id>.yaml` (y su equivalente `.pkl`) guarda algunos campos extra que pueden ser útiles si un análisis se queda corto:

- **`target_cord` y `target_norm_coord`**: la **posición del objetivo** en el tablero (la casilla y su coordenada normalizada). Permite, por ejemplo, medir la distancia de la mirada al objetivo o si llegó a mirarse. **No está en los CSV.**
- **`init_capture` y `end_capture`**: los **números de *frame*** exactos de inicio y fin de cada *trial* (los CSV solo dan la duración en segundos).
- **`video_fps`**: la frecuencia del vídeo de *World* usada para los cálculos.
- **`frames_info` y `fixations_info`**: el reparto de *frames* y de muestras de *gaze* por fase (también resumido en `result_log_<id>.txt`).

---

## 6. Cómo se leen los dos CSV (lo esencial)

Primero, dos conceptos que aparecen en ambos:

- **Ficha vs. hueco.** Cada casilla del tablero o bien tiene una **ficha** física de un color y forma, o bien es un **hueco vacío** reservado para una ficha de ese tipo. Un mismo "color + forma" puede estar en el tablero **como ficha en un sitio y como hueco vacío en otro**. Los CSV distinguen las dos cosas: no es lo mismo mirar la ficha que mirar su hueco.
- **`not_board`.** Cuando la mirada cae **fuera del tablero**, se registra con color y forma `not_board`.

### 6.1. `trials_data_<id>.csv` — resumen por *trial*

Cada fila resume **cuánto miró la persona a un color+forma concreto durante un *trial***. Hay una fila por cada combinación de color y forma que recibió alguna mirada en ese *trial*.

| Columna | Significado |
|---|---|
| `block_index` | Número de *block* (0–5). |
| `trial_index` | Número de *trial* dentro del *block* (0–9). |
| `trial_name` | **Objeto que se estaba buscando** en ese *trial*, p. ej. `red_hexagon` (color_forma). |
| `Color` | Color de la casilla mirada (`red`, `green`, `blue`, `yellow`) o `not_board`. |
| `Shape` | Forma de la casilla mirada (`circle`, `hexagon`, `triangle`, `square`, `trapezoid`) o `not_board`. |
| `Piece Fixations` | Nº de muestras de *gaze* que cayeron sobre la **ficha** de ese color+forma. |
| `Slot only Fixations` | Nº de muestras de *gaze* que cayeron sobre el **hueco vacío** de ese color+forma. |
| `trial_duration_s` | **Duración del *trial* en segundos** (tiempo real). Se repite en todas las filas del *trial*. |
| `Finish Status` | Cómo terminó el *trial*: `test_finish_execution` = terminó con normalidad (oclusión de la mano detectada); `test_finish_by_next_panel` = la oclusión no llegó a detectarse y el *trial* se cerró en el último *frame* con tablero visible — la duración es una **cota superior** (ver apartado 3); `test_finish_by_end_of_video` = *trial* cerrado porque la grabación terminó (típicamente el último *trial*). Otro valor indica final anómalo o problemas de procesamiento. |

**Ejemplo real** (participante 044, *block* 1, *trial* 0, buscando el **hexágono rojo**):

```
block,trial,trial_name,Color,Shape,Piece Fix.,Slot Fix.,dur_s,status
1,0,red_hexagon,green,hexagon,20,0,1.64,test_finish_execution
1,0,red_hexagon,red,trapezoid,31,0,1.64,test_finish_execution
1,0,red_hexagon,red,square,44,0,1.64,test_finish_execution
1,0,red_hexagon,red,triangle,0,20,1.64,test_finish_execution
```

Se lee así: durante este *trial* (que duró 1,64 s), la persona miró **20 veces la ficha del hexágono verde**, **31 veces la ficha del trapecio rojo**, **44 veces la ficha del cuadrado rojo** y **20 veces el hueco vacío del triángulo rojo** (0 veces a su ficha). Es decir, exploró sobre todo piezas rojas (el color que buscaba) antes de encontrar el objetivo.

### 6.2. `trials_data_<id>_sequence.csv` — recorrido temporal

Cada fila es **una muestra de *gaze* individual**, en orden cronológico. Sirve para reconstruir el **recorrido** de la mirada por el tablero a lo largo del *trial*.

| Columna | Significado |
|---|---|
| `block_index` | Número de *block* (0–5). |
| `trial_index` | Número de *trial* dentro del *block* (0–9). |
| `trial_name` | Objeto buscado en ese *trial*. |
| `Color` | Color de la casilla donde cayó esta mirada (o `not_board`). |
| `Shape` | Forma de esa casilla (o `not_board`). |
| `Piece=1/Slot=0` | Si esa casilla es **ficha** o **hueco vacío**. En el CSV aparece como `True`/`False` (pese a que la cabecera diga `1/0`). |
| `Frame_N` | **Número de imagen (*frame*) del vídeo de *World*** en la que se registró esta mirada. |
| `trial_duration_s` | Duración total del *trial* en segundos (se repite). |
| `Board Coord` | **Casilla del tablero** en forma `[columna, fila]` (`[-1, -1]` si cayó fuera). |
| `Board norm Coord` | Posición **dentro del tablero** como `[x, y]` entre 0 y 1 (0,0 = esquina superior izquierda; 1,1 = inferior derecha). Útil para dibujar el recorrido sobre la imagen del tablero. |
| `Finish Status` | Cómo terminó el *trial* (igual que arriba). |

**Ejemplo real** (mismas primeras miradas del *trial* anterior):

```
block,trial,trial_name,Color,Shape,Pieza?,Frame,dur_s,Casilla,PosNorm,status
1,0,red_hexagon,green,hexagon,True,169,1.64,"[5, 3]","[0.63, 0.62]",test_finish_execution
1,0,red_hexagon,green,hexagon,True,169,1.64,"[5, 3]","[0.63, 0.62]",test_finish_execution
1,0,red_hexagon,green,hexagon,True,170,1.64,"[5, 3]","[0.63, 0.62]",test_finish_execution
```

Se lee así: en la imagen **169** del vídeo, la mirada estaba en la casilla **[columna 5, fila 3]**, que es la **ficha del hexágono verde**, situada hacia el centro-derecha del tablero. Que haya **varias filas con el mismo `Frame_N` (169)** es justo lo explicado en el apartado 4: a una misma imagen del vídeo le corresponden varias muestras de *gaze* porque la mirada se registra más rápido que el vídeo.

---

## 7. Del conteo al tiempo: cuánto se mira cada cosa

La idea central para los análisis es sencilla: **como el *gaze* se muestrea a una frecuencia conocida y constante, contar muestras de *gaze* equivale a medir tiempo.** (Cuando en esta guía hablamos de "muestras" o de los conteos del CSV, siempre nos referimos a **muestras de *gaze***.) Cada muestra de *gaze* "representa" siempre la misma fracción de tiempo: `1 / f_gaze` segundos, donde `f_gaze` es la frecuencia de muestreo del *gaze* (≈ 200 muestras por segundo). Por tanto:

```
tiempo mirando algo  ≈  (nº de muestras de gaze sobre eso)  ×  (1 / f_gaze)
```

Esos "nº de muestras" son justo los conteos del CSV resumen: las columnas `Piece Fixations` y `Slot only Fixations`. Por ejemplo, las **44 muestras** sobre la ficha del cuadrado rojo del ejemplo anterior, a 200 Hz (frecuencia nominal del Pupil Labs Core), equivalen a `44 / 200 = 0,22 segundos` mirando esa ficha. Dado que parte de las muestras se descartan por baja confianza, el número de muestras válidas por segundo varía entre participantes; si se quiere precisión máxima, se puede usar el recuento real de muestras válidas de cada participante (ver sección 10) como denominador efectivo.

Importante: la frecuencia que importa aquí es la del *gaze* (`f_gaze`), **no** la del vídeo de *World*. Son dos relojes distintos (apartado 4): la **duración del *trial*** se mide con el vídeo de *World* (`frames / f_world`), y el **tiempo mirando cada cosa** se mide con las muestras de *gaze* (`muestras / f_gaze`).

### No todos los instantes tienen dato

Es importante **no esperar que haya dato de mirada para todo el tiempo del *trial***. Pueden darse tres situaciones:

- **Sin dato.** La máquina no registró *gaze* en ese instante, o la muestra se descartó por **baja confianza** (≤ 0,6). Ese tiempo simplemente **no se contabiliza** (no sabemos a dónde miraba).
- **Fuera del tablero.** Hay muestra, pero la mirada cae fuera del tablero: se cuenta aparte, como `not_board` (es "mirando fuera", no se pierde).
- **Sobre una casilla.** Se cuenta en el color+forma correspondiente.

**Consecuencia:** la suma de los tiempos de todas las casillas (más `not_board`) **no tiene por qué coincidir con la duración total del *trial***, porque hay huecos sin dato. Los tiempos por casilla son, por tanto, **tiempo de mirada efectivamente medido**, no un reparto del 100 % del *trial*.

---

## 8. Notas importantes para quien analice los CSV

- **Los conteos son de muestras de *gaze*, no de imágenes de vídeo.** Por la diferencia de velocidad (apartados 4 y 7), un *trial* puede tener más muestras que *frames*; la **duración en segundos** sí es tiempo real.
- **La `trial_duration_s` incluye el tiempo de respuesta motora.** El *trial* termina cuando la mano entra en el tablero (último *frame* con el tablero visible; el tiempo de confirmación de la oclusión ya no se incluye), no cuando se toma la decisión de búsqueda. Ver la subsección "Cómo se detecta el inicio y el fin de cada *trial*" en el apartado 3.
- ***Trials* con error.** Algunos *trials* no salen bien (el participante no pasó por la secuencia esperada, o hubo un salto entre paneles). Se reconocen porque el `trial_name` empieza por `missing_trial_error_` o `transition_error_`, y suelen tener duración 0 o estado anómalo. **Conviene filtrarlos antes de analizar.** Hay dos scripts de comprobación: `src/tools/check_correct_output.py` (avisa de carpetas con ficheros faltantes o con muchos errores) y `src/tools/check_correct_trials.py` (compara, participante a participante, los *trials* detectados frente a la secuencia esperada).
- **De CSV a gráficos.** Hay scripts que dibujan el recorrido de la mirada sobre la imagen del tablero (`src/tools/project_paths.py`, `src/tools/data_analysis.py`) y un vídeo con la mirada superpuesta sobre la grabación original (`src/tools/project_data.py`).

---

## 9. Limitaciones metodológicas

### Del diseño experimental

- **Delimitación temporal del *trial*.** El inicio y el fin del *trial* se detectan a partir de la visibilidad del tablero en el vídeo, no de un marcador cognitivo directo. El período registrado incluye un posible retardo al inicio y termina en la respuesta motora, no en el momento de la decisión. Es la única marca temporal sistemática disponible; su ventaja es que es consistente entre participantes. Ver apartado 3 para el detalle.
- **Uso de *gaze* en lugar de *fixations*.** La elección de *gaze* como unidad de medida implica trabajar con muestras crudas en lugar de eventos perceptivos interpretados. El tradeoff entre replicabilidad y granularidad perceptiva se discute en el apartado 4.

### Del equipo de medida (Pupil Labs Core)

El dispositivo empleado es el **Pupil Labs Core** (Kassner, Patera & Bulling, 2014), un eye-tracker binocular portátil de código abierto. Sus especificaciones técnicas relevantes son: cámaras oculares a **200 Hz** (192×192 px), cámara de escena (*World*) a **1080p/30 Hz**, exactitud de **0,60°** y precisión de **0,02°** (ambas con calibración y en condiciones de laboratorio).

- **Error angular y su traducción espacial.** La exactitud de 0,60° se obtiene en condiciones controladas; en uso naturalista (movimiento de cabeza, variabilidad de iluminación, tarea real) el error puede ser mayor. Este error angular se traduce en una **incertidumbre espacial** al asignar la mirada a una casilla del tablero: cuanto mayor sea la distancia participante–tablero y más pequeñas sean las casillas, más probable es que una muestra de *gaze* quede asignada a la casilla adyacente en lugar de la correcta.
- **Calidad de la señal y pérdida de datos.** En momentos de parpadeo, movimiento brusco o reflejo corneal desfavorable, el aparato puede no registrar *gaze* o registrarlo con baja confianza. Las muestras con confianza ≤ 0,6 se descartan (ver apartado 4); esos fragmentos no se contabilizan. Aunque la frecuencia nominal del dispositivo es 200 Hz, el número efectivo de muestras válidas varía entre participantes (ver tabla en apartado 10), lo que refleja diferencias en la calidad de la señal y no diferencias de hardware.
- **Características oculares individuales como variable no controlada.** El algoritmo de Pupil Labs Core detecta la pupila mediante técnica de pupila oscura sobre la imagen infrarroja del ojo. El color del iris, la forma del párpado, la presencia de pliegue epicántico u otras características anatómicas individuales influyen en la facilidad y precisión de esa detección. Estas diferencias no están controladas en el diseño experimental y pueden explicar parte de la variación en número de muestras válidas y en precisión entre participantes.
- **Deriva de la calibración.** La calibración se realiza al inicio de la sesión. Si las gafas se desplazan levemente durante el experimento —algo habitual en tareas naturalistas con movimiento de cabeza—, la correspondencia entre la dirección de mirada estimada y la posición real en la imagen puede degradarse progresivamente. El dispositivo incorpora compensación de deslizamiento (*slippage compensation*) mediante el modelo 3D del ojo, pero no elimina por completo este efecto.

---

## 10. Magnitud del procesamiento

Las tablas siguientes recogen los datos de los 20 participantes procesados, para ilustrar el volumen de información analizado de forma automática.

### Resumen global

| | |
|---|---|
| Participantes | 20 |
| Duración total de vídeo | 2 h 39 min |
| *Frames* de *World* (vídeo completo) | 285.541 |
| Muestras de *gaze* (confianza > 0,6) | 990.081 |
| *Trials* válidos segmentados automáticamente | 1.109 de 1.200 esperados (92 %) |

### Desglose por participante

| Participante | Duración (min:s) | *Frames* World | FPS World | Muestras *gaze* ¹ | *Trials* válidos |
|---|---|---|---|---|---|
| 001 | 9:04 | 16.233 | 29,79 | 49.027 | 58 |
| 002 | 9:35 | 17.140 | 29,78 | 45.853 | 60 |
| 007 | 10:08 | 18.111 | 29,76 | 98.515 | 60 |
| 007_1 | 7:30 | 13.423 | 29,82 | 42.562 | 60 |
| 008 | 9:28 | 16.935 | 29,81 | 33.890 | 50 |
| 009 | 9:30 | 16.994 | 29,81 | 86.786 | 60 |
| 011 | 9:19 | 16.667 | 29,80 | 41.108 | 60 |
| 012 | 7:55 | 14.195 | 29,84 | 43.477 | 59 |
| 024 | 6:57 | 12.448 | 29,79 | 38.217 | 60 |
| 027 | 7:21 | 13.161 | 29,82 | 40.878 | 60 |
| 032 | 6:48 | 12.194 | 29,82 | 37.085 | 56 |
| 035 | 7:05 | 12.697 | 29,84 | 43.707 | 49 |
| 042 | 7:17 | 13.068 | 29,84 | 43.642 | 50 |
| 044 | 5:56 | 10.612 | 29,79 | 30.849 | 50 |
| 049 | 5:43 | 10.254 | 29,84 | 33.582 | 60 |
| 051 | 6:42 | 12.013 | 29,81 | 38.438 | 50 |
| 054 | 9:49 | 17.579 | 29,80 | 51.069 | 48 |
| 055 | 7:19 | 13.131 | 29,84 | 41.665 | 49 |
| 064 | 5:50 | 10.446 | 29,84 | 35.795 | 60 |
| Ale_005 | 10:12 | 18.240 | 29,76 | 113.936 | 50 |
| **TOTAL** | **159:39** | **285.541** | 29,80 | **990.081** | **1.109** |

¹ Muestras de *gaze* con confianza > 0,6 según el criterio de Pupil Labs. El total pre-filtrado no está disponible al no conservarse los ficheros originales de captura. La variación entre participantes puede reflejar diferencias en calidad de señal y características oculares individuales.

---

## Referencias

Ehinger, B. V., Groß, K., Ibs, I., & König, P. (2019). A new comprehensive eye-tracking test battery concurrently evaluating the Pupil Labs glasses and the EyeLink 1000. *PeerJ*, *7*, e7086. https://doi.org/10.7717/peerj.7086

Kassner, M., Patera, W., & Bulling, A. (2014). Pupil: An open source platform for pervasive eye tracking and mobile gaze-based interaction. En *Adjunct Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing* (pp. 1151–1160). ACM. https://doi.org/10.1145/2638728.2641695

Land, M. F., & Hayhoe, M. (2001). In what ways do eye movements contribute to everyday activities? *Vision Research*, *41*(25–26), 3559–3565. https://doi.org/10.1016/s0042-6989(01)00102-x

Pupil Labs. (2024). *Pupil Player documentation*. https://docs.pupil-labs.com/core/software/pupil-player/

Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols. En *Proceedings of the Eye Tracking Research and Applications Symposium* (pp. 71–78). ACM. https://doi.org/10.1145/355017.355028

Vansteenkiste, P., Cardon, G., Philippaerts, R., & Lenoir, M. (2015). Measuring dwell time percentage from head-mounted eye-tracking data: Comparison of a frame-by-frame and a fixation-by-fixation analysis. *Ergonomics*, *58*(5), 712–721. https://doi.org/10.1080/00140139.2014.990524
