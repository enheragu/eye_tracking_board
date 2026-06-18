# Recomendaciones de diseño experimental (para facilitar el procesado)

Lecciones aprendidas procesando esta cohorte (eye-tracking + tablero físico). Cada punto
parte de un **caso real** observado en estos datos y propone qué cambiar en una futura toma
para que el procesado sea más robusto y necesite menos excepciones manuales. No invalida los
datos actuales: son mejoras para **próximas grabaciones**.

---

**Índice**

- [1. Marcadores ArUco](#1-marcadores-aruco)
- [2. Protocolo de presentación de paneles](#2-protocolo-de-presentación-de-paneles)
- [3. Registro de la sesión (lo más rentable)](#3-registro-de-la-sesión-lo-más-rentable)
- [4. Cámara y grabación](#4-cámara-y-grabación)
- [5. Iluminación y contraste](#5-iluminación-y-contraste)
- [6. Detección de la fase motora (limitación de método)](#6-detección-de-la-fase-motora-limitación-de-método)
- [7. Precisión de la mirada y compensación del error](#7-precisión-de-la-mirada-y-compensación-del-error)
- [8. Mejoras de procesado: implementadas y pendientes](#8-mejoras-de-procesado-implementadas-y-pendientes)
  - [Resumen accionable](#resumen-accionable)

---

## 1. Marcadores ArUco

**Más marcadores por panel.** Algunos paneles llevan un solo ArUco (p. ej. `blue_circle` =
id 34). Con un único marcador no se puede definir el cuadrilátero del panel, así que su
polígono se estima y sale torcido/desplazado (la identidad del panel sigue siendo
correcta —viene del id—, pero la clasificación "mirada sobre el panel" pierde precisión).
→ **Usar ≥2, idealmente 4 marcadores por panel** (uno por esquina): polígono limpio y robusto a
oclusión parcial de la mano al sostenerlo.

**Marcadores redundantes en el tablero y lejos de los bordes de imagen.** Al desdistorsionar
con `alpha=0` los marcadores muy pegados al borde del fotograma se salían del encuadre y se
perdían (medido: participante 042 perdía ~5 marcadores en 16/16 fotogramas), debilitando la
homografía. Se resolvió detectándolos en la imagen original antes de desdistorsionar, pero
el margen es ajustado. → **Distribuir marcadores con redundancia** y evitar colocarlos donde la
cabeza/encuadre los deje en el borde extremo; cuantos más marcadores válidos, mejor la
estimación de pose y la rejilla.

**Marcadores grandes y de diccionario amplio.** Marcadores pequeños o de pocos bits se confunden
más con texturas/sombras de la imagen. → **Marcadores grandes, alto contraste, diccionario con
margen de bits** (menos falsos positivos/negativos).

**Orientación inequívoca del tablero.** Hubo que tratar la rotación de 180° del tablero con
heurísticas (solo votan los marcadores del tablero, con histéresis). → Un **patrón asimétrico**
(un marcador distinto en una esquina, o una marca visual) hace la orientación inequívoca y
elimina esa ambigüedad.

## 2. Protocolo de presentación de paneles

**Cada panel UNA sola vez, sin re-presentaciones.** El caso de P-A: un `yellow_hexagon`
re-presentado justo tras el primer trial del bloque 3 (dos veces seguidas) hizo que el segundo
se emparejara con un trial posterior (en concreto con el trial 9 del bloque 4) y desincronizara
toda la secuencia a partir de ahí; se recuperó con una excepción de config que consume el panel
duplicado (un `[-1, "yellow_hexagon"]`). → **Mostrar cada panel una vez**; si por lo que sea se
re-muestra, anotarlo en un registro de sesión para que el procesado lo descarte.

**Dejar el tablero despejado entre paneles.** Cuando dos paneles se muestran muy seguidos, la
ventana de "tablero limpio" entre ambos puede quedar por debajo del umbral de confirmación de
inicio de trial (`board_contour_start_confirm_threshold = 6` fotogramas en la máquina de
estados): si el tablero no se ve despejado durante al menos ese número de fotogramas, el inicio
del siguiente trial no se confirma limpiamente. → **Dejar un margen claro** (tablero completamente
visible y despejado ~0,5 s) entre retirar un panel y mostrar el siguiente.

**Paneles de demo/práctica distinguibles.** Las demos previas al test (que aquí se marcan con
`-1` por participante: 008, 027, 035, 055, P-A…) se confunden con trials reales si no se
distinguen. → **Marcarlas** (un marcador propio, una posición fija, o un registro) para
descartarlas automáticamente sin desincronizar la secuencia.

**Panel de un color uniforme que cubra el tablero.** El procesado clasifica como "mirada sobre el
panel" (fuera de casilla, §7.5 de la [documentación técnica](documentacion_tecnica.md)) el gaze
que cae sobre una superficie plano-blanca que tapa la celda. En esta toma el panel a veces es
una carta blanca sobre un cartón anaranjado: el respaldo de color no se detecta como tapado
(no es blanco), así que una mirada sobre él puede contarse como casilla. → **Que el panel sea de un
único color uniforme y que cubra del todo el tablero** mientras se muestra; así "mirada sobre el
panel" se detecta de forma fiable y sin riesgo de confundir el respaldo con una pieza.

## 3. Registro de la sesión (lo más rentable)

Casi todas las excepciones manuales de este procesado (demos, re-presentaciones, reordenamientos
por participante) habrían sido triviales con un **log de sesión**. → **Registrar en cada toma**:
secuencia exacta de paneles mostrada, cuándo se muestra una demo, cualquier re-presentación o
incidencia, e inicio/fin de cada bloque. Con eso, la config de procesado se deriva del log en
vez de reconstruirse mirando el vídeo.

## 4. Cámara y grabación

**Resolución del vídeo de *World*.** Es 1280×720 (~30 fps) para todos. Esa resolución limita
la precisión por celda y la detección de ArUcos pequeños/lejanos. → Si el hardware lo permite,
**grabar a mayor resolución** (1080p+) mejora directamente la proyección mirada→celda y la
robustez ArUco.

**Frecuencia de *gaze* consistente y registrada.** La frecuencia real del *gaze* varía entre
participantes (se observaron ≈124 Hz y ≈248 Hz), por eso se mide empíricamente y no se asume
200 Hz. → **Fijar y registrar** la configuración de muestreo para que sea homogénea entre
participantes.

**Grabaciones completas y verificadas.** Hubo descargas corruptas/truncadas (032) y sesiones con
menos trials de lo esperado (008 con ~49, 044 con ~50). → **Verificar integridad** del vídeo y de
los datos de mirada tras cada sesión, y comprobar que se completaron todos los bloques.

## 5. Iluminación y contraste

El toque es **best-effort** (detección de cambio/oclusión sobre la casilla) y los objetivos
cálidos (rojo/amarillo) son más difíciles porque se parecen al tono de la piel; hizo falta
corrección de color y umbrales por color. → **Iluminación uniforme** (sin sombras ni reflejos
sobre tablero y paneles) y buen contraste pieza/mano facilitan tanto el ArUco como el toque.

## 6. Detección de la fase motora (limitación de método)

`motor_onset` (la mano entra al tablero) se infiere de la pérdida del contorno del tablero;
es un proxy: con una homografía robusta el contorno puede aguantar durante el alcance y la
marca no dispara limpiamente (algunos trials cierran "por panel siguiente"). → Si en el futuro
interesa el instante motor con precisión, ayudaría una **señal de entrada de mano dedicada**
(geometría de alcance consistente, o un sensor) en vez de inferirla de la imagen del tablero.

## 7. Precisión de la mirada y compensación del error

El eye-tracker tiene un **error espacial** (típicamente ~0,5–1°, mayor sin buena calibración). Para
compensarlo, el tablero lleva un margen blanco alrededor que se considera parte de su área: así
una mirada cercana al borde de una casilla sigue cayendo "dentro" y no se pierde. Aun así, miradas
justo en el borde exterior pueden caer fuera de la rejilla y clasificarse `not_board` aunque
fueran dirigidas a una casilla — un corner case que limita los datos (se ve en
`gaze_fuera`: la mirada cae un poco por encima del tablero).

**Formas de compensarlo mejor** (mejoras futuras; requieren reprocesar y validar, por eso no se
han aplicado a esta cohorte):

- **Inercia / suavizado temporal (la vía más prometedora).** El *gaze* se muestrea denso (≈124–248 Hz
  frente a 30 fps de vídeo: varias muestras por fotograma), así que una muestra dudosa tiene
  vecinas temporales muy próximas. Versión robusta y conservadora: para una muestra `not_board` o de
  borde, mirar la vecina anterior y la posterior ya clasificadas con confianza; si ambas coinciden
  en una casilla (o adyacentes), asignar la dudosa a esa casilla (es *jitter*); si discrepan o no
  hay confianza, dejarla donde cayó. La variante por dirección/velocidad (predecir
  la posición esperada de la trayectoria y marcar saltos fuera de eje como error) es más potente pero
  más difícil de hacer robusta (más parámetros, más casos límite); conviene empezar por la de
  acuerdo de vecinos.
- **Corrección de sesgo por participante** (factible *post-hoc*, sin recalibrar). Si el error es
  sistemático (un desplazamiento casi constante), estimarlo —p. ej. el desfase mediano entre la
  mirada y el centro de la casilla más cercana durante la búsqueda, o respecto al objetivo cuando se
  mira— y restarlo. Cautela: solo ayuda si el error es sistemático; si es dispersión aleatoria,
  no. Es comprobable: basta mirar la distribución de desfases antes de aplicarlo. *Variante más
  rigurosa:* usar los instantes de calibración (las marcas de la cartulina al inicio y a mitad
  del vídeo) como verdad-terreno del desfase —e incluso de la deriva entre ambas—. Conviene tener en cuenta que Pupil
  ya aplica su calibración a los datos de *gaze*, así que esto corregiría solo el residual
  posterior (pequeño salvo deriva), y requiere detectar/parsear las marcas y su *timing*: mayor complejidad y
  beneficio incierto. Por eso queda por debajo de la inercia temporal en prioridad.
- **Fijaciones — descartadas.** El *topic* de fijaciones exportado venía con falta de datos: si
  no se parametriza muy bien se pierde granularidad y quedaban casi sin muestras para analizar. Por
  eso se trabaja con *gaze* crudo. (Puede retomarse si se obtiene una exportación de fijaciones más
  densa.)
- **Mejor calibración en la toma.** Sería lo más efectivo (corrige el error de raíz), pero no se
  consiguió mejorar la calibración en estas grabaciones; queda como recomendación para futuras tomas.

**Compromiso:** cuanto más agresiva la compensación (ampliar el margen, asignar por proximidad a la casilla más
cercana), menos miradas se pierden pero más riesgo de contar como casilla una mirada
genuinamente fuera. El criterio actual (estricto + margen blanco) es conservador y transparente:
prefiere perder algún borde antes que inventar una casilla.

## 8. Mejoras de procesado: implementadas y pendientes

Varias de las mejoras planteadas están **implementadas** (detalle en el
[CHANGELOG](../CHANGELOG.md) y la [documentación técnica](documentacion_tecnica.md)); el resto
queda **pendiente** para versiones posteriores.

**Implementado:**

- **Compensación del error de mirada** (apartado 7): recalibraciones no registradas recuperadas
  desde los paneles del vídeo (matriz 9 puntos) → deriva por participante corregida con
  compuerta de validación cruzada (bootstrap, solo adopta si la ganancia es fiablemente
  positiva — nunca empeora); + exclusión de parpadeos y suavizado guiado por velocidad en
  el dataloader (misma frecuencia, sin tocar sacadas). Todo se aplica en carga y propaga a
  conteo, marcas y figuras. Mecanismo en [§7](documentacion_tecnica.md); ganancias por
  participante en [caracterizacion_dataset.md](caracterizacion_dataset.md).
- **Incertidumbre de la mirada por muestra** (apartado 7; [§7.4-7.5](documentacion_tecnica.md)):
  cada gaze lleva una covarianza 2×2 medida en la calibración (jitter + bias/drift + factores de
  confianza y espacial), propagada por el suavizador (varianza inversa). De ahí un reparto de
  probabilidad sobre casillas (`cell_dist`/`onboard_mass`) y un `target_found_confidence`
  graduado [0,1]: la mirada deja de ser un punto y pasa a ser una distribución, honesta con que
  una casilla está cerca del límite de resolución (~½ casilla).
- **`target_found` por masa de la elipse sobre una fijación**: se dispara en la primera fijación
  I-DT (dispersión en ventana sobre la mirada corregida) cuya masa de incertidumbre sobre la casilla
  objetivo alcanza `target_found_mass_threshold` (0,30; §7.5), no en el primer paso fugaz ni por
  mayoría de centroides en la celda exacta — reduce el falso positivo del "paso por encima" y cuenta
  como encontrada una mirada en la frontera del objetivo (dentro del error del aparato); el falso
  negativo (error del tracker) se mitiga con la corrección de deriva.
- **Marcas motoras por la CURVA de oclusión (modelo bump)** — *completo*: `motor_onset` /
  `target_touch` / `hand_exit` se re-derivan post-hoc del `signal_trace` (subida → pico → valle
  de `fT` y `board_occ`); `motor_onset` validado por oclusión (un corte de contorno sin subida
  es artefacto de borde/homografía y se mueve a la subida real); congruencia relajada + `reach_style`.
  La máquina en vivo sigue causal (segmenta); la curva solo refina.
- **Anomalía `off_target`** ([§12.2](documentacion_tecnica.md)): un alcance completo (mano entra y
  sale) sin tocar ni mirar el objetivo → "fue a otro sitio", con la pieza mirada como pista.
  Señal fiable; localizar la pieza tocada por oclusión queda experimental (no separa dedo de
  brazo) → ver pendientes.

**Pendiente:**

- **Detector de pieza tocada robusto**: correr el detector de toque del target por celda
  (compuertas color/edge/SSIM + separación local + sostenido) para localizar qué pieza se tocó en
  un error — la oclusión barata coge el brazo, no el dedo.
- **Config desde log de sesión**: derivar las excepciones por participante (demos, re-presentaciones)
  de un registro de la toma, en vez de a mano (P-A, 001). Ya hay un chequeo de seguridad que
  avisa de excepciones que descartan un trial real sin documentar, pero no la auto-derivación.
- **Unificar el detector de fijación/velocidad** (hoy calculado en varios sitios) y reabrir
  fijaciones / mayor resolución si se consigue una exportación más densa (apartado 7/4).
- **`SSIM` perezoso** para acelerar el toque: descartado — sin mejora, el cuello de botella es la
  detección de ArUcos (color-correction + `detectMarkers`), no el SSIM.

---

### Resumen accionable

| Prioridad | Cambio | Por qué (caso real) |
|---|---|---|
| Alta | **Log de sesión** (secuencia, demos, incidencias) | evita casi todas las excepciones manuales (P-A, 001, demos) |
| Alta | **≥4 ArUcos por panel** | `blue_circle` con 1 marcador → polígono malo |
| Alta | **Tablero despejado entre paneles** | cadencia apretada → ventana limpia < 6 fotogramas, inicio de trial sin confirmar |
| Media | **Marcadores redundantes, no en el borde** | 042 perdía marcadores al desdistorsionar |
| Media | **Mayor resolución de vídeo** | 720p limita precisión por celda y ArUco |
| Media | **Verificar integridad de cada grabación** | 032 truncado; sesiones incompletas |
| Baja | **Orientación de tablero inequívoca** | hubo que tratar la rotación 180° |
