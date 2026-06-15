# Recomendaciones de diseño experimental (para facilitar el procesado)

Lecciones aprendidas procesando esta cohorte (eye-tracking + tablero físico). Cada punto
parte de un **caso real** observado en estos datos y propone qué cambiar en una futura toma
para que el procesado sea más robusto y necesite menos excepciones manuales. No invalida los
datos actuales: son mejoras para **próximas grabaciones**.

---

## 1. Marcadores ArUco

**Más marcadores por panel.** Algunos paneles llevan **un solo ArUco** (p. ej. `blue_circle` =
id 34). Con un único marcador no se puede definir el cuadrilátero del panel, así que su
polígono se **estima** y sale torcido/desplazado (la **identidad** del panel sigue siendo
correcta —viene del id—, pero la clasificación "mirada sobre el panel" pierde precisión).
→ **Usar ≥2, idealmente 4 marcadores por panel** (uno por esquina): polígono limpio y robusto a
oclusión parcial de la mano al sostenerlo.

**Marcadores redundantes en el tablero y lejos de los bordes de imagen.** Al desdistorsionar
con `alpha=0` los marcadores muy pegados al borde del fotograma se salían del encuadre y se
**perdían** (medido: participante 042 perdía ~5 marcadores en 16/16 fotogramas), debilitando la
homografía. Se resolvió detectándolos en la imagen **original** antes de desdistorsionar, pero
el margen es ajustado. → **Distribuir marcadores con redundancia** y evitar colocarlos donde la
cabeza/encuadre los deje en el borde extremo; cuantos más marcadores válidos, mejor la
estimación de pose y la rejilla.

**Marcadores grandes y de diccionario amplio.** Marcadores pequeños o de pocos bits se confunden
más con texturas/sombras de la imagen. → **Marcadores grandes, alto contraste, diccionario con
margen de bits** (menos falsos positivos/negativos).

**Orientación inequívoca del tablero.** Hubo que tratar la **rotación de 180°** del tablero con
heurísticas (solo votan los marcadores del tablero, con histéresis). → Un **patrón asimétrico**
(un marcador distinto en una esquina, o una marca visual) hace la orientación inequívoca y
elimina esa ambigüedad.

## 2. Protocolo de presentación de paneles

**Cada panel UNA sola vez, sin re-presentaciones.** El caso de Vane: un `yellow_hexagon`
**re-presentado** al inicio del bloque 3 (dos veces seguidas) hizo que el segundo se emparejara
con un trial posterior, **desincronizó toda la secuencia** y se perdieron ~25 trials (se
recuperaron con una excepción de config). → **Mostrar cada panel una vez**; si por lo que sea se
re-muestra, **anotarlo** en un registro de sesión para que el procesado lo descarte.

**Dejar el tablero despejado entre paneles.** El caso de 001 (bloque 4): paneles mostrados muy
seguidos, con ventanas de "tablero limpio" de **<6 fotogramas** (el umbral de confirmación),
hizo que 3 trials no se segmentaran limpiamente. → **Dejar un margen claro** (tablero
completamente visible y despejado ~0,5 s) entre retirar un panel y mostrar el siguiente.

**Paneles de demo/práctica distinguibles.** Las demos previas al test (que aquí se marcan con
`-1` por participante: 008, 027, 035, 055, Vane…) se confunden con trials reales si no se
distinguen. → **Marcarlas** (un marcador propio, una posición fija, o un registro) para
descartarlas automáticamente sin desincronizar la secuencia.

## 3. Registro de la sesión (lo más rentable)

Casi todas las excepciones manuales de este procesado (demos, re-presentaciones, reordenamientos
por participante) habrían sido triviales con un **log de sesión**. → **Registrar en cada toma**:
secuencia exacta de paneles mostrada, cuándo se muestra una demo, cualquier re-presentación o
incidencia, e inicio/fin de cada bloque. Con eso, la config de procesado se deriva del log en
vez de reconstruirse mirando el vídeo.

## 4. Cámara y grabación

**Resolución del vídeo de *World*.** Es **1280×720** (~30 fps) para todos. Esa resolución limita
la precisión por celda y la detección de ArUcos pequeños/lejanos. → Si el hardware lo permite,
**grabar a mayor resolución** (1080p+) mejora directamente la proyección mirada→celda y la
robustez ArUco.

**Frecuencia de *gaze* consistente y registrada.** La frecuencia real del *gaze* **varía entre
participantes** (se observaron ≈124 Hz y ≈248 Hz), por eso se mide empíricamente y no se asume
200 Hz. → **Fijar y registrar** la configuración de muestreo para que sea homogénea entre
participantes.

**Grabaciones completas y verificadas.** Hubo descargas corruptas/truncadas (032) y sesiones con
menos trials de lo esperado (008 con ~49, 044 con ~50). → **Verificar integridad** del vídeo y de
los datos de mirada tras cada sesión, y comprobar que se completaron todos los bloques.

## 5. Iluminación y contraste

El toque es **best-effort** (detección de cambio/oclusión sobre la casilla) y los objetivos
**cálidos** (rojo/amarillo) son más difíciles porque se parecen al tono de la piel; hizo falta
corrección de color y umbrales por color. → **Iluminación uniforme** (sin sombras ni reflejos
sobre tablero y paneles) y **buen contraste** pieza/mano facilitan tanto el ArUco como el toque.

## 6. Detección de la fase motora (limitación de método)

`motor_onset` (la mano entra al tablero) se infiere de la **pérdida del contorno** del tablero;
es un **proxy**: con una homografía robusta el contorno puede aguantar durante el alcance y la
marca no dispara limpiamente (algunos trials cierran "por panel siguiente"). → Si en el futuro
interesa el instante motor con precisión, ayudaría una **señal de entrada de mano dedicada**
(geometría de alcance consistente, o un sensor) en vez de inferirla de la imagen del tablero.

## 7. Precisión de la mirada y compensación del error

El eye-tracker tiene un **error espacial** (típicamente ~0,5–1°, mayor sin buena calibración). Para
compensarlo, el tablero lleva un **margen blanco** alrededor que se considera parte de su área: así
una mirada cercana al borde de una casilla sigue cayendo "dentro" y no se pierde. Aun así, miradas
**justo en el borde exterior** pueden caer fuera de la rejilla y clasificarse `not_board` aunque
fueran dirigidas a una casilla — un **corner case** que limita los datos (se ve en
`gaze_fuera`: la mirada cae un pelín por encima del tablero).

**Formas de compensarlo mejor** (mejoras futuras; requieren reprocesar y **validar**, por eso no se
han aplicado a esta cohorte):

- **Inercia / suavizado temporal (la vía más prometedora).** El *gaze* se muestrea denso (≈124–248 Hz
  frente a 30 fps de vídeo: varias muestras por fotograma), así que una muestra **dudosa** tiene
  vecinas temporales muy próximas. Versión robusta y conservadora: para una muestra `not_board` o de
  borde, mirar la **vecina anterior y la posterior** ya clasificadas con confianza; si **ambas coinciden**
  en una casilla (o adyacentes), asignar la dudosa a esa casilla (es *jitter*); si **discrepan** o no
  hay confianza, **dejarla donde cayó**. Es lo que se describe coloquialmente como "si venía clara de un
  sitio o iba clara a un sitio, los dudosos van ahí". La variante por **dirección/velocidad** (predecir
  la posición esperada de la trayectoria y marcar saltos fuera de eje como error) es más potente pero
  **más difícil de hacer robusta** (más parámetros, más casos límite); conviene empezar por la de
  acuerdo de vecinos.
- **Corrección de sesgo por participante** (factible *post-hoc*, sin recalibrar). Si el error es
  **sistemático** (un desplazamiento casi constante), estimarlo —p. ej. el desfase mediano entre la
  mirada y el centro de la casilla más cercana durante la búsqueda, o respecto al objetivo cuando se
  mira— y **restarlo**. **Cautela:** solo ayuda si el error es sistemático; si es dispersión aleatoria,
  no. Es **comprobable**: basta mirar la distribución de desfases antes de aplicarlo. *Variante más
  principista:* usar los **instantes de calibración** (las marcas de la cartulina al inicio y a mitad
  del vídeo) como **verdad-terreno** del desfase —e incluso de la **deriva** entre ambas—. Ojo: Pupil
  **ya aplica** su calibración a los datos de *gaze*, así que esto corregiría solo el **residual**
  posterior (pequeño salvo deriva), y requiere detectar/parsear las marcas y su *timing*: más lío,
  beneficio incierto. Por eso queda **por debajo** de la inercia temporal en prioridad.
- **Fijaciones — *probado y descartado*.** El *topic* de fijaciones exportado venía con **falta de
  datos**: si no se parametriza muy bien se pierde granularidad y quedaban casi sin muestras para
  analizar. Por eso se trabaja con *gaze* crudo. (Reabrible si se consigue una exportación de
  fijaciones más densa.)
- **Mejor calibración en la toma.** Sería lo más efectivo (corrige el error de raíz), pero **no se
  consiguió** mejorar la calibración en estas grabaciones; queda como recomendación para futuras tomas.

**Compromiso:** cuanto más agresiva la compensación (ampliar el margen, "imantar" a la casilla más
cercana), menos miradas se pierden **pero** más riesgo de contar como casilla una mirada
genuinamente fuera. El criterio actual (estricto + margen blanco) es **conservador y transparente**:
prefiere perder algún borde antes que inventar una casilla.

## 8. Mejoras de procesado para versiones futuras (post v1.2.0)

Ideas para una versión posterior (v1.3+), una vez **fijada la v1.2.0**:

- **Compensación del error de mirada** (apartado 7): empezar por la **inercia temporal** (acuerdo de
  vecinos, conservadora); como complemento, **corrección de sesgo** por participante si la distribución
  de desfases resulta sistemática. Validar contra la cobertura actual antes de adoptarlo (no debe
  "inventar" casillas).
- **Marca "objetivo visto" (`target_found`) más fiable.** Hoy es el **primer gaze sobre la celda
  objetivo exacta**, con dos sesgos: (a) un **paso fugaz** de la mirada *por encima* del objetivo de
  camino a otro sitio la dispara aunque no lo "vieran" de verdad (falso positivo); y (b) el **error del
  eye-tracker** hace que a veces el gaze nunca caiga justo ahí aunque sí lo miraran (falso negativo —
  por eso el 86% es probablemente una **subestimación**: si lo tocan, lo vieron). Mejora: exigir
  **permanencia/fijación** sobre la celda (varias muestras seguidas o usar fijaciones) en vez de un solo
  paso, combinado con la compensación de error de arriba.
- **Detector de toque más rápido** (`SSIM` perezoso): la cobertura de toque subió en v1.2 pero el
  procesado es ~3× más lento; calcular el SSIM **solo** cuando los componentes baratos
  (bordes/oclusión) ya sugieren cambio recortaría tiempo sin perder cobertura.
- **Señal de entrada de mano dedicada**: `motor_onset` se infiere de la **pérdida de contorno** (un
  proxy frágil cuando la homografía es robusta y el contorno aguanta); una señal más directa daría el
  instante motor con más precisión.
- **Config desde log de sesión**: derivar automáticamente las excepciones por participante (demos,
  re-presentaciones) de un **registro de la toma**, en vez de reconstruirlas a mano (Vane, 001).
- **Fijaciones / mayor resolución**: reabrir el uso de fijaciones si se consigue una exportación más
  **densa** (apartado 7); grabar a mayor resolución en futuras tomas (apartado 4).

---

### Resumen accionable

| Prioridad | Cambio | Por qué (caso real) |
|---|---|---|
| Alta | **Log de sesión** (secuencia, demos, incidencias) | evita casi todas las excepciones manuales (Vane, 001, demos) |
| Alta | **≥4 ArUcos por panel** | `blue_circle` con 1 marcador → polígono malo |
| Alta | **Tablero despejado entre paneles** | 001 perdió 3 trials por cadencia apretada |
| Media | **Marcadores redundantes, no en el borde** | 042 perdía marcadores al desdistorsionar |
| Media | **Mayor resolución de vídeo** | 720p limita precisión por celda y ArUco |
| Media | **Verificar integridad de cada grabación** | 032 truncado; sesiones incompletas |
| Baja | **Orientación de tablero inequívoca** | hubo que tratar la rotación 180° |
