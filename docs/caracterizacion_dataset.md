# Caracterización del dataset (27 participantes)

Este documento describe **los datos concretos** procesados por este repositorio: quién es cada
registro y con qué configuración de captura se grabó. Es complementario, no redundante, con los
otros documentos:

- *Cómo funciona el procesado* (mecanismo, algoritmos, parámetros) → [documentacion_tecnica.md](documentacion_tecnica.md)
- *Cómo ejecutarlo* → [guia_procesamiento.md](guia_procesamiento.md)
- *Qué cambiar en el próximo experimento* → [recomendaciones_diseno_experimental.md](recomendaciones_diseno_experimental.md)
- *Valores calculados sobre un procesado concreto* → pestaña **Frecuencias** del informe HTML

El repositorio está **ajustado a este experimento** (tablero físico, umbrales de toque por color,
giro del *block* 3 —índice 0-base, el 4.º presentado—, etc.); no es una librería genérica. Por eso conviene tener esta caracterización
separada: el mecanismo es estable, pero los datos de este caso son heterogéneos y esa
heterogeneidad condiciona cualquier análisis posterior.

**Índice**

- [1. Equipo y captura](#1-equipo-y-captura)
- [2. Tabla por participante](#2-tabla-por-participante)
- [3. Resumen de heterogeneidad](#3-resumen-de-heterogeneidad)
- [4. Contexto de adquisición](#4-contexto-de-adquisición)
- [5. Reproducción offline de la calibración](#5-reproducción-offline-de-la-calibración)
- [6. Corrección de deriva de mirada: ganancias por participante](#6-corrección-de-deriva-de-mirada-ganancias-por-participante)
- [7. Calidad de mirada y cobertura de objetivo visto por modo de adquisición](#7-calidad-de-mirada-y-cobertura-de-objetivo-visto-por-modo-de-adquisición)

## 1. Equipo y captura

- **Dispositivo:** Pupil Labs **Core** (Kassner, Patera & Bulling, 2014).
- **Software de grabación:** **Pupil Capture 3.5.7** (los 27 registros).
- **Cámaras oculares:** 200 Hz nominal, 192×192 px, técnica de pupila oscura sobre imagen IR.
- **Cámara de escena (*World*):** **1280×720** en estas grabaciones (hasta 1080p/30 Hz nominal).
- **Convención de ojos de Pupil** (corroborada en el código del *gazer* 2D): `id 0` = ojo
  derecho (cámara *eye0*), `id 1` = ojo izquierdo (cámara *eye1*).

## 2. Tabla por participante

Medido directamente sobre los `.pldata`/`.npy` de entrada (`gaze.pldata`, `notify.pldata`,
`gaze_timestamps.npy`).

| Participante | Modelo de mapeo | Ojo(s) registrados | Gaze efectivo (Hz) | Nº calibraciones |
|---|---|---|---:|---:|
| 001     | 3D    | mono izq        | 124 | 4 |
| 002     | 3D    | mono izq        | 124 | 5 |
| 007     | 2D    | binocular (61 %) | 248 | 3 |
| 007_1   | 3D    | mono izq        | 124 | 2 |
| 008     | 3D    | mono izq        | 124 | 3 |
| 009     | 2D    | binocular (61 %) | 248 | 2 |
| 011     | 2D    | binocular*      | 124 | 3 |
| 012     | 2D    | mono izq        | 124 | 1 |
| 024     | 3D    | mono izq        | 124 | 3 |
| 027     | 2D+3D | mono izq        | 124 | 2 |
| 032     | 2D    | mono izq        | 124 | 2 |
| 035     | 2D    | mono der        | 124 | 1 |
| 042     | 2D    | mono der        | 124 | 1 |
| 044     | 2D    | mono der        | 124 | 1 |
| 049     | 2D    | mono der        | 124 | 1 |
| 051     | 2D    | mono der        | 124 | 3 |
| 054     | 2D    | mono der        | 124 | 5 |
| 055     | 2D    | mono izq        | 124 | 1 |
| 058     | 2D    | mono izq        | 124 | 4 |
| 064     | 2D    | mono izq        | 124 | 1 |
| P-B     | 2D    | binocular (72 %) | 248 | 3 |
| P-A     | 2D    | mono izq        | 124 | 1 |
| P-C   | 2D    | mono izq        | 124 | 5 |
| P-D   | 2D    | mono izq        | 124 | 2 |
| P-E   | 2D    | mono der        | 124 | 6 |
| P-F   | 2D    | mono izq        | 124 | 2 |
| P-G   | 2D    | mono izq        | 124 | 4 |

\* **011** se calibró en binocular, pero el gaze registrado es ~94 % monocular del ojo izquierdo
(uno de los ojos deja de detectarse con frecuencia), por lo que su tasa efectiva es la monocular
(~124 Hz). El porcentaje en las filas binoculares indica la fracción de muestras efectivamente
binoculares; el resto se registra como monocular cuando solo se detecta un ojo.

## 3. Resumen de heterogeneidad

- **Modelo de mapeo:** 17 participantes en 2D monocular, 5 en 3D monocular, 4 con
  calibración 2D binocular, 1 mixto 2D/3D (027). El 2D es más preciso en condiciones ideales
  (<1°) pero más sensible al deslizamiento; el 3D es más robusto pero menos exacto (1,5–2,5°); el
  monocular pierde la convergencia binocular. La mirada de un binocular 2D no es directamente
  comparable a la de un monocular 3D — el porqué (mecanismo) está en
  [documentacion_tecnica.md §13](documentacion_tecnica.md).
- **Ojo registrado:** entre los monoculares conviven ojo derecho (`id 0`) e izquierdo (`id 1`); no
  es uniforme.
- **Tasa de gaze efectiva:** ~124 Hz en monocular y ~248 Hz en binocular (el flujo binocular
  intercala las corrientes de ambos ojos). Es menor que los 200 Hz nominales de la cámara ocular;
  *hipótesis*: filtrado por confianza (≤0,6 descartado) y emparejamiento pupila→gaze reducen la tasa
  registrada. No está medida la causa, solo la tasa resultante.
- **Número de calibraciones (1–6):** distribución sobre los 27 registros — 1 calib.: 8 part.;
  2: 6; 3: 6; 4: 3; 5: 3; 6: 1.

  | Nº calibraciones | Participantes |
  |---|---|
  | 1 | 012, 035, 042, 044, 049, 055, 064, P-A |
  | 2 | 007_1, 009, 027, 032, P-D, P-F |
  | 3 | 007, 008, 011, 024, 051, P-B |
  | 4 | 001, 058, P-G |
  | 5 | 002, 054, P-C |
  | 6 | P-E |

  Cuando hay varias, el gaze grabado usa la más reciente en cada instante (en Pupil 3.5 cada
  calibración exitosa reemplaza al *gazer* activo — `uniqueness="by_base_class"`), lo que acota la
  deriva entre bloques. En los 8 participantes con una sola calibración no hay recalibración
  que acote la deriva intra-sesión.

## 4. Contexto de adquisición

La heterogeneidad de configuraciones responde a las condiciones de captura de la sesión. Según la
persona que realizó la toma (información cualitativa, no registrada en el momento), los factores
fueron:

- La intención inicial era registrar a todos los participantes en binocular, pero en varios
  casos no se obtenía señal estable de uno de los ojos y se optó por configuración monocular. Los
  datos son coherentes con ello: de las cuatro calibraciones binoculares, el participante 011
  registra aproximadamente un 94 % de muestras monoculares.
- El modelo 3D presentó dificultades en varios participantes, lo que explica su carácter
  minoritario en el conjunto final.
- Algunos participantes presentaron dificultades de calibración pese a varios intentos, lo que
  explica la variabilidad en el número de calibraciones (de 1 a 6) y los ocho registros con una
  sola.
- En los registros más tardíos se empleó directamente la configuración 2D, coherente con
  que los participantes de numeración más alta sean mayoritariamente 2D.
- Algunas sesiones tienen **incidencias por trial** registradas en el momento de la toma, que se
  documentan como descartes `[-1]` en el `trials_config_exceptions/<id>.yaml` del participante (no
  son fallos de detección y quedan fuera del recuento de trials): una carta mostrada pero no buscada
  porque el participante **cerraba los ojos** y había que re-presentarla (P-C, en los bloques 1, 3
  y 4), una carta **no presentada** en la toma (P-D bloque 3, P-E bloque 0), o la **cámara
  saliéndose del tablero** durante un trial (P-E bloque 4). Cada caso está verificado en el vídeo y
  anotado con su motivo y su marca de tiempo en el fichero de excepción.

Estos factores reflejan las limitaciones prácticas del equipo y del protocolo disponible en el
momento de la toma. Se documentan porque condicionan la comparabilidad entre participantes y
deben tenerse en cuenta en cualquier análisis posterior.

## 5. Reproducción offline de la calibración

Cada calibración puede **reconstruirse y reproducirse offline** desde su `calib_data` (lista de
marcas de referencia + pupila) con el mismo modelo polinómico que Pupil (rasgos
`[x, y, xy, x², y², x²y²]` + regresión lineal con descarte de *outliers* a 70 px), sin la GUI de
Pupil. Validación sobre el participante 049 (2D monocular, 1 calibración): el gaze recalculado
coincide con el grabado a 0,00 px sobre 36 527 muestras. Esto habilita el cálculo de mirada
offline propio para la compensación de error por participante (ver
[recomendaciones_diseno_experimental.md](recomendaciones_diseno_experimental.md)).

## 6. Corrección de deriva de mirada: ganancias por participante

La etapa de corrección de deriva (mecanismo en [documentacion_tecnica.md §7](documentacion_tecnica.md))
recupera las recalibraciones no registradas desde los paneles del vídeo y mide, por participante, la
reducción de error fuera de muestra (CV *leave-one-panel-out*, mediana en píxeles de imagen). La
compuerta adopta la corrección solo si la ganancia es fiablemente positiva (el límite inferior,
percentil 5 por *bootstrap*, es > 0) y el error base es relevante (≥12 px); si no, identidad.
Resultado sobre los 27 (artefactos en `calibration/gaze/`):

| Participante | nº cal | base→corr (px) | ganancia | IC inf. | `apply` |
|---|---|---:|---:|---:|:--:|
| 044 | 1 | 47.0 → 13.2 | **+72 %** | +58 % | ✅ |
| P-A | 1 | 31.4 → 13.2 | **+58 %** | +46 % | ✅ |
| 009 | 2 | 47.8 → 28.8 | +40 % | +16 % | ✅ |
| 054 | 5 | 34.9 → 21.7 | +38 % | +17 % | ✅ |
| 055 | 1 | 26.4 → 17.7 | +33 % | +18 % | ✅ |
| 032 | 2 | 22.6 → 15.8 | +30 % | +19 % | ✅ |
| P-D | 2 | 28.9 → 20.3 | +30 % | +17 % | ✅ |
| P-E | 6 | 29.1 → 20.7 | +29 % | +6 % | ✅ |
| 064 | 1 | 16.4 → 12.5 | +24 % | +9 % | ✅ |
| 011 | 3 | 30.8 → 23.9 | +23 % | +2 % | ✅ |
| P-F, P-G | 2, 4 | — | +23…+28 % | <0 | — |
| 012, 035, 051, 049 | 1–3 | — | +10…+15 % | <0 | — |
| 024, 027, 002, 001, 008, P-B | — | — | +1…+6 % | <0 | — |
| 042, 007, 007_1, 058, P-C | — | — | −2…−37 % | <0 | — |

**Lectura:** 10 participantes reciben una corrección clara (varios >30 %, hasta +72 %), la mayoría
de pocas calibraciones —los que Pupil nunca reseteó—. El resto queda en identidad: tanto
los claramente perjudicados (058 −37 %) como los positivos pero inciertos (la CV con 6 paneles
tiene varianza alta, su IC inferior cae por debajo de 0). La compuerta corrige los errores sólidos
(IC inferior > 0) y deja identidad en lo dudoso: nunca arriesga empeorar a nadie.

## 7. Calidad de mirada y cobertura de objetivo visto por modo de adquisición

Esta sección complementa el modo de mapeo (§2), el número de calibraciones (§3) y el error base de
la corrección de deriva (§6), relacionando la configuración de adquisición con dos magnitudes que
aquéllas no recogen: (a) el **tamaño de la elipse de incertidumbre** de la mirada —`bias_rms` =
√(½·traza de `bias_cov_px`), el error de exactitud/deriva en px de imagen— y (b) la **cobertura de
la marca `frame_target_found`** («objetivo visto»): la fracción de *trials* en los que el modelo de
incertidumbre confirma que se miró la casilla objetivo (masa de la elipse sobre la casilla ≥ umbral
0,34; mecanismo en [documentacion_tecnica.md §7.5](documentacion_tecnica.md)). Una cobertura baja no
indica necesariamente peor búsqueda: con baja calidad de mirada no puede afirmarse con confianza que
el participante mirara la casilla exacta.

Medido sobre el procesado **v1.4.1** (`frame_target_found` a umbral 0,34) y los artefactos de
`calibration/gaze/`, ordenado por cobertura. Los descartes `[-1]` documentados (§4) quedan fuera
del denominador (no son *trials* reales):

| Participante | Modo | Objetivo visto | Cobertura | `bias_rms` (px) | jitter σ (px) | `med_conf` |
|---|---|---:|---:|---:|---:|---:|
| P-G | 2D    | 24/60 | 40 %  | 22.6 | 23.7 | 0.28 |
| 008   | 3D    | 22/49 | 45 %  | 29.7 | 17.9 | 0.22 |
| 011   | 2D    | 28/53 | 53 %  | 19.8 | 16.8 | 0.40 |
| 002   | 3D    | 33/60 | 55 %  | 28.2 | 21.1 | 0.42 |
| 009   | 2D    | 34/59 | 58 %  | 25.7 | 24.5 | 0.39 |
| 001   | 3D    | 38/60 | 63 %  | 31.6 | 19.8 | 0.37 |
| P-E | 2D    | 41/58 | 71 %  | 18.2 | 24.3 | 0.59 |
| 058   | 2D    | 44/60 | 73 %  | 20.6 | 23.0 | 0.58 |
| P-C | 2D    | 45/60 | 75 %  | 21.7 | 18.5 | 0.61 |
| 051   | 2D    | 47/60 | 78 %  | 20.9 | 17.1 | 0.71 |
| 024   | 3D    | 49/60 | 82 %  | 24.9 | 20.1 | 0.51 |
| 035   | 2D    | 50/59 | 85 %  | 24.1 | 13.6 | 0.57 |
| P-D | 2D    | 52/59 | 88 %  | 16.4 | 16.8 | 0.76 |
| 007_1 | 3D    | 53/60 | 88 %  | 28.7 | 16.8 | 0.63 |
| 054   | 2D    | 49/54 | 91 %  | 20.6 | 20.3 | 0.79 |
| 042   | 2D    | 54/59 | 92 %  | 20.0 |  9.5 | 0.84 |
| P-F | 2D    | 55/60 | 92 %  | 26.1 | 20.3 | 0.63 |
| 007   | 2D    | 55/59 | 93 %  | 23.7 | 20.5 | 0.61 |
| 012   | 2D    | 56/60 | 93 %  | 24.0 | 13.8 | 0.67 |
| 044   | 2D    | 47/50 | 94 %  | 14.5 | 19.4 | 0.76 |
| 064   | 2D    | 56/59 | 95 %  | 12.7 |  7.9 | 0.90 |
| 032   | 2D    | 57/59 | 97 %  | 15.4 | 12.2 | 0.83 |
| 055   | 2D    | 55/56 | 98 %  | 15.9 | 17.7 | 0.88 |
| P-B   | 2D    | 59/60 | 98 %  | 18.4 | 17.3 | 0.80 |
| 027   | 2D+3D | 59/59 | 100 % | 19.8 | 11.6 | 0.83 |
| 049   | 2D    | 58/58 | 100 % | 12.2 | 11.2 | 0.96 |
| P-A   | 2D    | 60/60 | 100 % | 14.9 |  9.4 | 0.90 |

(total cohorte: **1280/1570 = 82 %**.)

**La cobertura la determina el tamaño de la elipse, no el modo en sí.** Correlaciones sobre los 27:

| factor | correlación con cobertura |
|---|---:|
| `med_conf` (mediana de la masa de la mejor fijación del *trial*) | **+0,92** |
| `bias_rms` (exactitud/deriva, px) | −0,56 |
| jitter σ (px) | −0,59 |
| `base_px` (residual de calibración, §6) | −0,45 |

La cobertura está **casi determinada** (`med_conf` r=+0,92) por lo concentrada que es la elipse de
cada participante: elipse pequeña → masa concentrada en la casilla → supera el umbral.

**Asociación con el modo de mapeo (2D vs 3D):**

| grupo | n | cobertura media | `bias_rms` medio |
|---|---:|---:|---:|
| **3D** | 5 | **67 %** | 28.6 px |
| **2D** | 21 | **84 %** | 19.4 px |

Los 5 participantes mapeados en **3D** (001, 002, 007_1, 008, 024) concentran las coberturas más
bajas: su elipse de exactitud/deriva es mayor (28.6 vs 19.4 px de media). Coherente con que el 3D
monocular sea menos exacto (§3) y con las dificultades de ajuste del modelo 3D reportadas en la toma
(§4). El **ojo registrado y el carácter mono/binocular** (detallados por participante en §2) no
añaden poder explicativo: la media de los registros monoculares (~81 %) y de los binoculares
(~83 %) es prácticamente la misma, y con solo 3 registros binoculares (007, 009, P-B) el contraste
mono/binocular no es concluyente; el factor asociado a la cobertura es el modo de mapeo y, sobre
todo, el tamaño de la elipse.

**Factores descartados o no registrables.** El **brillo y el contraste de la imagen de ojo**
—medidos sobre los vídeos IR `eyeN.mp4` (media, contraste RMS, rango dinámico y contraste
pupila-iris, ~80 fotogramas por registro)— **no se asocian** con la cobertura ni con `bias_rms`
(|r| ≤ 0,20): en imagen IR de pupila oscura el color de iris visible apenas se traslada a la imagen,
de modo que «ojos claros/oscuros» no es un factor medible aquí. El **color de iris** y la
**iluminación** de escena no se registran. La baja **confianza de pupila** de algunos registros
(p. ej. 008 ≈ 0,43) apunta a peor calidad de imagen ocular, pero su causa (iluminación,
pestañas/maquillaje, deslizamiento, fisiología) no puede aislarse con el material disponible.

**Cautelas.** (1) n pequeño —5 registros en 3D— ⇒ es una **asociación**, no una causa aislada: el
3D va confundido con la dificultad de calibración (§4). (2) La cobertura baja es una medida
**honesta de la incertidumbre del aparato**, no de peor búsqueda; bajar el umbral inflaría la cifra
contando miradas que el aparato sitúa a más de una casilla del objetivo. (3) Para análisis
posteriores conviene **estratificar/ponderar por calidad de mirada** (`bias_rms` / `med_conf`) y
tratar los 5 registros 3D como un subgrupo de menor exactitud.
