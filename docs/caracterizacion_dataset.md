# Caracterización del dataset (22 participantes)

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

## 1. Equipo y captura

- **Dispositivo:** Pupil Labs **Core** (Kassner, Patera & Bulling, 2014).
- **Software de grabación:** **Pupil Capture 3.5.7** (los 22 registros).
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

\* **011** se calibró en binocular, pero el gaze registrado es ~94 % monocular del ojo izquierdo
(uno de los ojos deja de detectarse con frecuencia), por lo que su tasa efectiva es la monocular
(~124 Hz). El porcentaje en las filas binoculares indica la fracción de muestras efectivamente
binoculares; el resto se registra como monocular cuando solo se detecta un ojo.

## 3. Resumen de heterogeneidad

- **Modelo de mapeo:** 12 participantes en 2D monocular, 5 en 3D monocular, 4 con
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
- **Número de calibraciones (1–5):** distribución sobre los 22 registros — 1 calib.: 8 part.;
  2: 4; 3: 6; 4: 2; 5: 2.

  | Nº calibraciones | Participantes |
  |---|---|
  | 1 | 012, 035, 042, 044, 049, 055, 064, P-A |
  | 2 | 007_1, 009, 027, 032 |
  | 3 | 007, 008, 011, 024, 051, P-B |
  | 4 | 001, 058 |
  | 5 | 002, 054 |

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
  explica la variabilidad en el número de calibraciones (de 1 a 5) y los ocho registros con una
  sola.
- En los registros más tardíos se empleó directamente la configuración 2D, coherente con
  que los participantes de numeración más alta sean mayoritariamente 2D.

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
Resultado sobre los 22 (artefactos en `calibration/gaze/`):

| Participante | nº cal | base→corr (px) | ganancia | IC inf. | `apply` |
|---|---|---:|---:|---:|:--:|
| 044 | 1 | 47.0 → 13.2 | **+72 %** | +58 % | ✅ |
| P-A | 1 | 31.4 → 13.2 | **+58 %** | +46 % | ✅ |
| 009 | 2 | 47.8 → 28.8 | +40 % | +16 % | ✅ |
| 054 | 5 | 34.9 → 21.7 | +38 % | +17 % | ✅ |
| 055 | 1 | 26.4 → 17.7 | +33 % | +18 % | ✅ |
| 032 | 2 | 22.6 → 15.8 | +30 % | +19 % | ✅ |
| 064 | 1 | 16.4 → 12.5 | +24 % | +9 % | ✅ |
| 011 | 3 | 30.8 → 23.9 | +23 % | +2 % | ✅ |
| 012, 035, 051, 049 | 1–3 | — | +10…+15 % | <0 | — |
| 024, 027, 002, 001, 008, P-B | — | — | +1…+6 % | <0 | — |
| 042, 007, 007_1, 058 | — | — | −2…−37 % | <0 | — |

**Lectura:** 8 participantes reciben una corrección clara (varios >30 %, hasta +72 %), casi
todos de 1 sola calibración —los que Pupil nunca reseteó—. El resto queda en identidad: tanto
los claramente perjudicados (058 −37 %) como los positivos pero inciertos (la CV con 6 paneles
tiene varianza alta, su IC inferior cae por debajo de 0). La compuerta corrige los errores sólidos
(IC inferior > 0) y deja identidad en lo dudoso: nunca arriesga empeorar a nadie.
