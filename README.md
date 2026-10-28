# EduEval Assistant

MVP local para generar una evaluación preliminar con LangGraph, con intervención
obligatoria del profesor antes de aprobar el feedback. Streamlit, transcripción con
faster-whisper, extracción de documentos con PyMuPDF/BeautifulSoup, OCR opcional
con PaddleOCR y evaluación textual con un GGUF local mediante llama.cpp.

## Flujo y memoria

```text
START
  -> prepare_inputs
  -> transcribe_audio       [faster-whisper -> liberar modelo y audio]
  -> analyze_deliverables   [HTML/PDF -> OCR solo si es necesario -> liberar OCR]
  -> generate_ai_evaluation [LLM local -> liberar]
  -> calibrate_with_teacher[LLM local solo si hay transcripción -> liberar]
  -> save_evaluation        [SQLite, estado draft]
  -> END
```

El estado `EvaluationState` transporta identificadores, descripción, rúbrica,
archivos en memoria, audio mutable, transcripción, texto extraído, evaluación
preliminar/final y advertencias. Los nodos son secuenciales; cada carga usa un
context manager que libera referencias, ejecuta `gc.collect()` y vacía las cachés
CUDA disponibles. El nombre no forma parte del prompt; se enmascara en el texto
extraído antes de enviarlo al LLM. No se persisten audio ni entregables.

## Requisitos

- Python 3.11 recomendado (64-bit).
- Windows o Linux; 16 GB de RAM. En Iris Xe, la inferencia se ejecutará en CPU:
  no se asume soporte CUDA. Se recomienda `STT_MODEL=base`, contexto LLM de 6144
  tokens y procesamiento de un estudiante a la vez.
- Espacio local para los modelos. El primer uso de faster-whisper puede descargar
  el modelo CTranslate2. Para un entorno totalmente desconectado, descarga los
  pesos previamente y apunta `STT_MODEL` a su carpeta local.

## Instalación

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Instala el binding de `llama-cpp-python` con la variante que corresponda a tu
equipo. La instalación predeterminada es CPU; compilar con CUDA es opcional y no
aplica a Iris Xe. Descarga manualmente un GGUF instruct cuantizado (por ejemplo,
Qwen3-4B-Instruct Q4_K_M) desde una fuente autorizada y configura la ruta local.
La aplicación no descarga ni invoca APIs de LLM en la nube.

OCR solo se requiere para PDFs compuestos por imágenes. El MVP fija PaddleOCR
2.9.1 con su detector/reconocedor ligero PP-OCR; no es la variante de 0.9B
PaddleOCR-VL y prioriza compatibilidad y RAM sobre reconstrucción avanzada del
layout/tablas:

```powershell
pip install -r requirements-ocr.txt
```

El backend opcional usa PaddleOCR en CPU de forma predeterminada. Descarga
previamente los modelos PP-OCR de detección y reconocimiento compatibles con
PaddleOCR 2.9, guárdalos en carpetas locales y establece `OCR_DET_MODEL_DIR` y
`OCR_REC_MODEL_DIR`. El código exige estas rutas y no descarga pesos OCR durante
la evaluación. Se requiere una instalación de Paddle compatible con la versión
de Python y el sistema operativo; si la rueda no está disponible en Windows,
instala la versión indicada por la documentación de PaddlePaddle para tu
plataforma.

## Configuración y ejecución

```powershell
Copy-Item .env.example .env
```

Streamlit no carga `.env` por sí mismo: exporta las variables en PowerShell o
usa un cargador de entorno aprobado localmente. Como mínimo establece
`LLM_MODEL_PATH` a una ruta real:

```powershell
$env:LLM_MODEL_PATH = "C:\models\Qwen3-4B-Instruct-Q4_K_M.gguf"
$env:STT_MODEL = "base"
streamlit run main.py
```

`LLM_GPU_LAYERS=0` fuerza CPU. Si PyTorch detecta CUDA, el código intenta el número
configurado de capas y reintenta en CPU ante un error de memoria. El audio se
recibe con `st.audio_input`, se decodifica desde RAM y se sobrescribe el búfer
mutable tras transcribir; no se escribe un archivo de audio. Streamlit puede
retener temporalmente el objeto del widget en la sesión hasta el rerun que lo
limpia. No habilites logging de datos sensibles.

La aceleración GPU requiere que cada backend (CTranslate2, llama.cpp y, si aplica,
PaddlePaddle) esté instalado con soporte compatible para esa GPU. Iris Xe no es
CUDA; usa los backends CPU predeterminados y no instales una variante CUDA para
este equipo.

## Uso y almacenamiento

1. Escribe la actividad y la rúbrica o sube una rúbrica `.md`/`.csv`.
2. Sube uno o más PDFs/HTML y, opcionalmente, graba la observación del profesor.
3. Pulsa **Procesar evaluación**. El resultado se guarda como `draft`.
4. Revisa y edita el puntaje y feedback; **Guardar cambios** mantiene el borrador,
   **Aprobar feedback** cambia el estado a `approved`.

SQLite guarda únicamente actividad/rúbrica, nombre de estudiante, puntuación,
criterios, transcripción y feedback. La transcripción de voz puede contener datos
personales: confirma las políticas institucionales antes de persistirla; para
deshabilitar ese almacenamiento hace falta ajustar el esquema/persistencia.
Los archivos del estudiante se conservan solo en memoria durante el proceso.

## Límites deliberados del MVP

- `faster-whisper` usa modelos CTranslate2 `base`/`small` e `int8` en CPU. `Q5_K_M`
  es una cuantización GGUF empleada por llama.cpp, no un formato de faster-whisper.
- PDF digital y HTML no necesitan un modelo visual. Solo las páginas sin texto
  extraíble se envían a PaddleOCR; las páginas se procesan de forma anónima y el
  nombre del archivo no se entrega al OCR.
- La evaluación es sugerencia; el puntaje automático y su calibración no
  constituyen una decisión final. El profesor debe revisar y aprobar.
- El reconocimiento de nombres es de mejor esfuerzo: se enmascara el nombre
  proporcionado y líneas etiquetadas como `Nombre:`/`Name:`; el docente debe
  verificar que el texto no incluya otros identificadores.
- Se aceptan hasta 25 MB por archivo y 50 MB en total para acotar los picos de RAM.
- El documento se limita a 8,000 caracteres, la rúbrica a 4,000, la actividad a
  2,000 y la transcripción del profesor a 2,000 para controlar contexto y memoria.
  La interfaz muestra advertencias cuando aplica truncamiento.
- No se ha añadido una política de retención/consentimiento, autenticación,
  gestión multiusuario ni exportación de notas para LMS.

Rúbricas de ejemplo: [Markdown](rubric-example.md) y [CSV](rubric-example.csv).
