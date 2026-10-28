# EduEval Assistant

MVP local para generar una evaluación preliminar con LangGraph, con intervención
obligatoria del profesor antes de aprobar el feedback. Streamlit, transcripción con
faster-whisper, extracción de documentos con PyMuPDF/BeautifulSoup, OCR opcional
con PaddleOCR y evaluación textual mediante el servicio local de Ollama.

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
  no se asume soporte CUDA. Se recomienda `STT_MODEL=base`, contexto LLM de 4096
  tokens y procesamiento de un estudiante a la vez.
- Espacio local para los modelos. El primer uso de faster-whisper puede descargar
  el modelo CTranslate2. Para un entorno totalmente desconectado, descarga los
  pesos previamente y apunta `STT_MODEL` a su carpeta local.

## Instalación

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows se fija `av==14.2.0`: `faster-whisper` lo requiere indirectamente,
pero algunas versiones de PyAV pueden no tener un wheel compatible y pip intenta
compilarlas con Microsoft C++ Build Tools. Usa Python 3.11 de 64 bits y ejecuta
la instalación desde el entorno virtual activado para que `python` y `pip`
correspondan al mismo intérprete.

Instala Ollama para Windows y descarga el modelo una sola vez. Consulta el
identificador exacto con:

```powershell
ollama list
ollama show qwen2.5:3b
```

Usa el nombre que aparece en `ollama list` como `OLLAMA_MODEL`. Ollama mantiene
la gestión de pesos y la selección CPU/GPU; la aplicación le envía solicitudes
solo a `http://127.0.0.1:11434` y no requiere una ruta GGUF ni un paquete Python
de Ollama.

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

Streamlit no carga `.env` por sí mismo: exporta las variables en PowerShell.
Configura el nombre del modelo según `ollama list`:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "qwen2.5:3b"
$env:LLM_CONTEXT_SIZE = "4096"
$env:STT_MODEL = "base"
streamlit run main.py
```

Comprueba que Ollama responda antes de iniciar Streamlit:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

Cada solicitud usa `keep_alive: 0` para pedir a Ollama que descargue el modelo
después de generar. Ante un error de memoria se reintenta una vez con la mitad
del contexto (mínimo 2048); la selección CPU/GPU sigue bajo control de Ollama.
El audio se recibe con `st.audio_input`, se decodifica desde RAM y se sobrescribe
el búfer mutable tras transcribir; no se escribe un archivo de audio. Streamlit
limpia los widgets de carga tras el procesamiento. No habilites logging de datos
sensibles. La grabación WAV se decodifica con la biblioteca estándar de Python.
Aunque `faster-whisper` requiere PyAV indirectamente, el pin de la sección de
instalación evita compilarlo en la combinación soportada de Windows y Python 3.11.

Ollama decide la aceleración disponible. Iris Xe no es CUDA; si Ollama no puede
usar la GPU integrada, ejecutará el modelo en CPU. El rendimiento dependerá del
modelo instalado y la memoria disponible.

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

- `faster-whisper` usa modelos CTranslate2 `base`/`small` e `int8` en CPU; la
  cuantización de Ollama es independiente del formato de faster-whisper.
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
