"""Local model lifecycle and memory-conscious inference helpers."""

from __future__ import annotations

import gc
import json
import os
from contextlib import contextmanager
from itertools import chain
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np


def _is_out_of_memory(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, MemoryError) or any(
        marker in message
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "not enough memory",
            "failed to allocate",
            "cannot allocate memory",
        )
    )


class ModelLoader:
    """Loads one local model at a time and releases it even when inference fails."""

    def __init__(self) -> None:
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "")
        self.llm_context_size = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
        self.stt_model = os.getenv("STT_MODEL", "base")
        self.ocr_enabled = os.getenv("OCR_ENABLED", "true").lower() == "true"

    @contextmanager
    def _managed(self, factory: Any) -> Iterator[Any]:
        model = None
        try:
            model = factory()
            yield model
        finally:
            try:
                if model is not None:
                    close = getattr(model, "close", None)
                    if callable(close):
                        close()
            finally:
                model = None
                self.release_memory()

    @staticmethod
    def release_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe in RAM; retry on CPU after a CUDA OOM."""
        from faster_whisper import WhisperModel

        device = "cuda" if self._cuda_available() else "cpu"

        def run(target: str) -> str:
            compute_type = "float16" if target == "cuda" else "int8"
            with self._managed(
                lambda: WhisperModel(
                    self.stt_model,
                    device=target,
                    compute_type=compute_type,
                    cpu_threads=max(1, (os.cpu_count() or 2) - 1),
                )
            ) as model:
                segments, _ = model.transcribe(audio, vad_filter=True)
                transcript = " ".join(
                    segment.text.strip() for segment in segments
                ).strip()
                del segments
                return transcript

        try:
            return run(device)
        except Exception as error:
            if device != "cuda" or not _is_out_of_memory(error):
                raise
            self.release_memory()
            return run("cpu")

    def create_ocr(self, use_gpu: bool = False) -> Any:
        if not self.ocr_enabled:
            raise RuntimeError(
                "Se necesita OCR para este PDF escaneado, pero OCR_ENABLED=false."
            )
        det_model_dir = os.getenv("OCR_DET_MODEL_DIR", "")
        rec_model_dir = os.getenv("OCR_REC_MODEL_DIR", "")
        if not det_model_dir or not os.path.isdir(det_model_dir):
            raise RuntimeError(
                "Configura OCR_DET_MODEL_DIR con la carpeta local del modelo "
                "de detección PaddleOCR."
            )
        if not rec_model_dir or not os.path.isdir(rec_model_dir):
            raise RuntimeError(
                "Configura OCR_REC_MODEL_DIR con la carpeta local del modelo "
                "de reconocimiento PaddleOCR."
            )
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError(
                "El PDF requiere OCR. Instala las dependencias opcionales indicadas "
                "en README.md (requirements-ocr.txt)."
            ) from error
        return PaddleOCR(
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            use_angle_cls=False,
        )

    def extract_ocr(
        self, pages: Iterable[tuple[str, np.ndarray]], max_chars: int = 8_000
    ) -> list[tuple[str, str]]:
        """Run OCR on CPU by default; retry on CPU if optional GPU inference OOMs."""
        if max_chars <= 0:
            return []
        page_iterator = iter(pages)
        first_page = next(page_iterator, None)
        if first_page is None:
            return []
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Falta OpenCV para OCR. Instala requirements-ocr.txt."
            ) from error
        all_pages = chain((first_page,), page_iterator)
        try_gpu = os.getenv("OCR_USE_GPU", "false").lower() == "true"
        ocr = None
        current_page = first_page
        started = False
        results: list[tuple[str, str]] = []
        extracted_chars = 0

        def append_result(label: str, words: list[str]) -> None:
            nonlocal extracted_chars
            text = "\n".join(words)
            text = text[: max_chars - extracted_chars]
            results.append((label, text))
            extracted_chars += len(text)

        try:
            ocr = self.create_ocr(use_gpu=try_gpu)
            for current_page in all_pages:
                started = True
                label, image = current_page
                result = ocr.ocr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cls=False)
                words: list[str] = []
                for page in result or []:
                    for line in page or []:
                        if len(line) > 1 and line[1]:
                            words.append(str(line[1][0]))
                append_result(label, words)
                if extracted_chars >= max_chars:
                    break
            return results
        except Exception as error:
            if not try_gpu or not _is_out_of_memory(error):
                raise
            if ocr is not None:
                del ocr
            self.release_memory()
            ocr = self.create_ocr(use_gpu=False)
            remaining_pages = (
                chain((current_page,), all_pages)
                if started
                else chain((first_page,), all_pages)
            )
            for label, image in remaining_pages:
                result = ocr.ocr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cls=False)
                words = [
                    str(line[1][0])
                    for page in result or []
                    for line in page or []
                    if len(line) > 1 and line[1]
                ]
                append_result(label, words)
                if extracted_chars >= max_chars:
                    break
            return results
        finally:
            if ocr is not None:
                del ocr
            self.release_memory()

    def invoke_llm(self, prompt: str) -> str:
        if not self.ollama_model:
            raise RuntimeError(
                "Configura OLLAMA_MODEL con el nombre que muestra `ollama list`."
            )
        parsed_host = urlparse(self.ollama_host)
        if (
            parsed_host.scheme != "http"
            or parsed_host.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed_host.path not in {"", "/"}
            or parsed_host.username
            or parsed_host.password
        ):
            raise RuntimeError(
                "OLLAMA_HOST debe apuntar al servicio local, por ejemplo "
                "http://127.0.0.1:11434."
            )

        def run(context_size: int) -> str:
            request_body = json.dumps(
                {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "keep_alive": 0,
                    "options": {
                        "num_ctx": context_size,
                        "num_predict": 800,
                        "temperature": 0.1,
                    },
                }
            ).encode("utf-8")
            request = Request(
                f"{self.ollama_host}/api/generate",
                data=request_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=600) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"Ollama respondió HTTP {error.code}: {detail or error.reason}"
                ) from error
            except URLError as error:
                raise RuntimeError(
                    "No se pudo conectar con Ollama en "
                    f"{self.ollama_host}. Verifica que Ollama esté iniciado."
                ) from error
            except json.JSONDecodeError as error:
                raise RuntimeError("Ollama devolvió una respuesta JSON inválida.") from error

            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(f"Ollama: {result['error']}")
            if not isinstance(result, dict) or not isinstance(
                result.get("response"), str
            ):
                raise RuntimeError("Ollama no devolvió el campo de texto `response`.")
            return result["response"]

        try:
            return run(self.llm_context_size)
        except Exception as error:
            fallback_context = max(2048, self.llm_context_size // 2)
            if not _is_out_of_memory(error) or fallback_context >= self.llm_context_size:
                raise
            self.release_memory()
            return run(fallback_context)
