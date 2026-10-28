"""Streamlit interface for the local EduEval Assistant MVP."""

from __future__ import annotations

import streamlit as st

from db import initialize_db, update_evaluation
from graph import EvaluationState, StudentFile, build_evaluation_graph

st.set_page_config(page_title="EduEval Assistant", page_icon="📚", layout="wide")
initialize_db()

if st.session_state.pop("reset_upload_widgets", False):
    st.session_state["upload_widget_version"] = (
        st.session_state.get("upload_widget_version", 0) + 1
    )
upload_widget_version = st.session_state.get("upload_widget_version", 0)
processing_error = st.session_state.pop("processing_error", None)
if processing_error:
    st.error(f"No se pudo completar la evaluación: {processing_error}")

st.title("EduEval Assistant")
st.caption("Evaluación asistida localmente; el profesor conserva la decisión final.")

with st.sidebar:
    st.header("Actividad y entregables")
    activity_desc = st.text_area("Descripción de la actividad", height=110)
    rubric_format = st.radio("Formato de rúbrica", ["Markdown", "CSV"], horizontal=True)
    rubric_upload = st.file_uploader(
        "Sube la rúbrica (opcional)", type=["md", "markdown", "csv"]
    )
    rubric_text = st.text_area("Rúbrica", height=180)
    if rubric_upload is not None:
        uploaded_rubric = rubric_upload.getvalue().decode("utf-8-sig", errors="replace")
        rubric_text = uploaded_rubric
    student_name = st.text_input("Nombre del estudiante (solo SQLite)")
    uploaded_files = st.file_uploader(
        "Entregables PDF/HTML",
        type=["pdf", "html"],
        accept_multiple_files="directory",
        help="Selecciona una carpeta. Los nombres de archivo no se comparten con los modelos.",
        key=f"student_files_input_{upload_widget_version}",
    )

st.subheader("Observación del profesor")
st.caption("Graba una nota de voz; el audio se decodifica y transcribe en RAM.")
teacher_audio = st.audio_input(
    "Grabar observación", key=f"teacher_audio_input_{upload_widget_version}"
)

if rubric_upload is not None and not rubric_text.strip():
    st.warning(f"Verifica que la rúbrica subida esté en formato {rubric_format}.")

run_evaluation = st.button(
    "Procesar evaluación",
    type="primary",
    disabled=not activity_desc.strip() or not rubric_text.strip() or not uploaded_files,
)

if run_evaluation:
    files: list[StudentFile] = [
        {"name": file.name, "content": bytearray(file.getvalue())}
        for file in uploaded_files
    ]
    audio_bytes = bytearray(teacher_audio.getvalue()) if teacher_audio else None
    initial_state: EvaluationState = {
        "activity_desc": activity_desc,
        "rubric": rubric_text,
        "student_name": student_name.strip() or "Sin nombre",
        "student_files": files,
        "teacher_audio": audio_bytes,
    }
    try:
        with st.spinner(
            "Procesando secuencialmente: transcripción, documentos y evaluación local..."
        ):
            result = build_evaluation_graph().invoke(initial_state)
        st.session_state["evaluation_state"] = result
    except Exception as error:
        for item in files:
            content = item["content"]
            if isinstance(content, bytearray):
                content.clear()
        if audio_bytes is not None:
            audio_bytes.clear()
        st.session_state["processing_error"] = str(error)
    finally:
        st.session_state["reset_upload_widgets"] = True
    st.rerun()

result = st.session_state.get("evaluation_state")
if result:
    st.divider()
    st.subheader("Resultado para revisión del profesor")
    for warning in result.get("warnings", []):
        st.warning(warning)
    if result.get("teacher_transcript"):
        st.write("**Transcripción del profesor:**")
        st.info(result["teacher_transcript"])

    evaluation = result["final_evaluation"]
    with st.form("evaluation_review"):
        score = st.number_input(
            "Calificación sugerida",
            min_value=0.0,
            max_value=max(0.0, float(evaluation.get("max_score", 100))),
            value=float(evaluation.get("score", 0)),
            step=0.5,
        )
        feedback = st.text_area(
            "Retroalimentación",
            value=str(evaluation.get("feedback", "")),
            height=180,
        )
        strengths = st.text_area(
            "Fortalezas (una por línea)",
            value="\n".join(evaluation.get("strengths", [])),
        )
        improvements = st.text_area(
            "Áreas de mejora (una por línea)",
            value="\n".join(evaluation.get("improvements", [])),
        )
        col_save, col_approve = st.columns(2)
        save_clicked = col_save.form_submit_button("Guardar cambios")
        approve_clicked = col_approve.form_submit_button(
            "Aprobar feedback", type="primary"
        )
    if save_clicked or approve_clicked:
        evaluation["score"] = score
        evaluation["feedback"] = feedback
        evaluation["strengths"] = [
            line.strip() for line in strengths.splitlines() if line.strip()
        ]
        evaluation["improvements"] = [
            line.strip() for line in improvements.splitlines() if line.strip()
        ]
        try:
            update_evaluation(result, evaluation, approved=approve_clicked)
            result["final_evaluation"] = evaluation
            st.session_state["evaluation_state"] = result
            st.success(
                "Feedback aprobado y guardado."
                if approve_clicked
                else "Cambios guardados como borrador."
            )
        except Exception as error:
            st.error(f"No se pudo guardar en SQLite: {error}")

    with st.expander("Detalle de criterios y discrepancias"):
        st.json(
            {
                "criteria": evaluation.get("criteria", []),
                "discrepancies": evaluation.get("discrepancies", []),
                "evaluation_id": result.get("evaluation_id"),
            }
        )
