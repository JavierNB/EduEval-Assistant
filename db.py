"""SQLite persistence for activities, students, evaluation drafts, and feedback."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

DB_PATH = os.getenv("EDUEVAL_DB_PATH", "edueval.sqlite3")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS actividades (
                id TEXT PRIMARY KEY,
                descripcion TEXT NOT NULL,
                rubrica TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS estudiantes (
                id TEXT PRIMARY KEY,
                nombre TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS evaluaciones (
                id TEXT PRIMARY KEY,
                actividad_id TEXT NOT NULL REFERENCES actividades(id),
                estudiante_id TEXT NOT NULL REFERENCES estudiantes(id),
                puntaje REAL,
                puntaje_maximo REAL,
                criterios_json TEXT NOT NULL,
                transcripcion_profesor TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN ('draft', 'approved')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS feedback (
                evaluacion_id TEXT PRIMARY KEY REFERENCES evaluaciones(id)
                    ON DELETE CASCADE,
                fortalezas_json TEXT NOT NULL,
                mejoras_json TEXT NOT NULL,
                comentarios TEXT NOT NULL,
                discrepancias_json TEXT NOT NULL DEFAULT '[]'
            );
            """
        )


def _persist(
    state: dict[str, Any], evaluation: dict[str, Any], status: str
) -> None:
    initialize_db()
    with _connect() as connection:
        connection.execute(
            """INSERT INTO actividades (id, descripcion, rubrica)
               VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
               descripcion=excluded.descripcion, rubrica=excluded.rubrica""",
            (state["activity_id"], state["activity_desc"], state["rubric"]),
        )
        connection.execute(
            """INSERT INTO estudiantes (id, nombre) VALUES (?, ?)
               ON CONFLICT(id) DO UPDATE SET nombre=excluded.nombre""",
            (state["student_id"], state.get("student_name", "Sin nombre")),
        )
        connection.execute(
            """INSERT INTO evaluaciones (
                   id, actividad_id, estudiante_id, puntaje, puntaje_maximo,
                   criterios_json, transcripcion_profesor, status
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   puntaje=excluded.puntaje,
                   puntaje_maximo=excluded.puntaje_maximo,
                   criterios_json=excluded.criterios_json,
                   transcripcion_profesor=excluded.transcripcion_profesor,
                   status=excluded.status,
                   updated_at=CURRENT_TIMESTAMP""",
            (
                state["evaluation_id"],
                state["activity_id"],
                state["student_id"],
                evaluation.get("score"),
                evaluation.get("max_score"),
                json.dumps(evaluation.get("criteria", []), ensure_ascii=False),
                state.get("teacher_transcript", ""),
                status,
            ),
        )
        connection.execute(
            """INSERT INTO feedback (
                   evaluacion_id, fortalezas_json, mejoras_json, comentarios,
                   discrepancias_json
               ) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(evaluacion_id) DO UPDATE SET
                   fortalezas_json=excluded.fortalezas_json,
                   mejoras_json=excluded.mejoras_json,
                   comentarios=excluded.comentarios,
                   discrepancias_json=excluded.discrepancias_json""",
            (
                state["evaluation_id"],
                json.dumps(evaluation.get("strengths", []), ensure_ascii=False),
                json.dumps(evaluation.get("improvements", []), ensure_ascii=False),
                str(evaluation.get("feedback", "")),
                json.dumps(evaluation.get("discrepancies", []), ensure_ascii=False),
            ),
        )


def create_draft_evaluation(state: dict[str, Any]) -> None:
    _persist(state, state["final_evaluation"], "draft")


def update_evaluation(
    state: dict[str, Any], evaluation: dict[str, Any], approved: bool
) -> None:
    _persist(state, evaluation, "approved" if approved else "draft")
