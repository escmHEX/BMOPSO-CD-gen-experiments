from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE_DB = {
    "items": [
        {
            "id": "default-action-center-smb",
            "label": "Default - action center SMB",
            "paper": "Default web",
            "ref": "default",
            "source": "Default usado por la web",
            "text": "Our action center has been updated with more information about restaurant shutdowns and disaster financing options for SMBs.",
            "createdAt": "2026-05-23T00:00:00Z",
        }
    ]
}


class ReferenceTextStore:
    def __init__(self, root: Path, relative_path: str = "data/reference_texts.json") -> None:
        self.path = root / relative_path
        self._lock = threading.RLock()

    def list_texts(self) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._read_payload()
            return [self._public_item(item) for item in payload["items"]]

    def save_text(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = self._required_text(payload.get("text"), "text", 20_000)
        label = self._optional_text(payload.get("label"), 120) or self._label_from_text(text)
        paper = self._optional_text(payload.get("paper"), 120) or "Usuario"
        ref = self._optional_text(payload.get("ref"), 60) or ""
        source = self._optional_text(payload.get("source"), 160) or "Guardado desde la web"
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        item = {
            "id": self._unique_id(label),
            "label": label,
            "paper": paper,
            "ref": ref,
            "source": source,
            "text": text,
            "createdAt": now,
        }
        with self._lock:
            db = self._read_payload()
            db["items"].append(item)
            self._write_payload(db)
        return self._public_item(item)

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            self._write_payload(DEFAULT_REFERENCE_DB)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"No se pudo leer la BD de textos de referencia: {error}") from error
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("La BD de textos de referencia debe tener una lista items.")
        return {"items": [self._normalize_item(item) for item in items if isinstance(item, dict)]}

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        last_error: OSError | None = None
        for attempt in range(6):
            temp_path = self.path.with_name(f"{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp")
            try:
                temp_path.write_text(encoded, encoding="utf-8")
                temp_path.replace(self.path)
                return
            except OSError as error:
                last_error = error
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(0.05 * (attempt + 1))
        if last_error is not None:
            raise last_error

    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        text = self._required_text(item.get("text"), "text", 20_000)
        label = self._optional_text(item.get("label"), 120) or self._label_from_text(text)
        return {
            "id": self._optional_text(item.get("id"), 160) or self._unique_id(label),
            "label": label,
            "paper": self._optional_text(item.get("paper"), 120) or "",
            "ref": self._optional_text(item.get("ref"), 60) or "",
            "source": self._optional_text(item.get("source"), 160) or "",
            "text": text,
            "createdAt": self._optional_text(item.get("createdAt"), 40) or "",
        }

    def _public_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item["id"],
            "label": item["label"],
            "paper": item["paper"],
            "ref": item["ref"],
            "source": item["source"],
            "text": item["text"],
            "createdAt": item["createdAt"],
        }

    def _required_text(self, value: Any, label: str, max_length: int) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{label} is required.")
        if len(text) > max_length:
            raise ValueError(f"{label} must be at most {max_length} characters.")
        return text

    def _optional_text(self, value: Any, max_length: int) -> str:
        text = str(value or "").strip()
        if len(text) > max_length:
            raise ValueError(f"Text fields must be at most {max_length} characters.")
        return text

    def _label_from_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        return normalized[:76] + ("..." if len(normalized) > 76 else "")

    def _unique_id(self, label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:60] or "reference"
        return f"{slug}-{uuid.uuid4().hex[:8]}"
