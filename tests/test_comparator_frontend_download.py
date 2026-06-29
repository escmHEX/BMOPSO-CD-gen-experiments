from __future__ import annotations

import re
import unittest
from pathlib import Path


class ComparatorFrontendDownloadTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_download_button_is_after_logs_without_tab_behavior(self):
        html = (self.root / "LLM" / "index.html").read_text(encoding="utf-8")
        logs_index = html.index('data-comparator-tab="logs" aria-selected="false">Logs</button>')
        match = re.search(
            r'<button(?P<attrs>[^>]*id="downloadComparatorRunDataButton"[^>]*)>Descargar datos</button>',
            html[logs_index:],
        )

        self.assertIsNotNone(match)
        attrs = match.group("attrs")
        self.assertIn('class="secondary"', attrs)
        self.assertNotIn("comparator-tab", attrs)
        self.assertNotIn("data-comparator-tab", attrs)

    def test_download_button_uses_encoded_run_id_download_route(self):
        app = (self.root / "LLM" / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            'downloadComparatorRunDataButton: document.querySelector("#downloadComparatorRunDataButton")',
            app,
        )
        self.assertIn('${COMPARATOR_API}/runs/${encodeURIComponent(runId)}/download', app)
        self.assertIn(
            'dom.downloadComparatorRunDataButton?.addEventListener("click", downloadComparatorRunData)',
            app,
        )

    def test_recontinue_button_is_next_to_clear_and_uses_recontinue_route(self):
        html = (self.root / "LLM" / "index.html").read_text(encoding="utf-8")
        clear_index = html.index('id="clearComparatorButton"')
        recontinue_index = html.index('id="recontinueComparatorButton"')
        self.assertGreater(recontinue_index, clear_index)
        self.assertLess(recontinue_index - clear_index, 180)
        self.assertIn(">Re-continuar</button>", html[recontinue_index:recontinue_index + 120])

        app = (self.root / "LLM" / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'recontinueComparatorButton: document.querySelector("#recontinueComparatorButton")',
            app,
        )
        self.assertIn('/recontinue', app)
        self.assertIn(
            'dom.recontinueComparatorButton?.addEventListener("click", recontinueComparatorRun)',
            app,
        )


if __name__ == "__main__":
    unittest.main()
