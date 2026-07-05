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

    def test_comparator_chart_label_modal_and_charting_route_are_wired(self):
        html = (self.root / "LLM" / "index.html").read_text(encoding="utf-8")
        app = (self.root / "LLM" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="comparatorChartLabelsModal"', html)
        self.assertIn('id="comparatorChartTitleInput"', html)
        self.assertIn('id="comparatorChartXAxisInput"', html)
        self.assertIn('id="comparatorChartYAxisInput"', html)
        self.assertIn('comparatorChartLabelsModal: document.querySelector("#comparatorChartLabelsModal")', app)
        self.assertIn('requestComparatorJson("/charting"', app)
        self.assertIn('COMPARATOR_CHART_LABEL_TOOL_KEY', app)
        self.assertIn('openComparatorChartLabelsModal', app)
        self.assertIn('downloadComparatorChartImage', app)
        self.assertIn('comparatorChartExportOption', app)
        self.assertIn('COMPARATOR_TRANSPARENT_BACKGROUND', app)
        self.assertIn('installComparatorLocalLegend(chart, chartNode, nextOption.series || [])', app)
        self.assertIn('exportOptionFactory: () => buildOption({ publicationMode: true })', app)
        self.assertIn('COMPARATOR_CHART_EXPORT_WIDTH = 1280', app)
        self.assertIn('COMPARATOR_CHART_EXPORT_HEIGHT = 760', app)
        self.assertIn('COMPARATOR_PUBLICATION_BORDER_WIDTH = 3', app)
        self.assertIn('chart.resize({', app)
        self.assertNotIn('backgroundColor: "#ffffff",\n    excludeComponents: ["toolbox", "dataZoom", "brush"]', app)
        self.assertNotIn('publicationMode: true,\n          chartKey: "embeddingProjection"', app)
        self.assertNotIn('publicationMode: true,\n      chartKey: "embeddingOverlay"', app)

    def test_comparator_instance_label_modal_and_route_are_wired(self):
        html = (self.root / "LLM" / "index.html").read_text(encoding="utf-8")
        app = (self.root / "LLM" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="editComparatorInstanceLabelsButton"', html)
        self.assertIn('id="comparatorInstanceLabelsModal"', html)
        self.assertIn('id="comparatorInstanceLabelsModalBody"', html)
        self.assertIn('editComparatorInstanceLabelsButton: document.querySelector("#editComparatorInstanceLabelsButton")', app)
        self.assertIn('comparatorInstanceLabelsModal: document.querySelector("#comparatorInstanceLabelsModal")', app)
        self.assertIn('openComparatorInstanceLabelsModal', app)
        self.assertIn('saveComparatorInstanceLabels', app)
        self.assertIn('/instance-labels', app)
        self.assertIn('dom.editComparatorInstanceLabelsButton?.addEventListener("click", openComparatorInstanceLabelsModal)', app)

    def test_comparator_commercial_price_metrics_are_after_output_tokens(self):
        app = (self.root / "LLM" / "app.js").read_text(encoding="utf-8")

        output_tokens_index = app.index('label: "Tokens salida"')
        price_metric_id = 'id: "gpt55CommercialExecutionPrice"'
        price_label = 'label: "Precio comercial (GPT-5.5) de 1 ejecución"'
        price_metric_index = app.index(price_metric_id)
        price_metric_block = app[price_metric_index:price_metric_index + 900]
        haiku_metric_id = 'id: "haiku45CommercialExecutionPrice"'
        haiku_label = 'label: "Precio comercial (Haiku 4.5) de 1 ejecución"'
        haiku_metric_index = app.index(haiku_metric_id)
        haiku_metric_block = app[haiku_metric_index:haiku_metric_index + 900]

        self.assertGreater(price_metric_index, output_tokens_index)
        self.assertIn(price_metric_id, price_metric_block)
        self.assertIn(price_label, price_metric_block)
        self.assertIn('kind: "cost"', price_metric_block)
        self.assertIn('direction: "min"', price_metric_block)
        self.assertIn('comparatorGpt55CommercialExecutionCost(cost)', price_metric_block)
        self.assertIn('formatComparatorGpt55CommercialExecutionCost', price_metric_block)
        self.assertIn("tarifas short context", price_metric_block)
        self.assertNotIn("272K", price_metric_block)
        self.assertNotIn("long context", price_metric_block)
        self.assertIn('from "./comparator_cost_helpers.mjs"', app)
        self.assertGreater(haiku_metric_index, price_metric_index)
        self.assertIn(haiku_metric_id, haiku_metric_block)
        self.assertIn(haiku_label, haiku_metric_block)
        self.assertIn('kind: "cost"', haiku_metric_block)
        self.assertIn('direction: "min"', haiku_metric_block)
        self.assertIn('comparatorHaiku45CommercialExecutionCost(cost)', haiku_metric_block)
        self.assertIn('formatComparatorHaiku45CommercialExecutionCost', haiku_metric_block)
        self.assertIn("Anthropic Claude Haiku 4.5", haiku_metric_block)

    def test_pareto_hypervolume_toggle_and_publication_styles_are_wired(self):
        app = (self.root / "LLM" / "app.js").read_text(encoding="utf-8")

        self.assertIn('COMPARATOR_PARETO_HV_TOOL_KEY', app)
        self.assertIn('showComparatorHypervolumeAreaByScope', app)
        self.assertIn('Mostrar área de hipervolumen', app)
        self.assertIn('hideMetricBadgeOnExport', app)
        self.assertIn('exportOptionFactory: () => buildOption({ exportMode: true, publicationMode: true })', app)
        self.assertIn('exportOptionFactory: () => buildFrontOption({ exportMode: true, publicationMode: true })', app)
        self.assertIn('fixedBadge: exportMode ? null : diagnosticsBadge', app)
        self.assertIn('publicationMode: Boolean(options.publicationMode)', app)
        self.assertIn('COMPARATOR_SELECTED_STAR_SYMBOL', app)
        self.assertIn('borderColor: "#111111"', app)
        self.assertIn('right: publicationMode ? legendLayout.gridRight : fixedBadgeDimensions ? fixedBadgeDimensions.width + 48 : 24', app)
        self.assertIn('right: 24,\n      top: 88,', app)
        self.assertNotIn('publicationMode: true,\n          chartKey: "pareto"', app)
        self.assertNotIn('publicationMode: true,\n      chartKey: "internalBmopsoPareto"', app)


if __name__ == "__main__":
    unittest.main()
