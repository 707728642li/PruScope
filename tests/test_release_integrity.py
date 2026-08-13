from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_audit_module():
    path = ROOT / "tools/audit_metrics.py"
    spec = importlib.util.spec_from_file_location("audit_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetricAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load_audit_module()

    def test_valid_fixture_matches_expected(self):
        rows = self.audit.load_rows(ROOT / "examples/synthetic_metric_audit/predictions.csv")
        actual = self.audit.summarize(rows)
        expected = json.loads(
            (ROOT / "examples/synthetic_metric_audit/expected_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual([], self.audit._compare(expected, actual))

    def test_invalid_fixture_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must equal reference_count"):
            self.audit.load_rows(ROOT / "examples/synthetic_metric_audit/invalid_predictions.csv")

    def test_cli_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.json"
            output.write_text("existing\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/audit_metrics.py"),
                    "--input",
                    str(ROOT / "examples/synthetic_metric_audit/predictions.csv"),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(2, completed.returncode)
            self.assertIn("refusing to overwrite", completed.stderr)


class MetadataTests(unittest.TestCase):
    def test_metadata_and_citation_version_align(self):
        metadata = (ROOT / "release/project_metadata.yml").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("public_version: 1.0.0-rc.1", metadata)
        self.assertIn("version: 1.0.0-rc.1", citation)
        repository = 'https://github.com/707728642li/PruScope'
        self.assertIn(repository, metadata)
        self.assertIn(repository, citation)
        self.assertIn("release_tag: null", metadata)
        self.assertIn("code_spdx: null", metadata)

    def test_paper_assets_exist(self):
        self.assertTrue((ROOT / "paper/PRUSCOPE_MANUSCRIPT_SOURCE.md").is_file())
        self.assertTrue((ROOT / "paper/PruScope_Plant_Phenomics_Manuscript.pdf").is_file())
        for stem in (
            "fig1_experimental_design_and_pruscope_architecture",
            "fig2_adverse_plum_scenes_and_detection_evidence",
            "fig3_detector_ablation_statistics_and_efficiency",
            "fig4_dcoh_stage_discrimination_and_robustness",
            "fig5_qualitative_detection_and_count_errors",
            "fig6_competitive_benchmark_and_end_to_end_stage",
            "figS1_mars_postaudit_optimization_and_confirmation",
            "figS2_public_citdet_external_validation",
            "figS3_postreview_difficult_target_update",
        ):
            for suffix in (".png", ".pdf", ".svg"):
                self.assertTrue((ROOT / "paper/figures" / f"{stem}{suffix}").is_file())
        for suffix in (".png", ".pdf"):
            self.assertTrue(
                (ROOT / "paper/figures" / f"figS4_dart_microfruit_refinement{suffix}").is_file()
            )


if __name__ == "__main__":
    unittest.main()
