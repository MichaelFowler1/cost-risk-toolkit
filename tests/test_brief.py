# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""The PowerPoint briefing each run writes: it opens with the bottom line,
says what the analysis says and nothing more, and carries the assumptions."""
import pytest

pptx = pytest.importorskip("pptx")

from cost_core import cli  # noqa: E402

TOPICS = ["evm", "schedule", "jcl", "aoa", "portfolio"]


def _texts(path):
    prs = pptx.Presentation(str(path))
    slides = []
    for slide in prs.slides:
        words = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                words.append(shape.text_frame.text)
            if shape.has_table:
                words += [c.text for row in shape.table.rows for c in row.cells]
        slides.append("\n".join(words))
    return prs, slides


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    root = tmp_path_factory.mktemp("briefs")
    for topic in TOPICS:
        if topic == "portfolio":
            pytest.importorskip("pulp")
        cli.main(["demo", topic, "--out", str(root / topic)])
    return root


@pytest.mark.parametrize("topic", TOPICS)
def test_every_run_writes_a_widescreen_brief_that_opens_with_the_bottom_line(demos, topic):
    prs, slides = _texts(demos / topic / "brief.pptx")
    assert prs.slide_width > prs.slide_height  # 16:9
    assert slides[1].startswith("Bottom line")
    assert "Assumptions and method" in slides[-1]
    assert all("listed above" not in s for s in slides)  # no terminal wording


def test_the_evm_brief_carries_the_forecast_and_the_warning(demos):
    from cost_core.evm import EvmData, forecast
    from cost_core.examples import example_path
    from cost_core.plain import _money
    import numpy as np

    _, slides = _texts(demos / "evm" / "brief.pptx")
    text = "\n".join(slides)
    data = EvmData.read(example_path("evm"))
    fc = forecast(data, n_iter=20000, seed=0)
    assert _money(float(np.quantile(fc.eac, 0.5)), "$K") in slides[1]
    assert "Warning signs" in text and "Contractor EAC below every independent EAC" in text
    assert "on the Warning signs slide" in slides[1]
    prs = pptx.Presentation(str(demos / "evm" / "brief.pptx"))
    pictures = [s for slide in prs.slides for s in slide.shapes if s.shape_type == 13]
    assert pictures, "the forecast chart is on a slide"


def test_the_schedule_brief_colours_failures_and_orders_what_to_fix(demos):
    prs, slides = _texts(demos / "schedule" / "brief.pptx")
    table = next(s for slide in prs.slides for s in slide.shapes
                 if s.has_table and "Threshold" in s.table.cell(0, 4).text).table
    fail = next(table.cell(r, 2) for r in range(1, 15) if table.cell(r, 2).text == "FAIL")
    assert fail.text_frame.paragraphs[0].font.color.rgb == pptx.dml.color.RGBColor(0xC0, 0, 0)
    fix = next(s for s in slides if s.startswith("Tasks to fix first"))
    # Missing logic before lags: what breaks the dates comes first.
    assert fix.index("Logic") < fix.index("Lags")


def test_without_python_pptx_the_run_says_how_to_get_a_brief(tmp_path, monkeypatch, capsys):
    import cost_core.reporting.brief as brief

    def missing():
        raise ImportError("no pptx")

    monkeypatch.setattr(brief, "_pptx", missing)
    cli.main(["demo", "jcl", "--out", str(tmp_path / "j")])
    out = capsys.readouterr().out
    assert 'pip install "cost-core[plots]"' in out
    assert not (tmp_path / "j" / "brief.pptx").exists()
    assert (tmp_path / "j" / "report.xlsx").exists()
