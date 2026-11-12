"""Public-repo hygiene check for notebooks (scripts/tools/check_notebooks.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools import check_notebooks as cn

REPO_ROOT = Path(__file__).resolve().parents[1]

# Built at runtime so this file itself never contains a local-looking path.
WIN_PATH = "\\".join(["C:", "Users", "jdoe", "project", "frames", "img.png"])
HOME_PATH = "/".join(["", "home", "jdoe", "data", "img.png"])
APPDATA_PATH = "\\".join(["see AppData", "Local", "Temp", "x"])

PNG = {"image/png": "iVBORw0KGgoAAAANSUhEUgAAAAE=", "text/plain": ["<Figure size 640x480>"]}


def _code(source: str, outputs: list | None = None, count: int = 1) -> dict:
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.splitlines(keepends=True),
    }


def _image() -> dict:
    return {"output_type": "display_data", "data": dict(PNG), "metadata": {}}


def _stream(text: str, name: str = "stdout") -> dict:
    return {"output_type": "stream", "name": name, "text": [text]}


def _notebook(*cells: dict) -> dict:
    return {"cells": list(cells), "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _write(tmp_path: Path, nb: dict, name: str = "01_test.ipynb") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    return path


def _rules(report) -> list[tuple[str, str, bool]]:
    return sorted((f.level, f.rule, f.fixable) for f in report.findings)


# ---------------------------------------------------------------------------
# check_notebook
# ---------------------------------------------------------------------------


def test_frame_image_in_an_image_open_cell_is_flagged(tmp_path):
    nb = _notebook(
        _code("from PIL import Image\nimg = Image.open(path)\nplt.imshow(img)", [_image()])
    )
    report = cn.check_notebook(_write(tmp_path, nb), nb=nb)
    assert _rules(report) == [("FAIL", "frame-image", True)]
    assert "Image.open" in report.findings[0].detail


def test_frame_helper_calls_are_flagged(tmp_path):
    nb = _notebook(
        _code("def show_frame(p):\n    plt.imshow(cv2.imread(p))\n"),
        _code("x = 1"),
        _code("show_frame(first_path)", [_image()]),
    )
    report = cn.check_notebook(_write(tmp_path, nb), nb=nb)
    assert _rules(report) == [("FAIL", "frame-image", True)]
    assert report.findings[0].cell == 2
    assert cn.frame_cells(nb["cells"]) == {
        0: "source matches 'imread'",
        2: "calls frame helper 'show_frame'",
    }


def test_plots_outside_frame_cells_are_allowed(tmp_path):
    nb = _notebook(_code("plt.plot([1, 2, 3])\nplt.show()", [_image()]))
    assert cn.check_notebook(_write(tmp_path, nb), nb=nb).findings == []


def test_every_image_is_a_frame_in_explainability_notebooks(tmp_path):
    nb = _notebook(_code("plt.plot([1, 2, 3])", [_image()]))
    report = cn.check_notebook(_write(tmp_path, nb, "explainability.ipynb"), nb=nb)
    assert _rules(report) == [("FAIL", "frame-image", True)]


@pytest.mark.parametrize("text", [WIN_PATH, HOME_PATH, APPDATA_PATH])
def test_local_path_in_an_output_is_flagged(tmp_path, text):
    nb = _notebook(_code("run()", [_stream(f"UserWarning: {text}\n", "stderr")]))
    report = cn.check_notebook(_write(tmp_path, nb), nb=nb)
    assert ("FAIL", "local-path", True) in _rules(report)
    # Matches are redacted in the report.
    assert all(text not in f.detail for f in report.findings)


def test_local_path_in_a_source_needs_a_manual_fix(tmp_path):
    nb = _notebook(_code(f"DATA = r'{WIN_PATH}'"))
    report = cn.check_notebook(_write(tmp_path, nb), nb=nb)
    assert report.failures and all(not f.fixable for f in report.failures)
    assert {f.rule for f in report.failures} == {"local-path"}


def test_patient_table_output_is_flagged(tmp_path):
    html = "<table><thead><tr><th>patient_id</th><th>n</th></tr></thead></table>"
    nb = _notebook(
        _code(
            "df",
            [
                {
                    "output_type": "execute_result",
                    "execution_count": 1,
                    "metadata": {},
                    "data": {"text/html": [html], "text/plain": ["..."]},
                }
            ],
        )
    )
    report = cn.check_notebook(_write(tmp_path, nb), nb=nb)
    assert ("FAIL", "patient-table", True) in _rules(report)


def test_extra_forbidden_patterns(tmp_path):
    nb = _notebook(_code("print(1)", [_stream("connected to lab-gpu-07\n")]))
    extra = [cn.Pattern("local-path", "host", cn.re.compile(r"lab-gpu-\d+"))]
    report = cn.check_notebook(_write(tmp_path, nb), nb=nb, extra_patterns=extra)
    assert _rules(report) == [("FAIL", "local-path", True)]


def test_clean_notebook(tmp_path):
    nb = _notebook(_code("print('ok')", [_stream("ok\n")]), _code("plt.plot([1])", [_image()]))
    assert cn.check_notebook(_write(tmp_path, nb), nb=nb).findings == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_exit_codes_and_fix(tmp_path, capsys):
    ok_output = _stream("fine\n")
    nb = _notebook(
        _code("img = Image.open(p)\nplt.imshow(img)", [ok_output, _image()]),
        _code("warn()", [_stream(f"{WIN_PATH}: warning\n", "stderr")]),
    )
    path = _write(tmp_path, nb)
    argv = [str(path), "--no-local-identity"]

    assert cn.main(argv) == 1
    assert cn.main([*argv, "--fix"]) == 0  # both failures are removable outputs
    fixed = json.loads(path.read_text(encoding="utf-8"))
    assert fixed["cells"][0]["outputs"] == [ok_output]
    assert fixed["cells"][1]["outputs"] == []
    assert fixed["cells"][0]["source"] == nb["cells"][0]["source"]
    assert cn.main(argv) == 0
    out = capsys.readouterr().out
    assert WIN_PATH not in out  # never echoes the match


def test_main_manual_failures_survive_fix(tmp_path):
    path = _write(tmp_path, _notebook(_code(f"p = '{HOME_PATH}'")))
    assert cn.main([str(path), "--no-local-identity", "--fix"]) == 1


def test_main_bad_inputs(tmp_path):
    bad = tmp_path / "broken.ipynb"
    bad.write_text("{not json", encoding="utf-8")
    assert cn.main([str(bad), "--no-local-identity"]) == 2
    assert cn.main([str(tmp_path / "missing.ipynb")]) == 2
    assert cn.main([str(tmp_path), "--forbid", "("]) == 2


@pytest.mark.integration
def test_committed_notebooks_are_clean():
    # The release gate: no local paths, study IDs, patient tables or frame images.
    notebooks = REPO_ROOT / "notebooks"
    if not any(notebooks.glob("*.ipynb")):
        pytest.skip("no notebooks")
    assert cn.main([str(notebooks)]) == 0
