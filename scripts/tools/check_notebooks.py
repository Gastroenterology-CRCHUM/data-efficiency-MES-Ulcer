"""Public-repo hygiene check for the notebooks, which are committed with their outputs.

Fails when an output (or, for some rules, the source) would publish something private:

    local-path     local path, user or host name (outputs, sources, metadata)
    study-id       study ID ``vid_NN_NNN...`` (outputs, sources)
    patient-table  rendered table with a ``patient_id`` column or index
    frame-image    image output of a cell that reads or plots dataset frames, or of an
                   explainability notebook (``*explainab*``)
    size           (warning) notebook larger than ``--max-size-mb`` (default 5 MB)
    attachment     (warning) markdown cell with embedded images

``--fix`` removes only the offending outputs; sources and metadata are never modified.
Matches are redacted in the report (``--show-matches`` prints them, for local use only).
Standard library only.

    python scripts/tools/check_notebooks.py                 # notebooks/*.ipynb
    python scripts/tools/check_notebooks.py --fix
    python -m scripts.tools.check_notebooks --forbid "my-lab-server"

Exit code: 0 = no failure, 1 = at least one failure left, 2 = bad argument or unreadable notebook.
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import json
import os
import re
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOK_DIR = REPO_ROOT / "notebooks"

# ── Rule definitions ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pattern:
    """A text pattern that must not appear in a public notebook."""

    rule: str
    label: str
    regex: re.Pattern[str]
    sensitive: bool = False  # never echo the match (current user / host name)


_SEP = r"[\\/]+"  # one or more path separators (also matches escaped "\\\\")

LOCAL_PATH_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "local-path",
        "Windows user directory",
        re.compile(
            rf"(?<![A-Za-z])[A-Za-z]:{_SEP}(?:Users|Documents and Settings){_SEP}[^\\/\s'\"]+", re.I
        ),
    ),
    Pattern(
        "local-path",
        "Windows absolute path",
        # Drive letter + at least two components (C:\Projects\repo\...); the
        # user directory is reported by the rule above.
        re.compile(
            rf"(?<![A-Za-z0-9])[A-Za-z]:{_SEP}(?!(?:Users|Documents and Settings)\b)"
            rf"[\w.$~-][\w .$~-]*{_SEP}[\w.$~-]",
            re.I,
        ),
    ),
    Pattern("local-path", "POSIX home directory", re.compile(r"(?<![\w.-])/home/[A-Za-z0-9._-]+")),
    Pattern(
        "local-path",
        "macOS / Git-Bash / WSL user directory",
        # /Users/<name>, /c/Users/<name>, /mnt/c/Users/<name>, /cygdrive/c/Users/<name>
        re.compile(r"(?<![\w.-])(?:/mnt|/cygdrive)?(?:/[A-Za-z])?/Users/[A-Za-z0-9._-]+", re.I),
    ),
    Pattern("local-path", "AppData directory", re.compile(rf"\bAppData{_SEP}", re.I)),
    Pattern(
        "local-path",
        "conda environment path",
        re.compile(rf"\b(?:miniconda3?|anaconda3?|miniforge3?|mambaforge|conda){_SEP}envs\b", re.I),
    ),
)

STUDY_ID_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "study-id", "study ID (vid_NN_NNN...)", re.compile(r"(?<![A-Za-z0-9])vid_\d{2}_\d{3,}")
    ),
)

# A table exposing patient identifiers: column/index header or Series name.
_PATIENT = r"patient_ids?"
PATIENT_TABLE_REGEXES: dict[str, re.Pattern[str]] = {
    "html": re.compile(rf"<th\b[^>]*>\s*{_PATIENT}\s*</th>", re.I),
    "latex": re.compile(r"patient\\_ids?\b", re.I),
    # DataFrame text repr: header token padded by 2+ spaces (or line edge);
    # Series repr: "Name: patient_id". A prose sentence has single spaces.
    "text": re.compile(rf"(?m)(?:^|\s{{2,}}){_PATIENT}(?:\s{{2,}}|[ \t]*$)|\bName:\s*{_PATIENT}\b"),
}

# Source patterns marking a cell that reads or renders dataset frames.
FRAME_SOURCE_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bImage\.open\b",
        r"\bimread\b",  # cv2.imread, plt.imread, imageio.imread ...
        r"\b(?:read|decode)_image\b",  # torchvision.io
        r"\bshow_sample_images\b",
        r"\bplot_saliency\b",
        r"saliency",
        r"grad_?cam",
        r"\w*attention\w*\s*\(",  # attention-map helpers / get_last_selfattention
        r"\bvisualize_predictions\b",
        r"\bUlcerDataset\b",
        r"\b(?:ResizeWithPad|CLAHE_Y)\b",  # frame preprocessing previews
    )
)

# Notebooks in which every figure is a dataset frame (explainability).
DEFAULT_FRAME_NOTEBOOK_GLOBS: tuple[str, ...] = ("*explainab*.ipynb",)

# Names too generic to be treated as the local user / host name.
_GENERIC_NAMES = frozenset(
    {
        "user",
        "users",
        "admin",
        "administrator",
        "root",
        "runner",
        "jovyan",
        "ubuntu",
        "localhost",
        "home",
        "public",
        "default",
        "guest",
        "vagrant",
        "codespace",
        "docker",
    }
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_DEF = re.compile(r"(?m)^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)")
_IMG_TAG = re.compile(r"<img\b", re.I)

DEFAULT_MAX_SIZE_MB = 5.0


def local_identity_patterns() -> list[Pattern]:
    """Patterns for the current user name and host name (never echoed)."""
    names: set[str] = set()
    for getter in (getpass.getuser, socket.gethostname, lambda: Path.home().name):
        try:
            value = getter()
        except Exception:  # best effort, platform dependent
            continue
        if value:
            names.add(value)
    for var in ("USERNAME", "USER", "COMPUTERNAME", "HOSTNAME"):
        if os.environ.get(var):
            names.add(os.environ[var])
    names |= {n.split(".")[0] for n in names if "." in n}  # short host name

    patterns = []
    for name in sorted(names):
        if len(name) < 3 or name.lower() in _GENERIC_NAMES:
            continue
        regex = re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", re.I)
        patterns.append(Pattern("local-path", "current user / host name", regex, sensitive=True))
    return patterns


# ── Findings ─────────────────────────────────────────────────────────────────


@dataclass
class Finding:
    """One problem found in a notebook."""

    level: str  # "FAIL" | "WARN"
    rule: str
    where: str  # human-readable location
    detail: str
    cell: int | None = None
    output: int | None = None  # index in cell["outputs"]: removable by --fix

    @property
    def fixable(self) -> bool:
        return self.level == "FAIL" and self.output is not None


@dataclass
class NotebookReport:
    """Findings of one notebook, plus what --fix did."""

    path: Path
    size_mb: float
    findings: list[Finding] = field(default_factory=list)
    removed_outputs: int = 0
    error: str | None = None

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "FAIL"]

    @property
    def remaining_failures(self) -> list[Finding]:
        if self.removed_outputs:
            return [f for f in self.failures if not f.fixable]
        return self.failures


# ── Helpers ──────────────────────────────────────────────────────────────────


def _join(value: Any) -> str:
    """Notebook multiline strings are either a str or a list of str."""
    if isinstance(value, list):
        return "".join(str(v) for v in value)
    return "" if value is None else str(value)


def _redact(text: str, pattern: Pattern, show: bool) -> str:
    if pattern.sensitive:
        return "<redacted>"
    if show:
        return repr(text)
    return repr(text[:6] + "..." if len(text) > 8 else text[:2] + "...")


def _search(text: str, patterns: list[Pattern], show: bool) -> list[tuple[Pattern, str]]:
    """Return (pattern, redacted first match) for each pattern found in text."""
    hits = []
    for pattern in patterns:
        match = pattern.regex.search(text)
        if match:
            hits.append((pattern, _redact(match.group(0), pattern, show)))
    return hits


def output_texts(output: dict[str, Any]) -> list[tuple[str, str]]:
    """Text payloads of an output as (kind, text); image payloads are skipped."""
    otype = output.get("output_type")
    if otype == "stream":
        return [(f"stream:{output.get('name', 'stdout')}", _join(output.get("text")))]
    if otype == "error":
        tb = "\n".join(_join(line) for line in output.get("traceback", []))
        text = f"{output.get('ename', '')}: {output.get('evalue', '')}\n{tb}"
        return [("error", _ANSI.sub("", text))]
    texts = []
    for mime, payload in (output.get("data") or {}).items():
        if mime.startswith("image/") and mime != "image/svg+xml":
            continue  # base64 payload
        if isinstance(payload, (dict, list)) and not mime.startswith("text/"):
            payload = json.dumps(payload)
        texts.append((mime, _ANSI.sub("", _join(payload))))
    return texts


def image_mime(output: dict[str, Any]) -> str | None:
    """MIME type of the image rendered by an output, or None."""
    if output.get("output_type") not in ("display_data", "execute_result"):
        return None
    data = output.get("data") or {}
    for mime in data:
        if mime.startswith("image/"):
            return mime
    if _IMG_TAG.search(_join(data.get("text/html"))):
        return "text/html <img>"
    return None


def _cell_label(index: int, cell: dict[str, Any]) -> str:
    count = cell.get("execution_count")
    return f"cell[{index}]" + (f" In[{count}]" if count is not None else "")


def frame_cells(cells: list[dict[str, Any]]) -> dict[int, str]:
    """Indices of code cells that read/render dataset frames -> reason.

    A cell qualifies when its source matches FRAME_SOURCE_REGEXES, or when it
    calls a function/class defined in a qualifying cell (propagated to a fixpoint).
    """
    sources = {
        i: _join(c.get("source")) for i, c in enumerate(cells) if c.get("cell_type") == "code"
    }
    flagged: dict[int, str] = {}
    for i, src in sources.items():
        for regex in FRAME_SOURCE_REGEXES:
            match = regex.search(src)
            if match:
                flagged[i] = f"source matches {match.group(0).strip()!r}"
                break

    helpers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for i in flagged:
            helpers |= set(_DEF.findall(sources[i]))
        for i, src in sources.items():
            if i in flagged:
                continue
            for name in helpers:
                if re.search(rf"(?<![\w.]){re.escape(name)}\s*\(", src):
                    flagged[i] = f"calls frame helper {name!r}"
                    changed = True
                    break
    return flagged


# ── Checks ───────────────────────────────────────────────────────────────────


def check_notebook(
    path: Path,
    *,
    extra_patterns: list[Pattern] | None = None,
    frame_notebook_globs: tuple[str, ...] = DEFAULT_FRAME_NOTEBOOK_GLOBS,
    max_size_mb: float = DEFAULT_MAX_SIZE_MB,
    show_matches: bool = False,
    nb: dict[str, Any] | None = None,
) -> NotebookReport:
    """Run every rule on one notebook and return its report (nothing is modified)."""
    report = NotebookReport(path=path, size_mb=path.stat().st_size / 1e6)
    if nb is None:
        try:
            nb = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            report.error = f"cannot read notebook: {exc}"
            return report

    text_patterns = [*LOCAL_PATH_PATTERNS, *STUDY_ID_PATTERNS, *(extra_patterns or [])]
    path_patterns = [p for p in text_patterns if p.rule != "study-id"]

    def add(level, rule, where, detail, cell=None, output=None):
        report.findings.append(Finding(level, rule, where, detail, cell, output))

    if report.size_mb > max_size_mb:
        add("WARN", "size", "file", f"{report.size_mb:.1f} MB > {max_size_mb:g} MB")

    for pattern, shown in _search(json.dumps(nb.get("metadata", {})), path_patterns, show_matches):
        add("FAIL", pattern.rule, "notebook metadata", f"{pattern.label} {shown} (fix by hand)")

    cells = nb.get("cells", [])
    frames = frame_cells(cells)
    frame_notebook = any(fnmatch.fnmatch(path.name, g) for g in frame_notebook_globs)

    for ci, cell in enumerate(cells):
        label = _cell_label(ci, cell)

        for pattern, shown in _search(_join(cell.get("source")), text_patterns, show_matches):
            add(
                "FAIL",
                pattern.rule,
                f"{label} source",
                f"{pattern.label} {shown} (fix by hand)",
                ci,
            )
        cell_meta = json.dumps(cell.get("metadata", {}))
        for pattern, shown in _search(cell_meta, path_patterns, show_matches):
            add(
                "FAIL",
                pattern.rule,
                f"{label} metadata",
                f"{pattern.label} {shown} (fix by hand)",
                ci,
            )

        attachments = cell.get("attachments") or {}
        if attachments:
            add(
                "WARN",
                "attachment",
                label,
                f"{len(attachments)} embedded attachment(s): verify they are not frames",
                ci,
            )

        for oi, output in enumerate(cell.get("outputs", [])):
            where = f"{label} output {oi}"
            mime = image_mime(output)
            if mime and (ci in frames or frame_notebook):
                reason = frames.get(ci, "explainability notebook: every figure is a frame")
                add(
                    "FAIL",
                    "frame-image",
                    f"{where} ({mime})",
                    f"image in a frame cell ({reason})",
                    ci,
                    oi,
                )
                continue  # the whole output goes; no need to scan its text
            for kind, text in output_texts(output):
                for pattern, shown in _search(text, text_patterns, show_matches):
                    add(
                        "FAIL",
                        pattern.rule,
                        f"{where} ({kind})",
                        f"{pattern.label} {shown}",
                        ci,
                        oi,
                    )
                family = (
                    "html" if kind == "text/html" else "latex" if kind == "text/latex" else "text"
                )
                if kind != "text/markdown" and PATIENT_TABLE_REGEXES[family].search(text):
                    add(
                        "FAIL",
                        "patient-table",
                        f"{where} ({kind})",
                        "table with a patient_id column/index",
                        ci,
                        oi,
                    )
    return report


def remove_outputs(nb: dict[str, Any], findings: list[Finding]) -> int:
    """Remove the outputs referenced by fixable findings. Returns how many were removed."""
    targets: dict[int, set[int]] = {}
    for f in findings:
        if f.fixable:
            targets.setdefault(f.cell, set()).add(f.output)
    removed = 0
    for ci, output_ids in targets.items():
        outputs = nb["cells"][ci].get("outputs", [])
        nb["cells"][ci]["outputs"] = [o for oi, o in enumerate(outputs) if oi not in output_ids]
        removed += len(outputs) - len(nb["cells"][ci]["outputs"])
    return removed


def write_notebook(path: Path, nb: dict[str, Any], original: str) -> None:
    """Write nb back keeping the original indent, key order, escaping and newlines."""
    indent_match = re.search(r"\n([ \t]+)\S", original)
    indent: int | str = 1
    if indent_match:
        ws = indent_match.group(1)
        indent = len(ws) if set(ws) == {" "} else ws
    ensure_ascii = original.isascii() and "\\u" in original
    text = json.dumps(nb, indent=indent, ensure_ascii=ensure_ascii)
    if original.endswith("\n"):
        text += "\n"
    newline = "\r\n" if "\r\n" in original else "\n"
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _collect(paths: list[str]) -> list[Path]:
    if not paths:
        paths = [str(DEFAULT_NOTEBOOK_DIR)]
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found += sorted(q for q in p.glob("*.ipynb") if ".ipynb_checkpoints" not in q.parts)
        elif p.suffix == ".ipynb" and p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"not a notebook or directory: {raw}")
    return found


def _display(path: Path) -> str:
    for base in (Path.cwd(), REPO_ROOT):
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def _print_report(report: NotebookReport, fix: bool) -> None:
    fails, warns = report.failures, [f for f in report.findings if f.level == "WARN"]
    status = "ERROR" if report.error else "FAIL" if fails else "WARN" if warns else "OK"
    print(f"[{status:5}] {_display(report.path)}  ({report.size_mb:.2f} MB)")
    if report.error:
        print(f"        {report.error}")
    for f in report.findings:
        tag = f"{f.level} {f.rule}"
        fix_note = "" if f.level == "WARN" else " [fixable]" if f.fixable else " [manual]"
        print(f"        {tag:<19} {f.where}: {f.detail}{fix_note}")
    if fix and report.removed_outputs:
        print(f"        -> --fix removed {report.removed_outputs} output(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail if committed notebooks leak local paths, study IDs or dataset frames.",
    )
    parser.add_argument("paths", nargs="*", help="Notebooks or directories (default: notebooks/)")
    parser.add_argument("--fix", action="store_true", help="Remove the offending outputs in place")
    parser.add_argument(
        "--forbid",
        action="append",
        default=[],
        metavar="REGEX",
        help="Extra forbidden pattern (repeatable), e.g. an internal host name",
    )
    parser.add_argument(
        "--no-local-identity",
        action="store_true",
        help="Do not flag the current user / host name (generic patterns still apply)",
    )
    parser.add_argument(
        "--frame-notebook",
        action="append",
        default=None,
        metavar="GLOB",
        help="Notebook name glob where every image is a frame "
        f"(repeatable; default: {' '.join(DEFAULT_FRAME_NOTEBOOK_GLOBS)})",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=DEFAULT_MAX_SIZE_MB,
        help=f"Warn above this notebook size (default: {DEFAULT_MAX_SIZE_MB:g})",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Print matches in full (local debugging only; never paste publicly)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)

    try:
        extra = [
            Pattern("local-path", f"forbidden pattern {p!r}", re.compile(p)) for p in args.forbid
        ]
    except re.error as exc:
        print(f"ERROR: invalid --forbid regex: {exc}", file=sys.stderr)
        return 2
    if not args.no_local_identity:
        extra += local_identity_patterns()
    globs = tuple(args.frame_notebook) if args.frame_notebook else DEFAULT_FRAME_NOTEBOOK_GLOBS

    try:
        notebooks = _collect(args.paths)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not notebooks:
        print("No notebooks found.")
        return 0

    reports: list[NotebookReport] = []
    for path in notebooks:
        try:
            original = path.read_text(encoding="utf-8-sig")
            nb = json.loads(original)
        except (OSError, ValueError) as exc:
            report = NotebookReport(
                path=path, size_mb=path.stat().st_size / 1e6, error=f"cannot read notebook: {exc}"
            )
        else:
            report = check_notebook(
                path,
                extra_patterns=extra,
                frame_notebook_globs=globs,
                max_size_mb=args.max_size_mb,
                show_matches=args.show_matches,
                nb=nb,
            )
            if args.fix and any(f.fixable for f in report.findings):
                report.removed_outputs = remove_outputs(nb, report.findings)
                write_notebook(path, nb, original)
        reports.append(report)
        _print_report(report, args.fix)

    n_fail = sum(len(r.failures) for r in reports)
    n_fixable = sum(1 for r in reports for f in r.failures if f.fixable)
    n_warn = sum(1 for r in reports for f in r.findings if f.level == "WARN")
    n_left = sum(len(r.remaining_failures) for r in reports)
    n_error = sum(1 for r in reports if r.error)
    print(
        f"\n{len(reports)} notebook(s): {n_fail} failure(s) ({n_fixable} fixable with --fix), "
        f"{n_warn} warning(s)" + (f", {n_error} unreadable" if n_error else "")
    )
    if args.fix:
        removed = sum(r.removed_outputs for r in reports)
        print(f"--fix removed {removed} output(s); {n_left} failure(s) need a manual fix.")
    if n_error:
        return 2
    return 1 if n_left else 0


if __name__ == "__main__":
    raise SystemExit(main())
