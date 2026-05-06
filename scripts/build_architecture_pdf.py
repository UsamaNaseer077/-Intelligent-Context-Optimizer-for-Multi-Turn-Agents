from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "ARCHITECTURE_AND_EVALUATION.md"
BUILD = ROOT / "build" / "architecture_pdf"
HTML_OUT = BUILD / "ARCHITECTURE_AND_EVALUATION.html"
PDF_OUT = ROOT / "docs" / "ARCHITECTURE_AND_EVALUATION.pdf"
PUPPETEER_CONFIG = BUILD / "puppeteer-config.json"


def inline_md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def render_mermaid_blocks(markdown: str) -> str:
    BUILD.mkdir(parents=True, exist_ok=True)
    for stale in BUILD.glob("diagram_*.*"):
        stale.unlink()
    PUPPETEER_CONFIG.write_text(
        '{"args":["--no-sandbox","--disable-setuid-sandbox","--disable-dev-shm-usage"]}',
        encoding="utf-8",
    )
    pattern = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)

    def replace(match: re.Match[str]) -> str:
        idx = len(list(BUILD.glob("diagram_*.mmd"))) + 1
        mmd = BUILD / f"diagram_{idx:02d}.mmd"
        image = BUILD / f"diagram_{idx:02d}.png"
        mmd.write_text(match.group(1), encoding="utf-8")
        subprocess.run(
            [
                "mmdc",
                "-i",
                str(mmd),
                "-o",
                str(image),
                "-b",
                "white",
                "--puppeteerConfigFile",
                str(PUPPETEER_CONFIG),
            ],
            check=True,
        )
        return f"\n@@DIAGRAM:{image.resolve()}@@\n"

    return pattern.sub(replace, markdown)


def table_to_html(rows: list[str]) -> str:
    parsed = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        parsed.append(cells)
    if len(parsed) >= 2 and all(set(cell) <= {"-", ":"} for cell in parsed[1]):
        header = parsed[0]
        body = parsed[2:]
    else:
        header = []
        body = parsed
    parts = ["<table>"]
    if header:
        parts.append("<thead><tr>")
        parts.extend(f"<th>{inline_md(cell)}</th>" for cell in header)
        parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in body:
        parts.append("<tr>")
        parts.extend(f"<td>{inline_md(cell)}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    code_lang = ""
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]

        if in_code:
            if line.startswith("```"):
                out.append(f"<pre><code class=\"language-{html.escape(code_lang)}\">{html.escape(chr(10).join(code_lines))}</code></pre>")
                in_code = False
                code_lines = []
                code_lang = ""
            else:
                code_lines.append(line)
            i += 1
            continue

        if line.startswith("```"):
            in_code = True
            code_lang = line[3:].strip()
            i += 1
            continue

        if line.startswith("@@DIAGRAM:"):
            path = line.removeprefix("@@DIAGRAM:").removesuffix("@@")
            out.append(f'<figure class="diagram"><img src="{html.escape(path)}" alt="Architecture diagram"/></figure>')
            i += 1
            continue

        if line.startswith("|") and "|" in line[1:]:
            table_rows = []
            while i < len(lines) and lines[i].startswith("|") and "|" in lines[i][1:]:
                table_rows.append(lines[i])
                i += 1
            out.append(table_to_html(table_rows))
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline_md(heading.group(2))}</h{level}>")
            i += 1
            continue

        if line.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{inline_md(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue

        if not line.strip():
            out.append("")
        else:
            out.append(f"<p>{inline_md(line)}</p>")
        i += 1

    return "\n".join(out)


def wrap_html(body: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MemoryOS Architecture and Evaluation</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17202a;
      line-height: 1.45;
      font-size: 13px;
      margin: 32px;
    }}
    h1 {{ font-size: 30px; margin: 0 0 18px; }}
    h2 {{ font-size: 22px; margin-top: 28px; border-bottom: 1px solid #d8dee4; padding-bottom: 6px; }}
    h3 {{ font-size: 17px; margin-top: 20px; }}
    code {{ background: #f3f4f6; color: #17202a; border-radius: 4px; padding: 1px 4px; }}
    pre {{ background: #f6f8fa; color: #17202a; border: 1px solid #d0d7de; border-radius: 6px; padding: 12px; overflow-wrap: break-word; white-space: pre-wrap; }}
    pre code {{ background: transparent; color: #17202a; padding: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 18px; page-break-inside: avoid; }}
    th, td {{ border: 1px solid #d0d7de; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #f6f8fa; font-weight: 700; }}
    figure.diagram {{ margin: 16px 0 22px; text-align: center; page-break-inside: avoid; }}
    figure.diagram img {{ max-width: 100%; height: auto; }}
    ul {{ margin-top: 6px; }}
    @page {{ size: A4; margin: 18mm 14mm; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    markdown = SRC.read_text(encoding="utf-8")
    with_diagrams = render_mermaid_blocks(markdown)
    body = markdown_to_html(with_diagrams)
    HTML_OUT.write_text(wrap_html(body), encoding="utf-8")
    subprocess.run(
        [
            "wkhtmltopdf",
            "--enable-local-file-access",
            "--print-media-type",
            str(HTML_OUT),
            str(PDF_OUT),
        ],
        check=True,
    )
    print(PDF_OUT)


if __name__ == "__main__":
    main()
