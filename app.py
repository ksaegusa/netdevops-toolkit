from __future__ import annotations

import csv
import difflib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jmespath
import yaml
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from textfsm import TextFSM

APP_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = APP_ROOT / "templates"
TEXTFSM_DIR = APP_ROOT / "textfsm_templates"
MAX_INPUT_BYTES = 2 * 1024 * 1024

app = FastAPI()
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")

jinja = Jinja2Templates(directory=str(TEMPLATE_DIR))


def list_textfsm_templates() -> list[str]:
    if not TEXTFSM_DIR.exists():
        return []
    return sorted([path.name for path in TEXTFSM_DIR.glob("*.textfsm")])


def parse_textfsm(raw_text: str, template_name: str) -> list[dict[str, Any]]:
    template_path = TEXTFSM_DIR / template_name
    with template_path.open("r", encoding="utf-8") as handle:
        fsm = TextFSM(handle)
    parsed = fsm.ParseText(raw_text)
    return [dict(zip(fsm.header, row)) for row in parsed]


def parse_textfsm_with_template(raw_text: str, template_body: str) -> list[dict[str, Any]]:
    with io.StringIO(template_body) as handle:
        fsm = TextFSM(handle)
    parsed = fsm.ParseText(raw_text)
    return [dict(zip(fsm.header, row)) for row in parsed]


@dataclass
class ConfigNode:
    line: str
    indent: int
    line_no: int
    children: list["ConfigNode"] = field(default_factory=list)


def parse_config(raw_config: str, ignore_patterns: list[re.Pattern[str]]) -> list[ConfigNode]:
    root = ConfigNode(line="ROOT", indent=-1, line_no=0)
    path: list[ConfigNode] = [root]
    for line_no, raw_line in enumerate(raw_config.splitlines(), start=1):
        trimmed = raw_line.strip()
        if not trimmed or trimmed.startswith("!"):
            continue
        if any(pattern.search(trimmed) for pattern in ignore_patterns):
            continue

        indent = 0
        for ch in raw_line:
            if ch == " ":
                indent += 1
            else:
                break

        node = ConfigNode(line=trimmed, indent=indent, line_no=line_no)
        while len(path) > 1 and path[-1].indent >= indent:
            path.pop()
        parent = path[-1]
        parent.children.append(node)
        path.append(node)
    return root.children


def filter_nodes(nodes: list[ConfigNode], target: str) -> list[ConfigNode]:
    if not target:
        return nodes
    return [node for node in nodes if node.line.startswith(target)]


def to_node_map(nodes: list[ConfigNode]) -> dict[str, ConfigNode]:
    return {node.line: node for node in nodes}


def has_node_diff(
    running: list[ConfigNode],
    candidate: list[ConfigNode],
    order_sensitive: bool,
) -> bool:
    if len(running) != len(candidate):
        return True
    if order_sensitive:
        for run_node, cand_node in zip(running, candidate):
            if run_node.line != cand_node.line:
                return True
            if has_node_diff(run_node.children, cand_node.children, order_sensitive):
                return True
        return False
    running_map = to_node_map(running)
    candidate_map = to_node_map(candidate)
    for line in running_map:
        if line not in candidate_map:
            return True
    for cand_node in candidate:
        run_node = running_map[cand_node.line]
        if has_node_diff(run_node.children, cand_node.children, order_sensitive):
            return True
    return False


@dataclass
class DiffLine:
    sign: str
    line: str
    depth: int
    kind: str
    line_no: int | None = None


def add_subtree(lines: list[DiffLine], node: ConfigNode, sign: str, depth: int, kind: str) -> None:
    lines.append(
        DiffLine(sign=sign, line=node.line, depth=depth, kind=kind, line_no=node.line_no)
    )
    for child in node.children:
        add_subtree(lines, child, sign, depth + 1, kind)


def diff_config(
    running: list[ConfigNode],
    candidate: list[ConfigNode],
    order_sensitive: bool,
) -> list[DiffLine]:
    if order_sensitive:
        return diff_config_ordered(running, candidate, depth=0)
    return diff_config_unordered(running, candidate, depth=0)


def diff_config_unordered(
    running: list[ConfigNode],
    candidate: list[ConfigNode],
    depth: int,
) -> list[DiffLine]:
    lines: list[DiffLine] = []
    running_map = to_node_map(running)
    candidate_map = to_node_map(candidate)

    for run_node in running:
        if run_node.line not in candidate_map:
            add_subtree(lines, run_node, "-", depth=depth, kind="remove")

    for cand_node in candidate:
        run_node = running_map.get(cand_node.line)
        if run_node is None:
            add_subtree(lines, cand_node, "+", depth=depth, kind="add")
        else:
            if has_node_diff(run_node.children, cand_node.children, order_sensitive=False):
                lines.append(
                    DiffLine(
                        sign=" ",
                        line=cand_node.line,
                        depth=depth,
                        kind="context",
                        line_no=cand_node.line_no,
                    )
                )
                lines.extend(diff_config_unordered(run_node.children, cand_node.children, depth + 1))

    return lines


def diff_config_ordered(
    running: list[ConfigNode],
    candidate: list[ConfigNode],
    depth: int,
) -> list[DiffLine]:
    lines: list[DiffLine] = []
    old_lines = [node.line for node in running]
    new_lines = [node.line for node in candidate]
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for old_index, new_index in zip(range(i1, i2), range(j1, j2)):
                run_node = running[old_index]
                cand_node = candidate[new_index]
                if has_node_diff(run_node.children, cand_node.children, order_sensitive=True):
                    lines.append(
                        DiffLine(
                            sign=" ",
                            line=cand_node.line,
                            depth=depth,
                            kind="context",
                            line_no=cand_node.line_no,
                        )
                    )
                    lines.extend(
                        diff_config_ordered(run_node.children, cand_node.children, depth + 1)
                    )
        elif tag in ("delete", "replace"):
            for run_node in running[i1:i2]:
                add_subtree(lines, run_node, "-", depth=depth, kind="remove")
        if tag in ("insert", "replace"):
            for cand_node in candidate[j1:j2]:
                add_subtree(lines, cand_node, "+", depth=depth, kind="add")

    return lines


def read_text_payload(raw_text: str, text_upload: UploadFile | None) -> tuple[str, str]:
    text_payload = raw_text
    if not raw_text.strip() and text_upload and text_upload.filename:
        try:
            text_bytes = text_upload.file.read()
            text_payload = text_bytes.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - surface upload error
            return "", f"テキストの読み込みに失敗しました: {exc}"
    if len(text_payload.encode("utf-8", errors="replace")) > MAX_INPUT_BYTES:
        return "", "入力サイズが上限を超えています。"
    return text_payload, ""


def parse_regex_flags(flag_names: list[str]) -> int:
    flags = 0
    mapping = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
    }
    for name in flag_names:
        if name in mapping:
            flags |= mapping[name]
    return flags


def normalize_jmespath_query(query: str) -> str:
    return (
        query.replace("“", "\"")
        .replace("”", "\"")
        .replace("＂", "\"")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("＇", "'")
    )


def normalize_jmespath_for_eval(query: str) -> str:
    normalized = normalize_jmespath_query(query)
    output: list[str] = []
    in_backtick = False
    in_double = False
    buffer: list[str] = []

    for ch in normalized:
        if in_double:
            if ch == "\"":
                escaped = "".join(buffer).replace("'", "\\'")
                output.append("'" + escaped + "'")
                buffer = []
                in_double = False
            else:
                buffer.append(ch)
            continue

        if ch == "`":
            in_backtick = not in_backtick
            output.append(ch)
            continue

        if not in_backtick and ch == "\"":
            in_double = True
            continue

        output.append(ch)

    if in_double:
        # Fallback: keep unterminated input as-is
        output.append("\"" + "".join(buffer))

    return "".join(output)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return jinja.TemplateResponse(
        request,
        "index.html",
        {
            "templates": list_textfsm_templates(),
        },
    )


@app.get("/regex", response_class=HTMLResponse)
def regex_page(request: Request) -> HTMLResponse:
    return jinja.TemplateResponse(
        request,
        "regex.html",
        {
            "error": "",
            "matches": [],
            "match_count": 0,
        },
    )


@app.post("/regex/text-preview", response_class=HTMLResponse)
def regex_text_preview(
    request: Request,
    text_upload: UploadFile | None = None,
) -> HTMLResponse:
    text_payload, error = read_text_payload("", text_upload)
    return jinja.TemplateResponse(
        request,
        "partials/regex_textarea.html",
        {
            "text_content": text_payload,
            "text_error": error,
        },
    )


@app.get("/diff", response_class=HTMLResponse)
def diff_page(request: Request) -> HTMLResponse:
    return jinja.TemplateResponse(request, "diff.html", {})


@app.post("/diff/old-preview", response_class=HTMLResponse)
def diff_old_preview(
    request: Request,
    old_upload: UploadFile | None = None,
) -> HTMLResponse:
    text_payload, error = read_text_payload("", old_upload)
    return jinja.TemplateResponse(
        request,
        "partials/diff_textarea_old.html",
        {
            "old_text": text_payload,
            "old_error": error,
        },
    )


@app.post("/diff/new-preview", response_class=HTMLResponse)
def diff_new_preview(
    request: Request,
    new_upload: UploadFile | None = None,
) -> HTMLResponse:
    text_payload, error = read_text_payload("", new_upload)
    return jinja.TemplateResponse(
        request,
        "partials/diff_textarea_new.html",
        {
            "new_text": text_payload,
            "new_error": error,
        },
    )


@app.post("/diff", response_class=HTMLResponse)
def diff_run(
    request: Request,
    old_text: str = Form(""),
    new_text: str = Form(""),
    ignore_patterns: str = Form(""),
    target: str = Form(""),
    order_sensitive: str = Form(""),
    old_upload: UploadFile | None = None,
    new_upload: UploadFile | None = None,
) -> HTMLResponse:
    error = ""
    compiled_patterns: list[re.Pattern[str]] = []
    if ignore_patterns.strip():
        for line in ignore_patterns.splitlines():
            pattern = line.strip()
            if not pattern:
                continue
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error as exc:
                error = f"正規表現の解析に失敗しました: {exc}"
                break

    diff_lines: list[DiffLine] = []
    if not error:
        old_payload, error = read_text_payload(old_text, old_upload)
    if not error:
        new_payload, error = read_text_payload(new_text, new_upload)

    if not error:
        if not old_payload.strip() or not new_payload.strip():
            error = "旧設定と新設定の両方を入力してください。"
        else:
            running_nodes = parse_config(old_payload, compiled_patterns)
            candidate_nodes = parse_config(new_payload, compiled_patterns)
            running_nodes = filter_nodes(running_nodes, target.strip())
            candidate_nodes = filter_nodes(candidate_nodes, target.strip())
            diff_lines = diff_config(
                running_nodes,
                candidate_nodes,
                order_sensitive=order_sensitive == "on",
            )

    return jinja.TemplateResponse(
        request,
        "partials/diff_result.html",
        {
            "error": error,
            "diff_lines": diff_lines,
            "order_sensitive": order_sensitive == "on",
        },
    )


@app.post("/regex", response_class=HTMLResponse)
def regex_run(
    request: Request,
    pattern: str = Form(""),
    raw_text: str = Form(""),
    text_upload: UploadFile | None = None,
    flags: list[str] = Form([]),
) -> HTMLResponse:
    error = ""
    matches: list[str] = []

    text_payload, error = read_text_payload(raw_text, text_upload)
    if not error and not pattern.strip():
        error = "正規表現パターンを入力してください。"

    if not error:
        try:
            compiled = re.compile(pattern, parse_regex_flags(flags))
            for line in text_payload.splitlines():
                if compiled.search(line):
                    matches.append(line)
        except re.error as exc:
            error = f"正規表現の解析に失敗しました: {exc}"

    return jinja.TemplateResponse(
        request,
        "partials/regex_result.html",
        {
            "error": error,
            "matches": matches,
            "match_count": len(matches),
        },
    )


@app.post("/parse", response_class=HTMLResponse)
def parse(
    request: Request,
    raw_text: str = Form(""),
    template_name: str = Form(""),
    template_body: str = Form(""),
    jmespath_query: str = Form(""),
    template_upload: UploadFile | None = None,
    text_upload: UploadFile | None = None,
) -> HTMLResponse:
    error = ""
    parsed: list[dict[str, Any]] | None = None
    filtered: Any | None = None
    headers: list[str] = []

    template_from_upload = None
    if template_upload and template_upload.filename:
        if not template_upload.filename.endswith(".textfsm"):
            error = "テンプレートは .textfsm 拡張子のみ対応しています。"
        else:
            template_from_upload = template_upload

    text_payload, error = read_text_payload(raw_text, text_upload)

    if not error:
        eval_query = normalize_jmespath_for_eval(jmespath_query)
        if not template_name and not template_from_upload and not template_body.strip():
            error = "テンプレートを選択するか、アップロード、または直接入力してください。"
        else:
            try:
                if template_body.strip():
                    parsed = parse_textfsm_with_template(text_payload, template_body)
                elif template_from_upload is not None:
                    template_bytes = template_from_upload.file.read()
                    template_body = template_bytes.decode("utf-8", errors="replace")
                    parsed = parse_textfsm_with_template(text_payload, template_body)
                else:
                    parsed = parse_textfsm(text_payload, template_name)
                if eval_query.strip():
                    filtered = jmespath.search(eval_query, parsed)
            except Exception as exc:  # pragma: no cover - surface parsing error
                error = f"パースに失敗しました: {exc}"

    if parsed:
        headers = list(parsed[0].keys())

    return jinja.TemplateResponse(
        request,
        "partials/result.html",
        {
            "error": error,
            "parsed_json": json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None else "",
            "parsed_headers": headers,
            "parsed_rows": parsed or [],
            "jmespath_query": normalize_jmespath_query(jmespath_query),
            "filtered_json": json.dumps(filtered, ensure_ascii=False, indent=2)
            if filtered is not None
            else "",
        },
    )


@app.post("/filter", response_class=HTMLResponse)
def filter_results(
    request: Request,
    parsed_json: str = Form(""),
    jmespath_query: str = Form(""),
) -> HTMLResponse:
    error = ""
    parsed: list[dict[str, Any]] | None = None
    filtered: Any | None = None
    headers: list[str] = []

    if not parsed_json.strip():
        error = "パース結果がありません。"
    else:
        try:
            eval_query = normalize_jmespath_for_eval(jmespath_query)
            parsed = json.loads(parsed_json)
            if not isinstance(parsed, list):
                raise ValueError("JSON形式が不正です。")
            if eval_query.strip():
                filtered = jmespath.search(eval_query, parsed)
        except Exception as exc:  # pragma: no cover - surface filter error
            error = f"JMESPathの適用に失敗しました: {exc}"

    if parsed:
        headers = list(parsed[0].keys())

    return jinja.TemplateResponse(
        request,
        "partials/result.html",
        {
            "error": error,
            "parsed_json": json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None else "",
            "parsed_headers": headers,
            "parsed_rows": parsed or [],
            "jmespath_query": normalize_jmespath_query(jmespath_query),
            "filtered_json": json.dumps(filtered, ensure_ascii=False, indent=2)
            if filtered is not None
            else "",
        },
    )


@app.get("/template-editor", response_class=HTMLResponse)
def template_editor(request: Request, template_name: str = "") -> HTMLResponse:
    template_text = ""
    if template_name in list_textfsm_templates():
        template_path = TEXTFSM_DIR / template_name
        template_text = template_path.read_text(encoding="utf-8", errors="replace")
    return jinja.TemplateResponse(
        request,
        "partials/template_editor.html",
        {
            "template_text": template_text,
        },
    )


@app.post("/download/csv")
def download_csv(parsed_json: str = Form("")) -> Response:
    if not parsed_json.strip():
        return Response("No data", status_code=400)
    data = json.loads(parsed_json)
    if not isinstance(data, list):
        return Response("Invalid data", status_code=400)

    output = io.StringIO()
    headers = list(data[0].keys()) if data else []
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    writer.writerows(data)
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=parsed.csv"},
    )


@app.post("/download/yaml")
def download_yaml(parsed_json: str = Form("")) -> Response:
    if not parsed_json.strip():
        return Response("No data", status_code=400)
    data = json.loads(parsed_json)
    yaml_text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    return Response(
        yaml_text,
        media_type="application/x-yaml",
        headers={"Content-Disposition": "attachment; filename=parsed.yaml"},
    )


@app.post("/download/regex")
def download_regex(matches_text: str = Form("")) -> Response:
    if not matches_text.strip():
        return Response("No data", status_code=400)
    return Response(
        matches_text,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=matches.txt"},
    )
