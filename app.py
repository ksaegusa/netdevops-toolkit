from __future__ import annotations

import csv
import difflib
import io
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jmespath
import yaml
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from textfsm import TextFSM

try:
    from pyang import context, repository
except ImportError:  # pragma: no cover - optional dependency
    context = None
    repository = None

APP_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = APP_ROOT / "templates"
TEXTFSM_DIR = APP_ROOT / "textfsm_templates"
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_INPUT_LINES = 20000
MAX_TEMPLATE_BYTES = 256 * 1024

app = FastAPI()
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")

jinja = Jinja2Templates(directory=str(TEMPLATE_DIR))

DEFAULT_YANG_SAMPLE = "\n".join(
    [
        "module sample-interfaces {",
        "  yang-version 1.1;",
        "  namespace \"http://example.com/sample-interfaces\";",
        "  prefix samp;",
        "",
        "  container interfaces {",
        "    list interface {",
        "      key \"name\";",
        "      leaf name { type string; }",
        "      leaf description { type string; }",
        "      leaf mtu { type uint16; }",
        "      leaf enabled { type boolean; }",
        "    }",
        "  }",
        "}",
    ]
)


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
    error = validate_text_blob(text_payload, MAX_INPUT_BYTES)
    if error:
        return "", error
    return text_payload, ""


def validate_text_blob(text: str, max_bytes: int) -> str:
    if len(text.encode("utf-8", errors="replace")) > max_bytes:
        return "入力サイズが上限を超えています。"
    if text.count("\n") + 1 > MAX_INPUT_LINES:
        return "入力行数が上限を超えています。"
    return ""


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
            "pattern": "",
            "flags": [],
            "text_content": "",
            "text_error": "",
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
    return jinja.TemplateResponse(
        request,
        "diff.html",
        {
            "ignore_patterns": "",
            "target": "",
            "order_sensitive": False,
            "old_text": "",
            "new_text": "",
        },
    )


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


@app.get("/telemetry", response_class=HTMLResponse)
def telemetry_page(request: Request) -> HTMLResponse:
    return jinja.TemplateResponse(
        request,
        "telemetry.html",
        {},
    )


@app.get("/transformer", response_class=HTMLResponse)
def transformer_redirect() -> RedirectResponse:
    return RedirectResponse(url="/model")


@app.get("/model", response_class=HTMLResponse)
def model_page(request: Request) -> HTMLResponse:
    return jinja.TemplateResponse(
        request,
        "transformer.html",
        {
            "default_yang": DEFAULT_YANG_SAMPLE,
        },
    )


def parse_yang_model(yang_text: str) -> tuple[list[str], str, str]:
    error = validate_text_blob(yang_text, MAX_TEMPLATE_BYTES)
    if error:
        return [], "", error

    def fallback_parse(text: str, include_groupings: bool) -> tuple[list[str], str]:
        paths: list[str] = []
        sample: dict[str, Any] = {}
        stack: list[dict[str, Any]] = []
        skip_depth = 0

        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("grouping ") and not include_groupings:
                skip_depth += line.count("{") or 1
                continue
            if skip_depth:
                skip_depth += line.count("{")
                skip_depth -= line.count("}")
                continue

            match = re.match(
                r"^(grouping|container|list|leaf|leaf-list)\\s+([\\w\\-\\.]+)",
                line,
            )
            if not match:
                if "}" in line and stack:
                    stack.pop()
                continue

            keyword, name = match.group(1), match.group(2)
            if keyword == "grouping":
                segment = f"grouping/{name}"
                key = name
                path = "/" + "/".join([item["segment"] for item in stack] + [segment])
                paths.append(path)
                node: dict[str, Any] = {}
                if stack:
                    parent = stack[-1]["node"]
                    parent[key] = node
                else:
                    sample[key] = node
                stack.append({"segment": segment, "key": key, "node": node, "keyword": keyword})
            elif keyword in ("container", "list"):
                segment = name
                key = name
                path = "/" + "/".join([item["segment"] for item in stack] + [segment])
                paths.append(path)
                node = {}
                if stack:
                    parent = stack[-1]["node"]
                    if keyword == "list":
                        parent.setdefault(key, []).append(node)
                    else:
                        parent[key] = node
                else:
                    if keyword == "list":
                        sample.setdefault(key, []).append(node)
                    else:
                        sample[key] = node
                stack.append({"segment": segment, "key": key, "node": node, "keyword": keyword})
            else:
                path = "/" + "/".join([item["segment"] for item in stack] + [name])
                paths.append(path)
                value = "<value>"
                if stack:
                    parent = stack[-1]["node"]
                    if keyword == "leaf":
                        parent[name] = value
                    else:
                        parent[name] = [value]
                else:
                    if keyword == "leaf":
                        sample[name] = value
                    else:
                        sample[name] = [value]

            if "}" in line and stack:
                stack.pop()

        return paths, json.dumps(sample, ensure_ascii=False, indent=2)

    is_submodule = yang_text.lstrip().startswith("submodule ")
    if context is None or repository is None:
        paths, sample_json = fallback_parse(yang_text, is_submodule)
        return paths, sample_json, ""

    try:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.yang"
            path.write_text(yang_text, encoding="utf-8")
            repo = repository.FileRepository(tmpdir)
            ctx = context.Context(repo)
            module = ctx.add_module("model.yang", yang_text)
            if module is None:
                paths, sample_json = fallback_parse(yang_text, is_submodule)
                return paths, sample_json, ""
            ctx.validate()

            def list_segment(stmt: Any) -> str:
                key_stmt = stmt.search_one("key")
                if key_stmt and key_stmt.arg:
                    selectors = "".join([f"[{k}=*]" for k in key_stmt.arg.split()])
                    return f"{stmt.arg}{selectors}"
                return stmt.arg

            def walk(stmt: Any, prefix: str, paths: list[str], sample: dict[str, Any]) -> None:
                for child in getattr(stmt, "i_children", []) or []:
                    if child.keyword not in ("container", "list", "leaf", "leaf-list"):
                        continue
                    segment = list_segment(child) if child.keyword == "list" else child.arg
                    path = f"{prefix}/{segment}" if prefix else f"/{segment}"
                    if child.keyword == "container":
                        paths.append(path)
                        sample[child.arg] = {}
                        walk(child, path, paths, sample[child.arg])
                    elif child.keyword == "list":
                        paths.append(path)
                        entry: dict[str, Any] = {}
                        key_stmt = child.search_one("key")
                        keys = key_stmt.arg.split() if key_stmt and key_stmt.arg else []
                        for key in keys:
                            entry[key] = f"<{key}>"
                        walk(child, path, paths, entry)
                        sample[child.arg] = [entry]
                    else:
                        paths.append(path)
                        type_stmt = child.search_one("type")
                        type_name = type_stmt.arg if type_stmt is not None else "string"
                        if child.keyword == "leaf":
                            sample[child.arg] = f"<{type_name}>"
                        else:
                            sample[child.arg] = [f"<{type_name}>"]

            paths: list[str] = []
            sample: dict[str, Any] = {}
            walk(module, "", paths, sample)
            if module.keyword == "submodule":
                for grouping in module.search("grouping"):
                    group_path = f"/grouping/{grouping.arg}"
                    paths.append(group_path)
                    group_sample: dict[str, Any] = {}
                    sample[grouping.arg] = group_sample
                    walk(grouping, group_path, paths, group_sample)
            sample_json = json.dumps(sample, ensure_ascii=False, indent=2)
            if not paths:
                paths, sample_json = fallback_parse(yang_text, is_submodule)
            return paths, sample_json, ""
    except Exception:  # pragma: no cover - surface parse error
        paths, sample_json = fallback_parse(yang_text, is_submodule)
        return paths, sample_json, ""


def extract_yang_module_name(yang_text: str) -> tuple[str, str]:
    match = re.search(r"^(module|submodule)\s+([\w\-\.]+)", yang_text, re.MULTILINE)
    if not match:
        return "module", "unknown"
    return match.group(1), match.group(2)


def build_model_tree(model_paths: list[str], yang_text: str) -> str:
    def render_pyang_tree() -> str:
        if shutil.which("pyang") is None:
            return ""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.yang"
            path.write_text(yang_text, encoding="utf-8")
            result = subprocess.run(
                ["pyang", "-f", "tree", "--tree-line-length", "120", "model.yang"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        return ""

    pyang_tree = render_pyang_tree()
    if pyang_tree:
        return pyang_tree
    if not model_paths:
        return ""

    def format_segment(segment: str) -> str:
        if "[" not in segment:
            return segment
        base = segment.split("[", 1)[0]
        keys = re.findall(r"\[([^\]=]+)=\*\]", segment)
        label = f"{base}*"
        if keys:
            label += f" [{', '.join(keys)}]"
        return label

    def insert_path(root: dict[str, Any], segments: list[str]) -> None:
        node = root
        for seg in segments:
            node = node.setdefault(seg, {})

    def render_tree(node: dict[str, Any], prefix: str = "") -> list[str]:
        lines: list[str] = []
        items = sorted(node.items(), key=lambda item: item[0])
        last_index = len(items) - 1
        for idx, (name, children) in enumerate(items):
            lines.append(f"{prefix}+-- {format_segment(name)}")
            if children:
                branch = "|   " if idx < last_index else "    "
                lines.extend(render_tree(children, prefix + branch))
        return lines

    root: dict[str, Any] = {}
    for path in model_paths:
        segments = [seg for seg in path.split("/") if seg]
        if len(segments) >= 2 and segments[0] == "grouping":
            segments = [f"grouping {segments[1]}"] + segments[2:]
        if segments:
            insert_path(root, segments)

    kind, name = extract_yang_module_name(yang_text)
    header = f"{kind}: {name}"
    tree_lines = render_tree(root)
    return "\n".join([header] + tree_lines)


def build_model_view(text_content: str) -> tuple[str, list[str], str, str, str]:
    error = ""
    model_paths: list[str] = []
    sample_json = ""
    tree_text = ""
    if not text_content.strip():
        return text_content, model_paths, sample_json, tree_text, error
    error = validate_text_blob(text_content, MAX_INPUT_BYTES)
    if error:
        return text_content, model_paths, sample_json, tree_text, error
    stripped = text_content.lstrip()
    if stripped.startswith("module ") or stripped.startswith("submodule "):
        model_paths, sample_json, error = parse_yang_model(text_content)
        tree_text = build_model_tree(model_paths, text_content)
        return text_content, model_paths, sample_json, tree_text, error
    try:
        data = json.loads(text_content)
        rows = flatten_json(data)
        model_paths = [row["path"] for row in rows]
        sample_json = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as exc:  # pragma: no cover - surface parse error
        error = f"JSONの解析に失敗しました: {exc}"
    return text_content, model_paths, sample_json, tree_text, error


@app.post("/model/upload-preview", response_class=HTMLResponse)
def model_upload_preview(
    request: Request,
    model_upload: UploadFile | None = None,
) -> HTMLResponse:
    text_payload, error = read_text_payload("", model_upload)
    text_content, model_paths, sample_json, tree_text, preview_error = build_model_view(
        text_payload
    )
    error = error or preview_error
    return jinja.TemplateResponse(
        request,
        "partials/model_preview.html",
        {
            "text_content": text_content,
            "text_error": error,
            "model_paths": model_paths,
            "sample_json": sample_json,
            "tree_text": tree_text,
            "default_yang": DEFAULT_YANG_SAMPLE,
        },
    )


@app.post("/model/inspect", response_class=HTMLResponse)
def model_inspect(
    request: Request,
    model_text: str = Form(""),
) -> HTMLResponse:
    _, model_paths, sample_json, tree_text, error = build_model_view(model_text)
    return jinja.TemplateResponse(
        request,
        "partials/model_result.html",
        {
            "model_paths": model_paths,
            "sample_json": sample_json,
            "tree_text": tree_text,
            "text_error": error,
        },
    )


def build_telemetry_preview(text_content: str) -> tuple[str, Any | None, str]:
    error = ""
    data: Any | None = None
    if text_content.strip():
        error = validate_text_blob(text_content, MAX_INPUT_BYTES)
        if not error:
            try:
                data = json.loads(text_content)
            except Exception:
                try:
                    data = yaml.safe_load(text_content)
                except Exception as exc:  # pragma: no cover - surface parse error
                    error = f"JSON/YAMLの解析に失敗しました: {exc}"
    return text_content, data, error


def stringify_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def flatten_json(value: Any, prefix: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_json(item, path))
        return rows
    if isinstance(value, list):
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            rows.extend(flatten_json(item, path))
        return rows
    rows.append({"path": prefix or ".", "value": stringify_value(value)})
    return rows


@app.post("/telemetry/upload-preview", response_class=HTMLResponse)
def telemetry_upload_preview(
    request: Request,
    telemetry_upload: UploadFile | None = None,
) -> HTMLResponse:
    text_payload, error = read_text_payload("", telemetry_upload)
    text_content, data, preview_error = build_telemetry_preview(text_payload)
    error = error or preview_error
    flat_rows = flatten_json(data) if data is not None else []
    return jinja.TemplateResponse(
        request,
        "partials/telemetry_preview.html",
        {
            "text_content": text_content,
            "text_error": error,
            "telemetry_data": data,
            "flat_rows": flat_rows,
            "filtered_json": "",
            "jmespath_query": "",
            "oob": True,
            "textarea_oob": False,
        },
    )


@app.post("/telemetry", response_class=HTMLResponse)
def telemetry_run(
    request: Request,
    telemetry_text: str = Form(""),
    jmespath_query: str = Form(""),
) -> HTMLResponse:
    error = validate_text_blob(telemetry_text, MAX_INPUT_BYTES)
    data: Any = None
    filtered: Any | None = None
    if not error:
        try:
            data = json.loads(telemetry_text)
            if jmespath_query.strip():
                filtered = jmespath.search(normalize_jmespath_for_eval(jmespath_query), data)
        except Exception:
            try:
                data = yaml.safe_load(telemetry_text)
                if jmespath_query.strip() and data is not None:
                    filtered = jmespath.search(normalize_jmespath_for_eval(jmespath_query), data)
            except Exception as exc:  # pragma: no cover - surface parse error
                error = f"JSON/YAMLの解析に失敗しました: {exc}"
    flat_rows = flatten_json(data) if data is not None else []

    return jinja.TemplateResponse(
        request,
        "partials/telemetry_result.html",
        {
            "error": error,
            "telemetry_data": data,
            "flat_rows": flat_rows,
            "filtered_json": json.dumps(filtered, ensure_ascii=False, indent=2)
            if filtered is not None
            else "",
            "jmespath_query": normalize_jmespath_query(jmespath_query),
            "oob": False,
            "textarea_oob": False,
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
    old_payload = old_text
    new_payload = new_text
    if not error:
        old_payload, error = read_text_payload(old_text, old_upload)
    if not error:
        new_payload, error = read_text_payload(new_text, new_upload)

    if not error:
        if not old_payload.strip() or not new_payload.strip():
            error = "比較元と比較先の両方を入力してください。"
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

    if request.headers.get("hx-request") == "true":
        return jinja.TemplateResponse(
            request,
            "partials/diff_result.html",
            {
                "error": error,
                "diff_lines": diff_lines,
                "order_sensitive": order_sensitive == "on",
            },
        )

    return jinja.TemplateResponse(
        request,
        "diff.html",
        {
            "error": error,
            "diff_lines": diff_lines,
            "order_sensitive": order_sensitive == "on",
            "ignore_patterns": ignore_patterns,
            "target": target,
            "old_text": old_payload,
            "new_text": new_payload,
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

    if request.headers.get("hx-request") == "true":
        return jinja.TemplateResponse(
            request,
            "partials/regex_result.html",
            {
                "error": error,
                "matches": matches,
                "match_count": len(matches),
            },
        )

    return jinja.TemplateResponse(
        request,
        "regex.html",
        {
            "error": error,
            "matches": matches,
            "match_count": len(matches),
            "pattern": pattern,
            "flags": flags,
            "text_content": text_payload,
            "text_error": "",
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

    if request.headers.get("hx-request") == "true":
        return jinja.TemplateResponse(
            request,
            "partials/result.html",
            {
                "error": error,
                "parsed_json": json.dumps(parsed, ensure_ascii=False, indent=2)
                if parsed is not None
                else "",
                "parsed_headers": headers,
                "parsed_rows": parsed or [],
                "jmespath_query": normalize_jmespath_query(jmespath_query),
                "filtered_json": json.dumps(filtered, ensure_ascii=False, indent=2)
                if filtered is not None
                else "",
            },
        )

    template_text = template_body
    if not template_text and template_name in list_textfsm_templates():
        template_path = TEXTFSM_DIR / template_name
        template_text = template_path.read_text(encoding="utf-8", errors="replace")

    return jinja.TemplateResponse(
        request,
        "index.html",
        {
            "templates": list_textfsm_templates(),
            "template_text": template_text,
            "raw_text": raw_text,
            "error": error,
            "parsed_json": json.dumps(parsed, ensure_ascii=False, indent=2)
            if parsed is not None
            else "",
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
