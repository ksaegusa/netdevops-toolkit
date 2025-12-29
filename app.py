from __future__ import annotations

import csv
import io
import json
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
    return jinja.TemplateResponse(request, "regex.html", {})


@app.get("/diff", response_class=HTMLResponse)
def diff_page(request: Request) -> HTMLResponse:
    return jinja.TemplateResponse(request, "diff.html", {})


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

    text_payload = raw_text
    if not raw_text.strip() and text_upload and text_upload.filename:
        try:
            text_bytes = text_upload.file.read()
            text_payload = text_bytes.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - surface upload error
            error = f"テキストの読み込みに失敗しました: {exc}"

    if not error:
        eval_query = normalize_jmespath_for_eval(jmespath_query)
        if not template_name and not template_from_upload and not template_body.strip():
            error = "テンプレートを選択するか、アップロード、または直接入力してください。"
        elif len(text_payload.encode("utf-8", errors="replace")) > MAX_INPUT_BYTES:
            error = "入力サイズが上限を超えています。"
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
