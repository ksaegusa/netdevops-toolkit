from fastapi.testclient import TestClient

from types import SimpleNamespace

import app as app_module
from app import app, build_model_view, normalize_jmespath_for_eval, normalize_jmespath_query


client = TestClient(app)


def test_normalize_jmespath_preserves_backticks():
    query = '[?VLAN_ID=="1" && SPEED==`1000`]'
    normalized = normalize_jmespath_for_eval(query)
    assert "VLAN_ID=='1'" in normalized
    assert "SPEED==`1000`" in normalized


def test_normalize_jmespath_query_converts_smart_quotes():
    query = '“foo” == ‘bar’'
    normalized = normalize_jmespath_query(query)
    assert normalized == '"foo" == \'bar\''


def test_filter_endpoint_accepts_double_quotes():
    parsed_json = '[{"VLAN_ID": "1"}, {"VLAN_ID": "2"}]'
    response = client.post(
        "/filter",
        data={"parsed_json": parsed_json, "jmespath_query": '[?VLAN_ID=="1"]'},
    )
    assert response.status_code == 200
    assert '"VLAN_ID": "1"' in response.text


def test_parse_with_template_body():
    template_body = "\n".join(
        [
            "Value WORD (\\S+)",
            "",
            "Start",
            "  ^${WORD} -> Record",
        ]
    )
    response = client.post(
        "/parse",
        data={
            "template_body": template_body,
            "raw_text": "hello\\nworld",
        },
    )
    assert response.status_code == 200
    assert "hello" in response.text
    assert "world" in response.text


def test_parse_requires_template_input():
    response = client.post(
        "/parse",
        data={"raw_text": "hello\nworld"},
        headers={"hx-request": "true"},
    )
    assert response.status_code == 200
    assert "テンプレートを選択するか、アップロード、または直接入力してください。" in response.text


def test_parse_rejects_invalid_template_extension():
    response = client.post(
        "/parse",
        data={"raw_text": "hello"},
        files={"template_upload": ("bad.txt", b"Value WORD (\\S+)", "text/plain")},
        headers={"hx-request": "true"},
    )
    assert response.status_code == 200
    assert "テンプレートは .textfsm 拡張子のみ対応しています。" in response.text


def test_parse_rejects_oversized_text():
    template_body = "\n".join(
        [
            "Value WORD (\\S+)",
            "",
            "Start",
            "  ^${WORD} -> Record",
        ]
    )
    too_large = "a" * (2 * 1024 * 1024 + 1)
    response = client.post(
        "/parse",
        data={
            "template_body": template_body,
            "raw_text": too_large,
        },
        headers={"hx-request": "true"},
    )
    assert response.status_code == 200
    assert "入力サイズが上限を超えています。" in response.text


def test_filter_rejects_invalid_json():
    response = client.post(
        "/filter",
        data={"parsed_json": "not-json", "jmespath_query": "[].foo"},
    )
    assert response.status_code == 200
    assert "JMESPathの適用に失敗しました" in response.text


def test_filter_rejects_non_list_json():
    response = client.post(
        "/filter",
        data={"parsed_json": "{\"foo\": 1}", "jmespath_query": "[].foo"},
    )
    assert response.status_code == 200
    assert "JMESPathの適用に失敗しました" in response.text


def test_regex_endpoint_extracts_lines():
    response = client.post(
        "/regex",
        data={
            "pattern": "ERROR",
            "raw_text": "INFO ok\nERROR failed\nWARN note",
        },
        headers={"hx-request": "true"},
    )
    assert response.status_code == 200
    assert "ERROR failed" in response.text
    assert "WARN note" not in response.text


def test_regex_rejects_empty_pattern():
    response = client.post(
        "/regex",
        data={"pattern": "", "raw_text": "ERROR failed"},
        headers={"hx-request": "true"},
    )
    assert response.status_code == 200
    assert "正規表現パターンを入力してください。" in response.text


def test_regex_text_preview_upload_shows_content():
    response = client.post(
        "/regex/text-preview",
        files={"text_upload": ("log.txt", b"alpha\nbeta", "text/plain")},
    )
    assert response.status_code == 200
    assert "alpha" in response.text
    assert "beta" in response.text


def test_diff_endpoint_ignores_order():
    old_cfg = "\n".join(
        [
            "router ospf 1",
            " network 10.0.0.0 0.0.0.255 area 0",
            "interface Gig0/0",
            " description uplink",
        ]
    )
    new_cfg = "\n".join(
        [
            "interface Gig0/0",
            " description uplink",
            "router ospf 1",
            " network 10.0.0.0 0.0.0.255 area 0",
        ]
    )
    response = client.post(
        "/diff",
        data={"old_text": old_cfg, "new_text": new_cfg},
    )
    assert response.status_code == 200
    assert "差分がありません" in response.text


def test_diff_endpoint_detects_order_when_enabled():
    old_cfg = "\n".join(
        [
            "router ospf 1",
            " network 10.0.0.0 0.0.0.255 area 0",
            "interface Gig0/0",
            " description uplink",
        ]
    )
    new_cfg = "\n".join(
        [
            "interface Gig0/0",
            " description uplink",
            "router ospf 1",
            " network 10.0.0.0 0.0.0.255 area 0",
        ]
    )
    response = client.post(
        "/diff",
        data={
            "old_text": old_cfg,
            "new_text": new_cfg,
            "order_sensitive": "on",
        },
    )
    assert response.status_code == 200
    assert "差分がありません" not in response.text


def test_diff_requires_both_inputs():
    response = client.post(
        "/diff",
        data={"old_text": "router ospf 1"},
        headers={"hx-request": "true"},
    )
    assert response.status_code == 200
    assert "比較元と比較先の両方を入力してください。" in response.text


def test_diff_rejects_invalid_ignore_pattern():
    response = client.post(
        "/diff",
        data={
            "old_text": "router ospf 1",
            "new_text": "router ospf 1",
            "ignore_patterns": "[",
        },
        headers={"hx-request": "true"},
    )
    assert response.status_code == 200
    assert "正規表現の解析に失敗しました" in response.text


def test_diff_old_preview_upload_shows_content():
    response = client.post(
        "/diff/old-preview",
        files={"old_upload": ("old.cfg", b"alpha\nbeta", "text/plain")},
    )
    assert response.status_code == 200
    assert "alpha" in response.text
    assert "beta" in response.text


def test_diff_new_preview_upload_shows_content():
    response = client.post(
        "/diff/new-preview",
        files={"new_upload": ("new.cfg", b"gamma\ndelta", "text/plain")},
    )
    assert response.status_code == 200
    assert "gamma" in response.text
    assert "delta" in response.text


def test_model_upload_preview_renders_tree_and_paths():
    yang_text = "\n".join(
        [
            "module demo-interfaces {",
            "  yang-version 1.1;",
            "  namespace \"http://example.com/demo\";",
            "  prefix demo;",
            "  container interfaces {",
            "    list interface {",
            "      key \"name\";",
            "      leaf name { type string; }",
            "    }",
            "  }",
            "}",
        ]
    )
    response = client.post(
        "/model/upload-preview",
        files={"model_upload": ("demo.yang", yang_text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    assert "module: demo-interfaces" in response.text
    assert "/interfaces" in response.text
    assert "/interfaces/interface" in response.text


def test_model_submodule_includes_grouping_paths():
    yang_text = "\n".join(
        [
            "submodule demo-sub {",
            "  belongs-to demo-root { prefix demo; }",
            "  grouping demo-group {",
            "    container settings {",
            "      leaf enabled { type boolean; }",
            "    }",
            "  }",
            "}",
        ]
    )
    response = client.post(
        "/model/inspect",
        data={"model_text": yang_text},
        headers={"hx-request": "true"},
    )
    assert response.status_code == 200
    assert "grouping demo-group" in response.text
    assert "/grouping/demo-group/settings" in response.text


def test_inspector_accepts_yaml_input():
    yaml_text = "\n".join(
        [
            "interfaces:",
            "  interface:",
            "    - name: Ethernet1",
            "      state:",
            "        counters:",
            "          in-octets: 100",
            "          out-octets: 200",
        ]
    )
    response = client.post(
        "/telemetry",
        data={"telemetry_text": yaml_text},
    )
    assert response.status_code == 200
    assert "interfaces.interface[0].state.counters.in-octets" in response.text
    assert "100" in response.text


def test_inspector_rejects_invalid_json_and_yaml():
    response = client.post(
        "/telemetry",
        data={"telemetry_text": "{not-valid"},
    )
    assert response.status_code == 200
    assert "JSON/YAMLの解析に失敗しました" in response.text


def test_inspector_rejects_empty_input():
    response = client.post(
        "/telemetry",
        data={"telemetry_text": ""},
    )
    assert response.status_code == 200
    assert "JSON/YAMLの解析に失敗しました" not in response.text


def test_inspector_rejects_oversized_payload():
    too_large = "a" * (2 * 1024 * 1024 + 1)
    response = client.post(
        "/telemetry",
        data={"telemetry_text": too_large},
    )
    assert response.status_code == 200
    assert "入力サイズが上限を超えています。" in response.text


def test_model_tree_fallback_without_pyang(monkeypatch):
    monkeypatch.setattr(app_module.shutil, "which", lambda _: None)
    yang_text = "\n".join(
        [
            "module demo-interfaces {",
            "  namespace \"http://example.com/demo\";",
            "  prefix demo;",
            "  container interfaces {",
            "    leaf enabled { type boolean; }",
            "  }",
            "}",
        ]
    )
    _, _, _, tree_text, error = build_model_view(yang_text)
    assert error == ""
    assert "module: demo-interfaces" in tree_text
    assert "+-- interfaces" in tree_text


def test_model_tree_uses_pyang_when_available(monkeypatch):
    monkeypatch.setattr(app_module.shutil, "which", lambda _: "/usr/bin/pyang")
    monkeypatch.setattr(
        app_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="module: demo\n +-- foo"),
    )
    yang_text = "\n".join(
        [
            "module demo {",
            "  namespace \"http://example.com/demo\";",
            "  prefix demo;",
            "  container foo {",
            "    leaf bar { type string; }",
            "  }",
            "}",
        ]
    )
    _, _, _, tree_text, error = build_model_view(yang_text)
    assert error == ""
    assert tree_text == "module: demo\n +-- foo"


def test_regex_hx_returns_partial_only():
    response = client.post(
        "/regex",
        data={
            "pattern": "ERROR",
            "raw_text": "INFO ok\nERROR failed\nWARN note",
        },
        headers={"hx-request": "true"},
    )
    assert response.status_code == 200
    assert "<html" not in response.text
    assert "抽出結果" in response.text


def test_regex_full_request_returns_page():
    response = client.post(
        "/regex",
        data={
            "pattern": "ERROR",
            "raw_text": "INFO ok\nERROR failed\nWARN note",
        },
    )
    assert response.status_code == 200
    assert "<html" in response.text
