from fastapi.testclient import TestClient

from app import app, normalize_jmespath_for_eval, normalize_jmespath_query


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


def test_regex_endpoint_extracts_lines():
    response = client.post(
        "/regex",
        data={
            "pattern": "ERROR",
            "raw_text": "INFO ok\nERROR failed\nWARN note",
        },
    )
    assert response.status_code == 200
    assert "ERROR failed" in response.text
    assert "WARN note" not in response.text


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
