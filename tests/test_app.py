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
