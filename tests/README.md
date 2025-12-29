# テスト方針

## 目的
- Parserの主要フロー（TextFSMパースとJMESPath適用）が崩れないことを確認する。
- UIの結果表示はテンプレ由来のHTMLが返ることを前提とし、API/ロジック側の整合性を優先する。

## 対象範囲
- 正規化処理（JMESPathのクォート補正）
- パース処理（テンプレ本文を直接入力した場合）
- フィルタ処理（JMESPathを適用した結果）

## 方針
- FastAPIのTestClientでエンドポイント単位の検証を行う。
- 例外の詳細よりも「期待する文字列が返ること」を重視する。
- 入力サイズの上限やファイルアップロードの詳細検証は今後追加する。

# テスト内容

## tests/test_app.py
- `test_normalize_jmespath_preserves_backticks`
  - バッククォートを含む式が保持されることを確認。
- `test_normalize_jmespath_query_converts_smart_quotes`
  - スマートクォートがASCIIクォートに変換されることを確認。
- `test_filter_endpoint_accepts_double_quotes`
  - `"` を含む式が評価されることを確認。
- `test_parse_with_template_body`
  - テンプレ本文の直接入力でパース結果が返ることを確認。

## 実行方法
```bash
uv run pytest
```

## CI（GitHub Actions）
- `push` / `pull_request` で `pytest` を実行。
- `uv` を使って `pyproject.toml` から依存を解決し、テストを実行する。
