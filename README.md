# netdevops-toolkit

ネットワーク運用向けのテキスト解析・比較・可視化をまとめたWebツールです。

## 特長

- **Parser**: TextFSMで表形式に変換し、JMESPathで絞り込み、CSV/YAMLでダウンロード。
- **Regex**: パターン一致行を抽出して件数表示、TXTでダウンロード。
- **Diff**: ignore/target/順序オプションで差分を整理。
- **Inspector**: JSON/YAMLをツリー表示とパス一覧で可視化。
- **Model**: YANGのパス一覧・ツリー表示・サンプルJSON生成。

## クイックスタート

### uvで起動
```bash
uv sync
uv run uvicorn app:app --reload
```

### Dockerで起動
```bash
docker compose up --build
```

`http://localhost:8000` を開きます。

## 使い方

- **Parser**: テンプレートを選択/アップロード → show出力を貼り付け → パース。
- **Regex**: パターンを入力 → ログを貼り付け/アップロード → 抽出。
- **Diff**: 比較元/比較先を貼り付け/アップロード → 比較。
- **Inspector**: JSON/YAMLを貼り付け/アップロード → 可視化。
- **Model**: YANGを貼り付け/アップロード → パス/ツリー表示。

## データの扱い

- アップロードしたファイルは保存せず、処理のみに使用します。
- 入力はUTF-8想定です（不正なバイトは置換されます）。
