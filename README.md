# netdevops-toolkit
## 1. 目的・スコープ

* ネットワーク運用作業（ログ/出力の解析、抽出、showコマンド実行）の **作業時間短縮と標準化**
* 対象機能（MVP）

  * **Parser**：テンプレで構造化 → 表表示 → JMESPathで絞り込み → CSV/YAMLダウンロード
  * **Regex**：行抽出 → 件数表示 → TXTダウンロード
  * **Diff**：ignore/target/順序無視の差分表示

## 2. ユーザーストーリー（MVP）

* NOC/運用者として、show出力を貼ってテンプレを当て、必要項目だけ抜きたい
* 障害対応で、ログから特定パターンだけ抽出して共有したい
* 端末に入れない環境でも、Webからshowを叩いて結果をテキストで回収したい
* 解析結果を再現できるよう、入力・テンプレ・クエリの組を保存/共有したい（※これはMVP外でもよい）

## 3. 画面・機能要件

### 3.1 共通

* 入力：テキスト貼り付け＋ファイルアップロード（UTF-8想定、失敗時は置換）
* 出力：画面表示＋コピーしやすい整形
* エラー：原因がわかるメッセージ（例：テンプレ不一致、正規表現コンパイル失敗等）
* ログ：入力本文は残さない（少なくともデフォルトは）

### 3.2 Parser

* テンプレ一覧の表示（サーバ側に置いたテンプレを選択）
* テンプレのアップロード/直接編集に対応（保存はしない）
* show出力の貼り付け + テキストアップロードに対応
* パース結果は表形式で表示
* JMESPathはパース結果の下で実行
* CSV/YAMLダウンロードに対応
* 大きい出力でも落ちない（サイズ上限・制限）

### 3.3 Regex

* pattern入力（オプションで flags: IGNORECASE/MULTILINE 等）
* マッチ行の一覧と件数
* 抽出結果をダウンロード（txt）

### 3.4 Diff

* 旧設定/新設定の貼り付け・アップロード
* ignoreパターン（1行1正規表現）
* target prefix で特定ブロックのみ比較
* 同一階層の順序入れ替わりは差分扱いしない（オプションで順序差分を有効化）
* 追加/削除を色分けして表示

## 4. 非機能要件（最低限）

* 認証：まずは社内なら Basic/OIDC のどれか（未定でも“差し込み可能”に）
* セキュリティ：

  * secret類（セッション鍵、API key）を環境変数管理
  * CSRF対策（少なくとも同一オリジン前提／必要ならトークン）
* パフォーマンス：

  * リクエストサイズ上限（例：2〜5MB）
* 運用：

  * Dockerで起動
  * ローカル実行（開発）と本番（gunicorn/uvicorn）を分離

## 5. 受け入れ条件（MVPのDone）

* Parser/Regex/Diff 画面が動作し、結果が部分更新（HTMX）で表示される
* 入力エラー時に落ちずに理由が表示される
* Dockerで `docker compose up` で起動できる
* AWS,AppRunnerにデプロイしたい

---

### 次にやると良いこと（すぐ決めたい2点）
1. **利用形態**：ローカル専用（個人）／社内LAN（チーム）／インターネット公開（基本非推奨）どれ想定ですか？
   → ここで認証・制限の強さが変わります。

## Parser MVP (FastAPI + HTMX)

### 起動（uv）
```bash
uv sync
uv run uvicorn app:app --reload
```

### 起動（Docker）
```bash
docker compose up --build
```

### 使い方
1. `http://localhost:8000` を開く
2. テンプレート `cisco_ios_show_ip_interface_brief.textfsm` を選択（またはアップロード）
3. テンプレート本文が表示されたら必要に応じて編集
4. `show ip interface brief` の出力を貼る（またはテキストアップロード）
5. パース結果を確認し、必要なら JMESPath で絞り込み
6. CSV/YAMLでダウンロード

### メモ
* TailwindはCDN読み込み（オフライン運用時は別途ビルドが必要）
* アップロードしたテンプレ/テキストは保存せず、解析にのみ使用

## 将来対応リスト
* Telemetry Inspector（テレメトリJSON/ログの抽出・可視化）
* Config Transformer（CLI/JSONの相互変換・整形）
* OpenConfig/YANG Explorer（モデル検索・パス補完）
* gNMI Playground（Get/Set/Subscribeのリクエスト作成・検証）
* REST API対応
  * Parser API（parse + JMESPath）
  * Regex/ログ判定 API（pattern/flags + 判定ルール）
