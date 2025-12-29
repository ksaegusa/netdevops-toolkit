# netdevops-toolkit UI設計メモ

## 目的
- Parser MVP の画面を最小構成で作り、将来のYANG/XML解析へ拡張できる土台を作る。
- 導入が簡単な Tailwind CDN を採用し、FastAPI + HTMX との相性を優先する。

## 設計方針
- 1ページ構成、フォームは右側のカードに集約。
- 解析結果はページ下部にまとめ、HTMXで部分更新。
- 色は暖色系のベージュとアクセントオレンジで視認性を確保。
- フォントは見出しに明朝体、本文は読みやすいサンセリフ。

## Tailwind構成
- CDN: `https://cdn.tailwindcss.com`
- 設定: `templates/base.html` 内で `tailwind.config` を上書き
  - `fontFamily.display`: `"Shippori Mincho"`
  - `fontFamily.body`: `"IBM Plex Sans JP"`
  - `colors`: `sand`, `ink`, `ember`, `emberDark`

## HTMX構成（学習メモ）
### 何ができる？
- フォーム送信やリンクのクリックで、ページ全体ではなく「一部のHTMLだけ」を差し替えられる。
- JavaScriptをほぼ書かずに、サーバが返すHTMLをそのままUIに反映できる。
- REST APIを作ってJSONを返す代わりに、HTMLフラグメントを返す構成に向く。

### このプロジェクトでの使い方
- **Parser実行**
  - `templates/index.html` のフォームに `hx-post="/parse"` を設定。
  - 送信結果は `hx-target="#result"` で結果パネルだけ差し替え。
  - FastAPI側は `templates/partials/result.html` を返している。
- **JMESPathの再適用**
  - `templates/partials/result.html` 内のフォームに `hx-post="/filter"` を設定。
  - 直前のパース結果JSONを hidden textarea で送って再フィルタ。

### 重要な属性
- `hx-post`: フォーム送信先を指定する。通常のPOSTと同様にサーバへ送信。
- `hx-target`: 返ってきたHTMLを挿入する対象（CSSセレクタ）。
- `hx-swap`: どのように差し替えるか。ここでは `innerHTML` を利用。
- `hx-trigger`: いつ発火するか（例: selectのchange）。
- `hx-include`: 追加で送信したい要素を指定。

### 使うときの注意点
- 返すHTMLは「差し替え対象に合った構造」にする必要がある。
- エラー時もHTMLで返すと、UI側の実装がシンプルになる。
- 大きな状態管理は不得意なので、必要ならAPI/JSに段階的に移行する。
## レスポンシブ設計
- 1カラム → 2カラム: `lg:flex-row`
- カード幅は `max-w-xl` を使い、大画面で余白が残るように調整。
- 結果は `lg:grid-cols-2` で2カラム表示し、モバイルは1カラム。

## 拡張の想定
- YANG/XML解析は `/parse` に別モードを追加するか、将来 `/yang` `/xml` ページを増やす。
- 追加ページも `base.html` に従いカード+結果の構成を踏襲する。
