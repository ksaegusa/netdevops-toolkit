# netdevops-toolkit UI設計メモ

## 目的
- Parser/Regex/Diff/Inspector/Model を並列で提供する運用向けUIを整える。
- Tailwind CDN と FastAPI + HTMX の組み合わせで、軽量に画面更新を行う。

## 画面構成
- 複数ページ構成（Parser / Regex / Diff / Inspector / Model）。
- 入力フォームはページ上部、結果は下部に配置。
- ファイルアップロードは各ページの入力欄の上部に統一。
- クリアはページリロード型で統一。

## ビジュアル方針
- 淡いブルー基調（`sand`/`ink`/`ember` 系）。
- 背景はグラデーションで奥行きを出す。
- 見出しは装飾性の高いフォント、本文は読みやすいサンセリフ。

## Tailwind構成
- CDN: `https://cdn.tailwindcss.com`
- 設定: `templates/base.html` 内で `tailwind.config` を上書き
  - `fontFamily.display`: `"Alegreya Sans SC"`
  - `fontFamily.body`: `"Noto Sans JP"`
  - `colors`: `sand`, `ink`, `ember`, `emberDark`

## HTMX構成
### 何ができる？
- フォーム送信で、ページ全体ではなく一部HTMLを差し替える。
- サーバ側は HTML フラグメントを返すだけで UI を更新可能。

### このプロジェクトでの使い方
- **Parser実行**
  - `templates/index.html` に `hx-post="/parse"`。
  - `hx-target="#result"` で結果領域のみ更新。
- **JMESPathの再適用**
  - `templates/partials/result.html` のフォームで `hx-post="/filter"`。
- **Regex/Diff/Inspector/Model**
  - 各ページの結果領域を `hx-target` で更新。
  - HTMX未使用時はフルページ返却にフォールバック。

### 注意点
- 差し替え先のHTML構造と `hx-target` を一致させる。
- エラー時もフラグメントで返すとUIが簡潔になる。

## レスポンシブ設計
- 1カラム → 2カラム: `lg:grid-cols-2`
- 主要カードは `max-w-screen-2xl` で横幅を確保。
- 結果領域はスクロール可能な領域に収める。

## 拡張の想定
- gNMI Inspector など、実データ入力の強化。
- YANG関連は解析の深掘り（import/uses解決）を段階的に追加。
