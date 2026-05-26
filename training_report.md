# TPU v6e-1 (単一チップ) Spot インスタンスによる LLM 事前学習 実践レポート

## 1. 概要と検証スペック
* **目的**: Google Cloud プロジェクト (`takashix-tpu`) 上で [tpuchat](https://github.com/vorushin/tpuchat) を活用し、単一の TPU v6e Spot インスタンスにおける GPT モデルの事前学習を検証する。
* **利用環境**:
  * **マシンタイプ**: `v6e-1` (32GB HBM)
  * **プロビジョニング**: Spot VM (`--spot`)
  * **データセット**: FineWeb-Edu-100B-Shuffle (52シャード)
  * **ライブラリ**: JAX (`raw JAX`), Optax, tiktoken

---

## 2. コード修正・パッチ差分 (Diff)

元の [02_train.py](https://github.com/vorushin/tpuchat/blob/master/02_train.py) は Colab ランタイム向けに最適化されているため、TPU VM 内でスタンドアロン実行できるよう以下の改修パッチを適用しました。

```diff
--- a/02_train.py
+++ b/02_train_single_patched.py
@@ -81,3 +81,4 @@
-    device_batch_size: int = 8
-    head_dim: int = 256
+    # 【修正1: HBM要件適正化 (32GB HBMのOOM回避)】
+    # 要求メモリを約35.4GBから32GB内に収めるためパラメータをスケールダウン
+    device_batch_size: int = 4
+    head_dim: int = 128

@@ -130,4 +131,4 @@
-from google.colab import userdata
-login(token=userdata.get("HF_TOKEN"))
+# 【修正2: Colab非依存・環境変数対応化】
+# from google.colab import userdata
+login(token=os.environ.get("HF_TOKEN"))

@@ -141,1 +142,2 @@
-TOKENIZER_DIR = '/content/tokenizer'
+# 【修正3: ローカル環境パス化 (ルート階層の権限エラー解消)】
+TOKENIZER_DIR = './content/tokenizer'

@@ -171,1 +173,1 @@
-DATA_DIR = '/content/base_data'
+DATA_DIR = './content/base_data'

@@ -931,1 +933,2 @@
-    %load_ext tensorboard
+# 【修正4: IPythonマジックコマンドの無効化 (SyntaxError対策)】
+#   %load_ext tensorboard
```

---

## 3. 実行経緯とトラブルシューティング（失敗とリトライの系譜）

### 試行 1: Spotインスタンスプロビジョニングと再認証トラップ
* **発生事象・失敗内容**:
  1. gcloud クライアントのローカル再認証プロンプトによりプロセスがハング。
  2. 認証クリア後、`queued-resources` のデフォルトモデルと `--spot` フラグの競合 (`INVALID_ARGUMENT: STANDARD provisioning model is incompatible with spot requests`) が発生。
* **解決策**: 
  * 即時VM生成である `gcloud alpha compute tpus tpu-vm create` に切り替え、`--version v2-alpha-tpuv6e --spot` フラグを指定してプロビジョニングに成功。

---

### 試行 2: スクリプトの CLI 実行化における権限エラー
* **発生事象・失敗内容**:
  1. マジックコマンド (`%load_ext`) による `SyntaxError`。
  2. Colab パス (`/content/...`) をそのまま参照したことによる `PermissionError: Permission denied`。
* **解決策**:
  * `sed` コマンドによりカレントディレクトリ配下 (`./content/...`) へ書き込む動的パッチを適用しクリア。

---

### 試行 3: モデル肥大化による XLA メモリ不足 (OOM)
* **発生事象・失敗内容**:
  * リポジトリコード内のデフォルト値が 1.5B 規模 (`device_batch_size=8`, `head_dim=256`) に引き上げられており、v6e-1 の実質 HBM 容量を超過して `XlaRuntimeError: RESOURCE_EXHAUSTED` が発生。
* **解決策**:
  * README の確実な仕様 (`device_batch_size=4`, `head_dim=128`) へスケールダウンし、HBM 内へ正常配置。

---

## 4. 最終トレーニング評価レポート

全 1,000 ステップの事前学習を正常に完走しました（末尾のプロット用モジュール `matplotlib` のみ欠品終了）。

### 主要メトリクス
* **合計実行時間**: 約 0.9 分 (FineWeb 52シャードの展開含む)
* **実用スループット**: 約 146,000 tok/s
* **最高検証損失 (Val Loss)**: `6.0158` (1,000ステップ終了時点)

---

### モデルによる出力テキストサンプル

```text
[Prompt]: The capital of France is
[Output]: The capital of France is an organized account for the way of research and the
As a result, the second technique, the risk of their lives and society, is the best way to develop a particular life of the human needs and it’s in the future...

[Prompt]: Machine learning is
[Output]: Machine learning is a short or a minimum problem at the Cot.
The latest value of the article is the “the same thing" to carry the way of the site, a lot of the current work in the National Association...
```
