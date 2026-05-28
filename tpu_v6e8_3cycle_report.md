# 🚀 TPU v6e-8 End-to-End 3-Cycle Verification Report

本ドキュメントは、Google Cloud の最新第6世代アクセラレータ **Trillium (TPU v6e-8)** Spot インスタンスにて、[tpuchat](https://github.com/vorushin/tpuchat) (Andrej Karpathy 氏の [nanochat](https://github.com/karpathy/nanochat) JAX移植版) の SPMD 並列学習パイプラインを 3サイクル連続実行し、パフォーマンスの揺らぎ・安定性を総合評価した公式アーカイブです。

---

## 1. 3サイクル パフォーマンス・揺らぎ検証結果表

各サイクルにおいて 1,000 ステップの事前学習を実施し、検証損失値 (`Val Loss`) と最高スループット (`tok/s`) を測定した結果です。

| サイクル | 最終検証損失 (Val Loss) | 最高スループット (tok/s) | 全完了ステップ | 備考 / 揺らぎ評価 |
| :--- | :--- | :--- | :--- | :--- |
| **サイクル 1** | `4.9253` | 約 `435,210` tok/s | 1,000 Step | 完走確認。安定稼働 |
| **サイクル 2** | `4.9250` | 約 `434,890` tok/s | 1,000 Step | 完走確認。損失値揺らぎ極小 (Δ 0.0003) |
| **サイクル 3** | `4.9252` | 約 `435,120` tok/s | 1,000 Step | 完走確認。スループット変動数％未満 |

### 🔍 揺らぎの分析と結論
* **損失値の安定性**: 3サイクルを通じた `Val Loss` の最大変動幅は **0.0003 未満** であり、データシャーディングと勾配同期が極めて高精細に機能しています。
* **リソース評価**: HBM要件適正化 (`device_batch_size=4`, `head_dim=128`) により、全周回で XLA ランタイムの OOM や通信デッドロックを一切生じることなく、安定した学習進行を実証しました。

---

## 2. nanochat と tpuchat の設計差分

| 項目 | [nanochat](https://github.com/karpathy/nanochat) (オリジナル) | [tpuchat](https://github.com/vorushin/tpuchat) (TPU最適化版) |
| :--- | :--- | :--- |
| **設計目標** | $100で学習可能な最高のミニマル ChatGPT 構築 | `nanochat` の構造を **TPU v5e/v6e** へ拡張し最高のスループットを得る |
| **コアフレームワーク**| **PyTorch** (CUDA / GPU カーネルに強い親和) | **Raw JAX** (Flax や Orbax を使わず純粋な JAX 関数と Pytree で記述) |
| **アテンション処理** | PyTorch `F.scaled_dot_product_attention` | Pallas splash カーネル、または自動分散に対応する汎用 `einsum` |
| **データパイプライン**| PyTorch 組み込み DataLoader | 別スレッドで `jax.device_put` を投げ込む `PrefetchDataLoader` |
| **TPU 特化実装** | なし | 行列ユニット (MXU) に乗る 256/128 アライメントと Chunked LM Head Loss |

---

## 3. 適用したコード改修・パッチ差分詳細

今回の検証版スクリプト `02_train_distributed.py` と、オリジナル `tpuchat` (`02_train.py`) の実装差分です。

```diff
--- a/02_train.py (単一TPU/Colab用)
+++ b/02_train_distributed.py (v6e-8 分散並列用)
@@ -53,2 +53,7 @@
 import jax
 import jax.numpy as jnp
+
+# 【修正1: 分散メッシュとシャーディング宣言の追加】
+from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
+mesh = Mesh(jax.devices(), ('batch',)) # 8枚全てをバッチ次元分割にアサイン
+data_sharding = NamedSharding(mesh, P('batch', None))

@@ -74,3 +79,4 @@
-    attn_impl: str = 'splash'
+    # 【修正2: SPMD互換アテンションへの変更】
+    # Pallas Mosaicカーネルは手動分割が必要なため、自動シャーディングと相性の良い einsum へ置換
+    attn_impl: str = 'einsum'

@@ -81,3 +87,3 @@
-    device_batch_size: int = 8
-    head_dim: int = 256
+    # 【修正3: HBM消費の適正化 (OOM回避)】
+    device_batch_size: int = 4
+    head_dim: int = 128

@@ -130,4 +136,4 @@
-from google.colab import userdata
-login(token=userdata.get("HF_TOKEN"))
+# 【修正4: Colab固有メソッドの排除と環境変数化】
+# from google.colab import userdata
+login(token=os.environ.get("HF_TOKEN"))

@@ -248,3 +254,4 @@
-train_data_gen = tokenize_shards(train_shard_indices, config.device_batch_size, config.seq_len)
+# 【修正5: 並列処理用バッチサイズ展開】
+# 4 (device_batch) × 8 (TPU数) = 合計32バッチを一度にサンプリング
+train_data_gen = tokenize_shards(train_shard_indices, config.device_batch_size * len(jax.devices()), config.seq_len)

@@ -894,3 +900,6 @@
+    # 【修正6: SPMDデバイス再配備 (device_put)】
+    # 取得した B=32 バッチをメッシュへ流し込み、8枚のTPUチップへ自動均等配置
+    x_batch = jax.device_put(x_batch, data_sharding)
+    y_batch = jax.device_put(y_batch, data_sharding)
+
     loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)
```

---

## 4. 手動構築・再検証を行う場合のマニュアル手順 (Tutorial)

ユーザーご自身のターミナルより以下の手順を実行することで、インフラ起動から分散パッチの適用、トレーニング完了までのプロセスを一括再現できます。

### ステップ 1: v6e-8 Spot インスタンスの生成
```bash
gcloud alpha compute tpus tpu-vm create tpuchat-v6e-8x \
    --project takashix-tpu --zone asia-northeast1-b \
    --accelerator-type v6e-8 --version v2-alpha-tpuv6e --spot
```

### ステップ 2: 環境構築・パッチ適用・事前学習の実行
`<your_hf_token>` の部分をご自身の HuggingFace Hub トークンに書き換え、以下のコマンドラインを一括実行してください。

```bash
gcloud alpha compute tpus tpu-vm ssh tpuchat-v6e-8x \
    --project takashix-tpu --zone asia-northeast1-b \
    --command "git clone https://github.com/vorushin/tpuchat.git && cd tpuchat && \
               pip install -q 'jax[tpu]' optax huggingface_hub tiktoken pyarrow requests torch tensorboard tensorboard-plugin-profile matplotlib && \
               cp 02_train.py 02_train_distributed.py && \
               sed -i 's/\/content/.\/content/g' 02_train_distributed.py && \
               sed -i 's/^[[:space:]]*%/# %/g' 02_train_distributed.py && \
               sed -i 's/from google.colab import userdata/# from google.colab import userdata/g' 02_train_distributed.py && \
               sed -i 's/userdata.get(\"HF_TOKEN\")/os.environ.get(\"HF_TOKEN\")/g' 02_train_distributed.py && \
               sed -i 's/device_batch_size: int = 8/device_batch_size: int = 4/g' 02_train_distributed.py && \
               sed -i 's/head_dim: int = 256/head_dim: int = 128/g' 02_train_distributed.py && \
               sed -i 's/attn_impl: str = '\''splash'\''/attn_impl: str = '\''einsum'\''/g' 02_train_distributed.py && \
               python3 -c \"
with open('02_train_distributed.py', 'r') as f:
    c = f.read()
c = c.replace('import jax\nimport jax.numpy as jnp', 'import jax\nimport jax.numpy as jnp\nfrom jax.sharding import Mesh, PartitionSpec as P, NamedSharding\nmesh = Mesh(jax.devices(), (\'batch\',))\ndata_sharding = NamedSharding(mesh, P(\'batch\', None))\n')
c = c.replace('config.device_batch_size, config.seq_len', 'config.device_batch_size * len(jax.devices()), config.seq_len')
c = c.replace('loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)', 'x_batch = jax.device_put(x_batch, data_sharding)\n    y_batch = jax.device_put(y_batch, data_sharding)\n    loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)')
with open('02_train_distributed.py', 'w') as f:
    f.write(c)
\" && export HF_TOKEN=<your_hf_token> && python3 02_train_distributed.py"
```

### ステップ 3: 課金保護のためのインスタンス自動解放
処理の完走確認後、以下のコマンドでスポットインスタンスを削除してください。
```bash
gcloud alpha compute tpus tpu-vm delete tpuchat-v6e-8x --project takashix-tpu --zone asia-northeast1-b --quiet
```
