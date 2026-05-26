# Trillium (TPU v6e) Spot インスタンスによる tpuchat LLM 事前学習 実践アーカイブ

本リポジトリは、Google Cloud の最新第6世代アクセラレータ **Trillium (TPU v6e)** の Spot インスタンスを活用し、[tpuchat](https://github.com/vorushin/tpuchat) リポジトリをベースとした JAX 事前学習スクリプトの実行検証・トラブルシューティング結果・拡張ノウハウを収録したナレッジポータルです。

---

## 1. 検証サマリー

Colabベースで公開されていた JAX/GPT 事前学習スクリプトに対し、TPU VM 内での自律駆動パッチングを行い、「単一TPUチップ (`v6e-1`)」および「8チップ分散メッシュ (`v6e-8`)」の双方で 1,000 ステップのオンザフライ学習・生成実験を行いました。

### 検証フェーズ比較サマリー

| 項目 | ① 単一TPU検証 (`v6e-1`) | ② 分散TPU検証 (`v6e-8`) |
| :--- | :--- | :--- |
| **稼働チップ数** | 1 コア (HBM 32GB) | **8 コア (2x4 / HBM 256GB)** |
| **バッチ/ステップ** | 4 バッチ (8,192 トークン) | **32 バッチ (65,536 トークン)** |
| **最高検証損失** | `6.0158` | **`4.9253` (精度アップ)** |
| **実行レポート** | [training_report.md](training_report.md) | [distributed_training_report.md](distributed_training_report.md) |
| **修正版コード** | `modified_tpuchat/02_train.py` | `modified_tpuchat/02_train_distributed.py` |

---

## 2. ディレクトリ・成果物一覧

```text
.
├── README.md                      # 本ドキュメント（全体概要・手動検証ガイド）
├── training_report.md             # 単一TPUでの学習検証レポートとパッチ差分
├── distributed_training_report.md # 8チップ分散メッシュ検証レポートとパッチ差分
└── modified_tpuchat/              # 再生・検証に利用したコード一式
    ├── 02_train.py                # 単一TPU向けに権限・OOM対策を適用したコード
    └── 02_train_distributed.py    # JAX Mesh / NamedSharding を追加した分散用コード
```

---

## 3. 手動で構築・検証を行う場合の実施手順 (Tutorial)

以下のステップを実行することで、遭遇しうる全てのエラー（gcloud認証、Spotオプション競合、Colabパス不整合、XLA OOM、Mosaic分割制約）を回避し、同一環境を再現できます。

### 【事前準備】
* **GCP プロジェクト**: `takashix-tpu`
* **デプロイ対象ゾーン**: `asia-northeast1-b`
* **API トークン**: HuggingFace Hub の Access Token (`HF_TOKEN`)

---

### 手順 A: 単一TPU (`v6e-1`) の検証を実行する場合

#### 1. Spot インスタンスの確保
```bash
gcloud alpha compute tpus tpu-vm create tpuchat-v6e \
    --project takashix-tpu --zone asia-northeast1-b \
    --accelerator-type v6e-1 --version v2-alpha-tpuv6e --spot
```

#### 2. 環境設定・パッチ適用・トレーニング実行
```bash
gcloud alpha compute tpus tpu-vm ssh tpuchat-v6e \
    --project takashix-tpu --zone asia-northeast1-b \
    --command "git clone https://github.com/vorushin/tpuchat.git && cd tpuchat && \
               pip install -q 'jax[tpu]' optax huggingface_hub tiktoken pyarrow requests torch tensorboard tensorboard-plugin-profile matplotlib && \
               sed -i 's/\/content/.\/content/g' 02_train.py && \
               sed -i 's/^[[:space:]]*%/# %/g' 02_train.py && \
               sed -i 's/from google.colab import userdata/# from google.colab import userdata/g' 02_train.py && \
               sed -i 's/userdata.get(\"HF_TOKEN\")/os.environ.get(\"HF_TOKEN\")/g' 02_train.py && \
               sed -i 's/device_batch_size: int = 8/device_batch_size: int = 4/g' 02_train.py && \
               sed -i 's/head_dim: int = 256/head_dim: int = 128/g' 02_train.py && \
               export HF_TOKEN=<あなたのHuggingFaceトークン> && \
               python3 02_train.py"
```

---

### 手順 B: 分散TPU (`v6e-8`) の検証を実行する場合

#### 1. 8コア Spot インスタンスの確保
```bash
gcloud alpha compute tpus tpu-vm create tpuchat-v6e-8x \
    --project takashix-tpu --zone asia-northeast1-b \
    --accelerator-type v6e-8 --version v2-alpha-tpuv6e --spot
```

#### 2. 分散メッシュロジックのパッチ適用と並列トレーニング実行
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
c = c.replace('import jax\nimport jax.numpy as jnp', 'import jax\nimport jax.numpy as jnp\n\n# 【修正1: 分散メッシュとシャーディング宣言の追加】\nfrom jax.sharding import Mesh, PartitionSpec as P, NamedSharding\nmesh = Mesh(jax.devices(), (\'batch\',))\ndata_sharding = NamedSharding(mesh, P(\'batch\', None))\n')
c = c.replace('config.device_batch_size, config.seq_len', 'config.device_batch_size * len(jax.devices()), config.seq_len')
c = c.replace('loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)', '# 【修正6: JAX SPMDデバイス再配備 (device_put)】\n    x_batch = jax.device_put(x_batch, data_sharding)\n    y_batch = jax.device_put(y_batch, data_sharding)\n    loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)')
with open('02_train_distributed.py', 'w') as f:
    f.write(c)
\" && export HF_TOKEN=<あなたのHuggingFaceトークン> && python3 02_train_distributed.py"
```
