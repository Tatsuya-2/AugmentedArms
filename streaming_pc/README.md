# Streaming PC — EEG live viewer / recorder / replay

B2J PC (`B2J-User.py` + OpenBCI Cyton) からリアルタイムにストリーミングされた
生 EEG を、別 PC で可視化・録画・再生するためのツール群です。

```
[B2J PC]  ──ws→  [Streaming PC: eeg_viewer.py (server)]  ──→  PyQtGraph 表示
                                       │
                                       └─ --record-dir → JSONL に保存

[後日]  replay_sender.py ──ws→  [Streaming PC: eeg_viewer.py (server)]  ──→  再現表示
```

ポイント:

- **ビューアは常に WebSocket サーバ**。ライブ時は B2J が、再生時は
  `replay_sender.py` がクライアントとして接続する。
- B2J 側のコードを差し替える必要はない (記録した JSONL を
  `replay_sender.py` で流すだけで同じ可視化を再現できる)。
- 記録は JSONL のため `jq` / `head` / `wc -l` 等で目視検査可能。

---

## セットアップ

streaming PC 側:

```bash
pip3 install -r requirements.txt    # websockets, pyqtgraph, PyQt5, numpy
pip3 install websocket-client       # replay_sender.py が使う
```

B2J PC 側はリポジトリ既存環境に `websocket-client` (DroneMonitorClient と
共通) が既にあれば追加インストール不要。

---

## ワークフロー A — ライブ可視化 + 録画

1. **Streaming PC でビューア起動** (録画ディレクトリ指定):

   ```bash
   python3 eeg_viewer.py \
       --host 0.0.0.0 --port 9091 \
       --record-dir ~/eeg_recordings
   ```

   接続待ち状態になる。`--record-dir` を指定すると、接続セッションごとに
   `eeg_<user_id>_<YYYY-MM-DD_HH-MM-SS>.jsonl` を自動生成して append する。

2. **B2J PC でクライアント起動**:

   ```bash
   python3 B2J-User.py \
       --eeg-stream-host <streaming-pc-ip> \
       --eeg-stream-port 9091
   ```

   フラグを付けない場合は既存の M5 シリアル動作のままで、ストリーミングは
   発生しない (デフォルト挙動と完全に等価)。

3. ビューアに 8ch 波形が流れ始め、ステータスバーに
   `recording to /home/.../eeg_xxx.jsonl` が表示されたら録画中。
4. B2J 側を Ctrl-C で停止するか WebSocket を切断するとビューア側の
   JSONL ファイルがクローズされる。再接続で新ファイルになる。

### 記録ファイルのサイズ目安

- 250 Hz × 8ch × JSON = だいたい **70–100 MB / 時間**。

### 記録ファイルの中身チェック

```bash
# 行数 (register 1 行 + batch 多数)
wc -l eeg_*.jsonl

# 最初と最後の seq_start で連続性をざっくり見る
head -2 eeg_*.jsonl | tail -1 | jq '{seq_start, t_start, n: (.samples | length)}'
tail -1 eeg_*.jsonl                    | jq '{seq_start, t_start, n: (.samples | length)}'

# サンプルレート確認 (登録行)
head -1 eeg_*.jsonl | jq '{sample_rate_hz, channels, user_id, units}'
```

---

## ワークフロー B — 後日検証 (再生)

1. **ビューアを起動** (ライブ時と同じコマンドで OK。録画は任意):

   ```bash
   python3 eeg_viewer.py --port 9091
   ```

2. **保存した JSONL を replay_sender で送り込む**:

   ```bash
   python3 replay_sender.py \
       --file ~/eeg_recordings/eeg_123_2026-05-21_10-22-15.jsonl \
       --host 127.0.0.1 --port 9091 \
       --speed 1.0
   ```

   `--speed` を上げれば早送り、`--loop` で無限ループ再生になる。
   ペーシングは記録された `t_start` の差分に基づくため、ネットワーク
   ジッターを含まない理想的なタイミングで再現される。

### 再生のオプション

| フラグ                       | 用途                                                          |
| ---------------------------- | ------------------------------------------------------------- |
| `--speed N`                  | N 倍速 (0.5 = スロー、10.0 = 早送り)                          |
| `--loop`                     | 終端で頭に戻り無限再生 (2 周目以降は register をスキップ)     |
| `--no-register`              | 記録の register 行を送らない                                  |
| `--synthetic-register '...'` | register 行が無い古い記録に対し、合成 JSON を 1 度だけ送る    |

---

## JSONL フォーマット

1 行 1 JSON、ストリーム中の WebSocket メッセージそのまま。

**1 行目 (register)**:

```json
{"type":"register","role":"eeg_sender","user_id":"123456789",
 "sample_rate_hz":250,"channels":8,"units":"uV"}
```

**2 行目以降 (eeg_batch、約 10 Hz)**:

```json
{"type":"eeg_batch","user_id":"123456789",
 "sample_rate_hz":250,"channels":8,
 "seq_start":12345,
 "t_start":1716271234.567,"t_end":1716271234.663,
 "samples":[[c1,c2,c3,c4,c5,c6,c7,c8], ...25 行...],
 "labels":["","",...]}
```

- `seq_start`: バッチ先頭サンプルの通し番号 (連続性チェックに使う)
- `t_start` / `t_end`: ボードのタイムスタンプ (BrainFlow 由来、UNIX epoch 秒)
- `samples`: float の 2 次元配列 (バッチ内サンプル数 × チャンネル数)
- `labels`: B2J 側の `stimulus_sound` を 1 サンプルごとに展開した文字列配列
  (刺激区間の特定に使う)

---

## ファイル一覧

| ファイル              | 役割                                                    |
| --------------------- | ------------------------------------------------------- |
| `eeg_viewer.py`       | WebSocket サーバ + PyQtGraph 8ch 波形ビューア + 録画機能 |
| `replay_sender.py`    | JSONL をビューアへ再送信する CLI                        |
| `requirements.txt`    | streaming PC 側の Python 依存                           |

B2J PC 側のペアコンポーネント (このディレクトリの外):

| ファイル                              | 役割                                                  |
| ------------------------------------- | ----------------------------------------------------- |
| `../EEGStreamClient.py`               | B2J 上で動く WebSocket クライアント (バッチ送信)      |
| `../ABMI_Utils.py` (`BCIBoard`)       | サンプル tee の取り付け口                             |
| `../B2J-User.py`                      | `--eeg-stream-host` でストリーミングをオプトイン      |

---

## トラブルシューティング

- **波形が描画されない**: ビューアのステータスに
  `client connected` が表示されているか確認。B2J 側で
  `[EEGStreamClient] connected` ログが出ているかも確認。
- **`connection error: Connection refused`**: ビューア起動前に B2J が
  接続しに来た可能性。`EEGStreamClient` は 2 秒後に自動再接続するので、
  ビューアを起動すれば自然に繋がる。
- **`seq_start` が飛ぶ**: B2J 側の input queue (2000 サンプル) が
  オーバフローしている可能性。送信速度より受信処理が遅い場合に発生。
  ネットワーク改善か `--eeg-batch-size` を増やす。
- **再生のタイミングがおかしい**: 記録が古いフォーマットだと `t_start`
  が欠落している場合がある → そのときは一律 0 秒間隔で送ってしまうため、
  `--speed` で代用するか記録を取り直す。
