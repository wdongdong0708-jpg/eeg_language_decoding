# ChineseEEG EEG–Language Retrieval

本项目研究 EEG 表征能否与文本或语音表征映射到共享空间，并用于
EEG→文本、EEG→语音片段检索。首要目标是排除时长、字符数、padding、
音频包络、受试者身份和故事位置等捷径，而不是单纯最大化检索准确率。

## 当前阶段

本仓库目前完成了研究骨架、数据契约、全量只读数据审计、完整
stimulus-row trial manifest、seed-42 协议级划分，以及文本/音频多粒度特征
接口。首个 ChineseEEG2 PL EEG–语音序列检索基线已完成窗口、特征、模型、
损失、评测和烟雾训练验证；正式多 seed 训练尚未作为科学结果发布。没有
修改或重新预处理任何原始/官方 EEG 数据。

正式 manifest 及其审计：

- `metadata/all_trials.parquet` 与 `metadata/all_trials.csv`
- `metadata/manifest_schema.md`
- `metadata/normalization_rules.json`
- `metadata/text_alignment_overrides.csv`
- `reports/manifest_audit.md` 与 `reports/manifest_audit.json`

协议级划分独立于 manifest 内置的 seed-20260730 split，且不会覆盖它：

- `splits/text_unseen_seed42.json`
- `splits/subject_text_unseen_seed42.json`
- `splits/cross_paradigm_seed42.json`
- `metadata/split_protocol_schema.md`
- `reports/split_protocol_audit.md` 与
  `reports/split_protocol_audit.json`

PL EEG–语音基线：

- `metadata/pl_speech_windows_seed42_3s_delay_000ms.jsonl`
- `metadata/pl_speech_window_schema.md`
- `reports/pl_speech_windows_seed42_3s_delay_000ms.md`
- `reports/pl_speech_shortcuts_seed42_delay000ms.md`
- `reports/pl_speech_delay_sweep_seed42.md`
- `reports/pl_speech_baseline_validation.md`

重新构建和独立核验：

```powershell
conda activate bm5060
python scripts/build_manifest.py
python scripts/create_splits.py
python scripts/audit_manifest.py
python -m pytest
```

已审阅的官方代码版本：

- 仓库：<https://github.com/ncclab-sustech/ChineseEEG-2>
- commit：`b1c4ba8afd738383cf5d4676f65256aa43d876e3`
- 目录：`data_preprocessing/`、`embeddings/`、`novel_segmentation/`、
  `experiment/`

机器可读与可读版审计见
[`reports/data_audit.json`](reports/data_audit.json) 和
[`reports/data_audit.md`](reports/data_audit.md)；特征字段和时间语义见
[`metadata/feature_schema.md`](metadata/feature_schema.md)，当前验证边界见
[`reports/feature_pipeline_validation.md`](reports/feature_pipeline_validation.md)。

## 科学约束

以下规则是代码接口的一部分：

1. 内容 partition 只由稳定 `split_group_id` 决定；受试者、范式、trial/
   窗口索引和 DataLoader 顺序不得参与内容划分。未见受试者协议另行对带
   dataset/cohort 命名空间的 `subject_group_id` 做独立划分。
2. 同一内容块的所有受试者、session、范式和滑动窗口必须继承同一 split。
3. 窗口不得跨 block 或 split 边界；音频特征提取遵守相同边界。
4. 固定字符节奏只描述视觉呈现时间，不能用作朗读词级发音时间。
5. 主要结果至少报告未见文本评测，并包含未见受试者协议。
6. 主要文本检索结果必须包含长度匹配候选池。
7. random、duration-only、character-count-only、padding-mask-only、
   sentence-position-only、subject-ID-only 和 audio-envelope baseline
   均为必需项。

## 数据位置

配置文件默认指向当前机器上已存在的数据，不会触发下载：

- ChineseEEG1：
  `D:/dataset/ChineseEEG/derivatives/preproc/filtered_0.5_30`
- ChineseEEG2 PL：
  `D:/dataset/ChineseEEG-2/PassiveListening/derivatives/preprocessed`
- ChineseEEG2 RA：
  `D:/dataset/ChineseEEG-2/ReadingAloud/derivatives/preprocessed`
- ChineseEEG2 材料、音频与官方 embedding：
  `D:/dataset/ChineseEEG-2/materials&embeddings`

注意：任务说明中的
`D:/dataset/ChineseEEG/filtered_0.5_30` 在当前文件系统中不存在；实际数据
多了一层 `derivatives/preproc/`。代码不会静默切换数据版本，路径必须在
audit 阶段显式确认。

## 项目结构

```text
configs/       数据、预处理、模型、实验和评测协议
src/data/      manifest、内容身份、划分、窗口和对齐
src/features/  文本/音频特征及缓存
src/models/    EEG 编码器、投影头、检索模型和损失
src/training/  训练、采样与 checkpoint
src/evaluation/候选池、检索指标、捷径 baseline 和统计检验
scripts/       分阶段命令行入口
tests/         防泄漏与数据契约测试
metadata/      人工核验映射和数据版本说明
splits/        经审计的版本化协议级 split 产物
reports/       audit、质量控制与实验报告
experiments/   运行产物约定
notebooks/     探索性分析；不得承载唯一核心逻辑
```

## 环境与基础检查

```powershell
conda activate bm5060
python -m pytest
```

若需要以 editable 方式安装：

```powershell
python -m pip install -e ".[features,dev]"
```

复跑只读数据审计：

```powershell
python scripts/audit_data.py `
  --json-output reports/data_audit.json `
  --markdown-output reports/data_audit.md
```

文本特征输入是每行包含 `content_id`、`text` 的 JSONL；输出同时保存句级、
token 级、字符级 hidden state 和原文 offset：

```powershell
python scripts/extract_text_features.py `
  --input-jsonl metadata/text_blocks.jsonl `
  --output-dir experiments/features/text `
  --device cuda `
  --local-files-only
```

音频特征输入必须已经在 manifest/block 层定义好 `content_id`、继承的
`split` 与 `start_sec`/`stop_sec`，输出不做时间池化的 wav2vec frame：

```powershell
python scripts/extract_audio_features.py `
  --input-jsonl metadata/audio_blocks.jsonl `
  --output-dir experiments/features/audio `
  --device cuda `
  --local-files-only
```

`--local-files-only` 会自动启用严格 Hugging Face 离线模式，并先把模型 ID
解析为本机缓存的 snapshot 目录；缓存不完整时立即失败，不允许联网 fallback。

PL 基线的顺序化音频目标只平均配置指定的 wav2vec hidden-state 层，不做
时间 pooling，并插值为与 EEG 相同的 750 个时间点：

```powershell
python scripts/build_pl_speech_windows.py
python scripts/extract_pl_speech_features.py --local-files-only
python scripts/evaluate_pl_shortcuts.py
python scripts/train_speech_retrieval.py
python scripts/prepare_pl_delay_sweep.py
python scripts/run_pl_delay_sweep.py
```

延迟搜索值位于 `configs/experiment/pl_speech_retrieval.yaml`。每个 delay
必须构建独立窗口索引和实验目录，不得在同一训练运行内混用。

## 分阶段路线

1. 数据审计：建立真实 BIDS/材料/音频清单，核对每个 run 的事件数、文本数、
   音频边界和缺失项。
2. manifest：构造 canonical block、`content_id`、对齐证据和来源版本。
3. split：先按 content block 划分，再生成窗口；另建未见受试者协议。
4. PL EEG–语音单元测试：固定窗口、严格同步、低层捷径 baseline。
5. EEG–文本主线：句级和局部片段检索、长度匹配候选池。
6. RA 与跨范式：区分视觉呈现、发音运动、肌电、听觉反馈和语义贡献。
7. 统计评测：多 seed、受试者 bootstrap/permutation、置信区间和消融。
