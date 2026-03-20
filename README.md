## CheXRel Global Report

This repository implements a two-stage pipeline for chest X-ray **image–text alignment** and **report generation/conditioned decoding** over temporal (current vs. prior) image pairs.

The project is driven by `main.py` (PyTorch Lightning). The core models are implemented in `model/model_diff.py` as:
- `Pretrain`: contrastive pretraining with multi-view/time-difference supervision
- `Finetune`: conditional generation of reports using a GPT-style decoder with multimodal fusion

---

## Key Ideas

### 1) Temporal samples (current vs. prior)
Training samples contain:
- current image (and its Rad-DINO representation)
- prior image (and its Rad-DINO representation)
- report text for the current time
- a “difference/comparison” text describing changes between current and prior
- optional prompt components: `indication` and `context` (prior report content)

### 2) Two pre-trained vision backbones
From the code paths in `model/model_diff.py`:
- `image_encoder_1`: `CheXRelFormer_FeatureExtractor` (optionally loaded/frozen; also used as a source of “diff” features)
- `image_encoder_2`: HuggingFace `AutoModel` loaded from `--rad_dino_path` (frozen; used for local/global visual features)

### 3) One text stack
- Text encoder: HuggingFace `AutoModel` loaded from `--cxr_bert_path` (trainable; used for contrastive objectives in pretraining, and for prompt encoding in finetuning)
- Text decoder (finetuning): HuggingFace GPT2-style `GPT2LMHeadModel` initialized from `--distilgpt2_path` (with cross-attention enabled)

### 4) Training objectives
- `Pretrain` losses (logged during training/val/test):
  - `instance_loss`: global image/text contrastive (multi-positive InfoNCE)
  - `sen_text_loss`: token-level local cross-modal alignment (optional, controlled by `--using_local_loss`)
  - `diff_loss`: contrastive alignment between difference-image embeddings and difference texts (optional, controlled by `--using_diff_loss`)
- `Finetune` loss:
  - `lm_loss`: decoder language modeling loss (teacher forcing)

---

## Directory Layout

- `main.py`: entry point (arg parsing, tokenizer setup, Lightning Trainer, routes to pretrain/finetune/test)
- `model/`
  - `model_diff.py`: `Pretrain` and `Finetune` LightningModules (main implementations)
- `dataset/`
  - `datasets.py`: `ReportDataset` (current implementation for both pretrain and finetune) + older dataset utilities
  - `dataloaders.py`: `DataLoaders` + collate functions
- `temporal_datasets/with-indication/`: **preprocessed temporal JSONs** used by `ReportDataset`
- `tools/metrics/`: evaluation metrics (COCO-style captioning + RadGraph + CheXbert-style metrics, etc.)
- `ext/`: external checkpoints/models (examples and model folders)

---

## Requirements

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. This repo also imports `openai` (for DeepSeek-based comparison/report preprocessing) and `torchvision` (for image transforms). If your environment does not already have them, install:
   ```bash
   pip install openai torchvision
   ```

Notes:
- `main.py` uses Lightning with `precision="bf16-mixed"`. A GPU/driver stack that supports bf16 is recommended.

---

## Data Preparation

### Important: the current `ReportDataset` uses hardcoded temporal JSON paths
`dataset/datasets.py` loads:
`temporal_datasets/with-indication/mimic-context-{split}-final-withind.json`
for `split in {train, val, test}`.

So, `--ann_path` in `main.py` is effectively not used by the current `ReportDataset`.

The JSON file is expected to be a dictionary keyed by sample index (stringified integers). Each sample contains (as referenced in `ReportDataset.__getitem__`):
- `id`
- `subject_id`, `study_id`
- `image_path`: a list; current image path is read as `image_path[0]`
- `context_image`: a list; prior image path is read as `context_image[0]`
- `report`: current report text
- `report_base`: reference report text (used for decoding evaluation/logging)
- `indication_pure_pro`: indication prompt text
- `context`: prior-report context text (used as prompt when `--is_prior_report` is enabled)
- `comprision`: comparison/difference text between current and prior
- `is_aligened`: alignment label/flag used by pretraining diff contrastive loss

### Image paths
`ReportDataset` reads images by:
`os.path.join(args.images_dir, image_path_cur[0])`
and similarly for `context_image`.

So you must ensure `--images_dir` matches the root folder of the DICOM-exported/converted JPEG images referenced inside the JSON.

---

## External Model Checkpoints

`main.py` and `model/model_diff.py` rely on multiple external checkpoints/paths (often local directories on your machine). Key CLI arguments in `utils.py` include:

Tokenizer/decoder:
- `--distilgpt2_path`: local DistilGPT2 / GPT2 directory (used by `GPT2TokenizerFast.from_pretrained(...)`)

Text encoder (HuggingFace model):
- `--cxr_bert_path

Metrics models:
- `--chexbert_path`: CheXbert checkpoint
- `--bert_path`: BERT model path used by CheXbert metric code
- `--radgraph_path`: RadGraph checkpoint/model archive

Vision model:
- `--vit_ChexRel_path`: used by `CheXRelVisualExtracter` (pretraining) and related feature extractors

You almost certainly need to edit these defaults before running.

---

## Running

The entry is `main.py`:

### 1) Pretrain
```bash
python main.py \
  --task pretrain \
  --images_dir /path/to/mimic-cxr-jpg/2.0.0/files \
  --num_gpus 1 \
  --distilgpt2_path /path/to/ext/distilgpt2 \
  --cxr_bert_path /path/to/ext/BiomedVLP-CXR-BERT \
  --rad_dino_path microsoft/rad-dino \
  --vit_ChexRel_path /path/to/your/CheXRelFormer.ckpt
```

Common optional flags (from `parse_agrs()`):
- `--using_local_loss yes|no`
- `--using_diff_loss yes|no`
- `--is_multiview_fusion yes|no`
- `--temp`, `--region_temp`, `--diff_temp`

### 2) Finetune (conditional report generation)
```bash
python main.py \
  --task finetune \
  --num_gpus 1 \
  --images_dir /path/to/mimic-cxr-jpg/2.0.0/files \
  --distilgpt2_path /path/to/ext/distilgpt2 \
  --cxr_bert_path /path/to/ext/BiomedVLP-CXR-BERT \
  --rad_dino_path microsoft/rad-dino \
  --vit_ChexRel_path /path/to/your/CheXRelFormer.ckpt \
  --load /path/to/pretrain_checkpoint.pt \
  --resume /path/to/finetune_resume.ckpt
```

Notes:
- `--load` is used to initialize from a pre-trained checkpoint (selected weights are merged by shape).
- `--resume` continues training from a Lightning checkpoint.

### 3) Test (generate and evaluate)
```bash
python main.py \
  --task test \
  --num_gpus 1 \
  --images_dir /path/to/mimic-cxr-jpg/2.0.0/files \
  --test_ckpt_path /path/to/best_model.ckpt
```

`test_ckpt_path` is required for test mode.

### Multi-GPU (DDP)
`main.py` uses Lightning DDP strategy internally (`strategy="ddp_find_unused_parameters_true"`).
For `--num_gpus > 1`, launch with a distributed launcher (recommended example):
```bash
torchrun --standalone --nproc_per_node <N> main.py --num_gpus <N> --task finetune ...
```

---

## Outputs

`utils.parse_agrs()` creates an experiment folder:
`results/<data_name>/<task>/<version>_<timestamp>/`

Generated report logs are saved by `ReportLogger` to:
`<exp_dir>/generated_reports/`
as CSV files per epoch and split (e.g. `val_epoch-...csv`, `test_epoch-...csv`).

Checkpoints are saved by Lightning’s `ModelCheckpoint` callback to:
`<exp_dir>/checkpoint/`
and (when `--is_save_checkpoint yes`) uses:
- `save_last=True`
- best model tracked by `--monitor_metric` (defaults to `RCB` for finetune/test)

---

## Evaluation Metrics (Finetune/Test)

During `Finetune` validation/test, the model computes:
- COCO-style caption metrics (BLEU, CIDEr, ROUGE, METEOR) via `COCOCaptionMetrics`
- RadGraph metrics (F1-based variants)
- CheXbert metrics (F1-based variants)

The code additionally defines combined scores:
- `RB` = `F1-Radgraph-partial` + `chen_bleu_4`
- `RC` = `F1-Radgraph-partial` + `chexbert_all_micro_f1`
- `RCB` = `F1-Radgraph-partial` + `chen_bleu_4` + `chexbert_all_micro_f1`

`--monitor_metric` selects which metric is used to pick the best checkpoint (default: `RCB`).

---

## Notes / Caveats

1. Security: DeepSeek/OpenAI
   - `dataset/datasets.py` and `dataset/deepseek_api.py` contain example code that calls DeepSeek via `openai`.
   - Do not commit real API keys; configure your environment securely if you use these scripts.

2. Hardcoded dataset paths
   - `ReportDataset` uses the `temporal_datasets/with-indication/` JSONs directly.
   - If you want to plug in your own temporal dataset, update the JSON loading line(s) in `dataset/datasets.py`.

3. bf16 precision
   - `main.py` uses bf16 mixed precision. If you run on a GPU without bf16 support, you may need to change precision in `main.py`.

