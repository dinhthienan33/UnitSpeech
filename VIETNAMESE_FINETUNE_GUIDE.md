# UnitSpeech — Vietnamese Fine-tuning Guide

## 1. Project Overview

**UnitSpeech** (INTERSPEECH 2023, Oral) is a speaker-adaptive speech synthesis system that uses discrete speech units as intermediate representation. The core pipeline:

```
Discrete Units (0-999) → Encoder → Conditional Features → Diffusion Decoder → Mel-spectrogram → BigVGAN → Audio
```

Key components:
| Component | Role | Language-dependent? |
|-----------|------|-------------------|
| **mHuBERT** (unit extractor) | Audio → discrete tokens (K=1000) | ❌ Multilingual (en, es, fr) |
| **Unit Encoder** | Unit ID → conditional features | ❌ Just embedding lookup |
| **Decoder** (Diffusion U-Net) | Features → mel-spectrogram | ❌ Acoustic model |
| **Speaker Encoder** (ECAPA-TDNN/WavLM) | Audio → speaker embedding (256-dim) | ❌ Speaker identity only |
| **BigVGAN** (vocoder) | Mel → waveform | ❌ Signal processing |
| **Text Encoder** | Phonemes → features (TTS only) | ✅ English IPA phonemizer |
| **Duration Predictor** | Predict phoneme durations (TTS only) | ✅ Trained on English |

## 2. Can We Use This for Vietnamese?

### 2.1 Voice Conversion: ✅ Works immediately
- No text processing involved
- ContentVec extracts language-agnostic features
- Just needs source audio + speaker reference

### 2.2 Unit → Speech (the main focus): ✅ Works, fine-tuning recommended
- All components in the units→speech path are language-agnostic
- mHuBERT is already multilingual — works on Vietnamese audio
- Fine-tuning the decoder on Vietnamese data improves quality

### 2.3 Text → Speech: ⚠️ Requires significant modification
- Phonemizer is hardcoded to `en-us`
- Text cleaners are English-specific
- Text encoder + duration predictor trained on English only
- Would need: Vietnamese phonemizer, retrained text encoder + duration predictor

## 3. Fine-tuning Does NOT Require Vietnamese-specific Models

The pretrained models work because:
- **mHuBERT** extracts acoustic features (not linguistic) — multilingual by design
- **Unit Encoder** maps integer IDs (0-999) to embeddings — no language info
- **Decoder** generates mel from features — purely acoustic
- **BigVGAN** converts mel to waveform — purely signal processing

Fine-tuning is recommended (not required) to:
- Adapt the decoder to Vietnamese acoustic characteristics (tones, phoneme inventory)
- Improve speaker similarity for Vietnamese speakers

## 4. Required Pretrained Models

All from the same [Google Drive folder](https://drive.google.com/drive/folders/1yFkb2TAYB_zMmoTuUOXu-zXb3UI9pVJ9?usp=sharing):

| File | Purpose |
|------|---------|
| `unit_encoder.pt` | Unit → feature encoding |
| `pretrained_decoder.pt` | Diffusion mel generation |
| `speaker_encoder.pt` | Speaker embedding extraction |
| `bigvgan.pt` + `bigvgan-config.json` | Vocoder |

Base model HuggingFace links:
- mHuBERT: auto-downloaded via fairseq/textlesslib
- WavLM (speaker encoder base): [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large)
- BigVGAN: [nvidia/bigvgan](https://huggingface.co/nvidia/bigvgan)
- ContentVec (for VC only): [lengyue233/content-vec-best](https://huggingface.co/lengyue233/content-vec-best)

## 5. UnitSpeech vs HiFi-GAN Comparison

| Aspect | **UnitSpeech** | **HiFi-GAN** |
|--------|---------------|-------------|
| **Role** | Full synthesis system (units → audio) | Vocoder only (mel → audio) |
| **Input** | Discrete units (0-999) or text | Mel-spectrogram |
| **Decoder** | Score-based Diffusion (U-Net) | GAN (Generator + Discriminator) |
| **Generation** | Iterative denoising (~50-1000 steps) | Single forward pass |
| **Speed** | Slower | Real-time capable |
| **Speaker adaptation** | Built-in (fine-tune + speaker embedding) | Not built-in |
| **Relationship** | **Contains BigVGAN (HiFi-GAN variant) as its vocoder** | Is the vocoder itself |

BigVGAN (used inside UnitSpeech) is an improved HiFi-GAN with anti-aliased Snake activations.

## 6. Training a Units → Mel Mapper

### 6.1 Three Training Modes

| Mode | What's trained | What's frozen | Use case |
|------|---------------|---------------|----------|
| `decoder_only` | Decoder (diffusion U-Net, ~20M params) | Encoder | Safe first choice. Improves audio quality, speaker similarity |
| `encoder_only` | Encoder (Embedding + Transformer, ~5M params) | Decoder | When pronunciation is wrong. Learns better unit→feature mapping |
| `full` | Both Encoder + Decoder | Nothing | Best quality. Needs most data |

### 6.2 `decoder_only` vs `encoder_only` Explained

**`decoder_only`**: Encoder feature mapping stays the same. Decoder learns to generate better mel-spectrograms for Vietnamese acoustic space. Lower risk, good for adapting audio quality and prosody.

**`encoder_only`**: Decoder generation stays the same. Encoder learns to produce features that better represent Vietnamese phonetics (tones, phoneme boundaries). Risk: features may drift from what decoder expects.

**Recommendation**: Try `decoder_only` first → if pronunciation issues persist → try `encoder_only` → if still not enough → use `full`.

## 7. Implementation — Scripts Created

### 7.1 Data Preparation (`scripts/prepare_dataset.py`)

Converts raw audio directory into preprocessed `.pt` files:
```bash
python scripts/prepare_dataset.py \
    --audio_dir data/vietnamese_audio/ \
    --output_dir data/preprocessed/
```

Each `.pt` file contains: `unit`, `duration`, `mel`, `spk_emb`, `mel_min`, `mel_max`, `speaker_id`

Supports nested speaker directories:
```
data/vietnamese_audio/
    speaker_01/
        001.wav
        002.wav
    speaker_02/
        001.wav
```

### 7.2 Training (`scripts/train_unit2mel.py`)

Multi-file, multi-speaker training loop with 3 modes:
```bash
# Full training (best quality)
python scripts/train_unit2mel.py \
    --data_dir data/preprocessed/ \
    --output_dir outputs/vi_model/ \
    --mode full --epochs 100 --batch_size 8

# Decoder only (fast, less data)
python scripts/train_unit2mel.py --mode decoder_only ...

# Encoder only (fix pronunciation)
python scripts/train_unit2mel.py --mode encoder_only ...
```

Features: batched training, random segment cropping, gradient clipping, FP16 support, periodic checkpointing, best model saving.

### 7.3 Inference (`scripts/units_to_speech.py`)

Generate audio from units or source audio:
```bash
# From source audio (extract units internally)
python scripts/units_to_speech.py \
    --source_audio test_vi.wav \
    --speaker_reference speaker_ref.wav \
    --output_path output.wav

# From pre-extracted units file
python scripts/units_to_speech.py \
    --units_path units.txt \
    --speaker_reference speaker_ref.wav \
    --output_path output.wav

# With fine-tuned model
python scripts/units_to_speech.py \
    --source_audio test_vi.wav \
    --decoder_path outputs/vi_model/best_decoder.pt \
    --speaker_reference speaker_ref.wav \
    --output_path output.wav
```

Supports input formats: `.txt` (space-separated), `.json`, `.pt`, tab-separated with durations.

### 7.4 Config (`unitspeech/checkpoints/train-vietnamese.json`)

Same architecture as original (identical to `finetune.json`). No Vietnamese-specific config changes needed since the pipeline is language-agnostic at the unit level.

## 8. Recommended Vietnamese Datasets

- **VIVOS** (~15h, multi-speaker) — good starting point
- **VietTTS / VLSP** — if you have access
- Any clean Vietnamese audio ≥ 1 second, WAV/FLAC format

## 9. Quick Start Summary

```bash
# 1. Install
pip install -e .
pip install --no-deps s3prl==0.4.10

# 2. Download pretrained models to correct paths (see Section 4)

# 3. Prepare Vietnamese data
python scripts/prepare_dataset.py \
    --audio_dir data/vietnamese_audio/ \
    --output_dir data/preprocessed/

# 4. Train
python scripts/train_unit2mel.py \
    --data_dir data/preprocessed/ \
    --output_dir outputs/vi_model/ \
    --mode decoder_only --epochs 50

# 5. Inference
python scripts/units_to_speech.py \
    --source_audio test_vi.wav \
    --decoder_path outputs/vi_model/best_decoder.pt \
    --speaker_reference speaker_ref.wav \
    --output_path output.wav
```
