"""
Prepare Vietnamese audio dataset for training units → mel mapper.

Input: A directory of .wav files (can have subdirectories per speaker).
Output: A directory of preprocessed .pt files containing:
    - unit: discrete unit sequence [T_unit]
    - duration: duration for each unit [T_unit]
    - mel: normalized mel-spectrogram [80, T_mel]  
    - spk_emb: speaker embedding [256]
    - mel_min, mel_max: normalization params

Usage:
    python scripts/prepare_dataset.py \
        --audio_dir data/vietnamese_audio/ \
        --output_dir data/preprocessed/ \
        --config_path unitspeech/checkpoints/finetune.json
"""

import argparse
import json
import glob
import librosa
import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN_SMALL
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder
from unitspeech.util import HParams, process_unit
from unitspeech.vocoder.meldataset import mel_spectrogram


def main(args):
    with open(args.config_path, "r") as f:
        config = json.loads(f.read())
    hps = HParams(**config)

    # Initialize models
    print('Initializing Speaker Encoder...')
    spk_embedder = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(args.speaker_encoder_path, map_location="cpu")
    spk_embedder.load_state_dict(state_dict['model'], strict=False)
    _ = spk_embedder.cuda().eval()

    print('Initializing Unit Extractor...')
    unit_extractor = SpeechEncoder.by_name(
        dense_model_name="mhubert-base-vp_en_es_fr",
        quantizer_model_name="kmeans",
        vocab_size=1000,
        deduplicate=True,
        need_f0=False
    )
    _ = unit_extractor.cuda().eval()

    resample_fn = torchaudio.transforms.Resample(hps.data.sampling_rate, 16000).cuda()

    # Find all audio files
    audio_extensions = ['*.wav', '*.flac', '*.mp3', '*.ogg']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, '**', ext), recursive=True))
    
    print(f'Found {len(audio_files)} audio files.')

    os.makedirs(args.output_dir, exist_ok=True)

    skipped = 0
    processed = 0
    manifest = []

    for audio_path in tqdm(audio_files, desc="Processing"):
        try:
            # Load audio
            wav, sr = librosa.load(audio_path, sr=hps.data.sampling_rate)
            
            # Skip very short audio (< 1 second) or very long (> 30 seconds)
            duration_sec = len(wav) / hps.data.sampling_rate
            if duration_sec < args.min_duration or duration_sec > args.max_duration:
                skipped += 1
                continue

            wav_tensor = torch.FloatTensor(wav).unsqueeze(0)

            # Extract mel-spectrogram
            mel = mel_spectrogram(
                wav_tensor, hps.data.n_fft, hps.data.n_feats,
                hps.data.sampling_rate, hps.data.hop_length,
                hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax,
                center=False
            )

            # Normalize mel
            mel_min = mel.min(-1, keepdim=True)[0]
            mel_max = mel.max(-1, keepdim=True)[0]
            mel_norm = (mel - mel_min) / (mel_max - mel_min + 1e-8) * 2 - 1

            # Extract speaker embedding
            wav_16k = resample_fn(wav_tensor.cuda())
            with torch.no_grad():
                spk_emb = spk_embedder(wav_16k)
                spk_emb = spk_emb / spk_emb.norm()

            # Extract units and durations
            with torch.no_grad():
                encoded = unit_extractor(wav_16k)
            unit, unit_duration = process_unit(encoded, hps.data.sampling_rate, hps.data.hop_length)

            # Verify alignment: total duration should roughly match mel length
            total_unit_frames = unit_duration.sum().item()
            mel_frames = mel_norm.shape[-1]
            if abs(total_unit_frames - mel_frames) > 5:
                # Adjust last duration to match
                diff = mel_frames - (unit_duration.sum().item() - unit_duration[-1].item())
                if diff > 0:
                    unit_duration[-1] = diff
                else:
                    skipped += 1
                    continue

            # Derive speaker ID from parent directory name
            rel_path = os.path.relpath(audio_path, args.audio_dir)
            speaker_id = Path(rel_path).parts[0] if len(Path(rel_path).parts) > 1 else "default"

            # Save preprocessed data
            basename = Path(audio_path).stem
            output_name = f"{speaker_id}_{basename}.pt"
            output_path = os.path.join(args.output_dir, output_name)

            torch.save({
                'unit': unit.cpu(),
                'duration': unit_duration.cpu(),
                'mel': mel_norm.squeeze(0).cpu(),
                'spk_emb': spk_emb.squeeze(0).cpu(),
                'mel_min': mel_min.squeeze(0).cpu(),
                'mel_max': mel_max.squeeze(0).cpu(),
                'speaker_id': speaker_id,
                'audio_path': audio_path,
            }, output_path)

            manifest.append({
                'file': output_name,
                'speaker': speaker_id,
                'n_units': unit.shape[0],
                'mel_frames': mel_frames,
                'duration_sec': duration_sec,
            })
            processed += 1

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            skipped += 1
            continue

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Vietnamese dataset for unit→mel training")
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files (can have speaker subdirectories)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save preprocessed .pt files')
    parser.add_argument('--config_path', type=str, default="unitspeech/checkpoints/finetune.json")
    parser.add_argument('--speaker_encoder_path', type=str,
                        default="unitspeech/speaker_encoder/checkpts/speaker_encoder.pt")
    parser.add_argument('--min_duration', type=float, default=1.0,
                        help='Minimum audio duration in seconds')
    parser.add_argument('--max_duration', type=float, default=30.0,
                        help='Maximum audio duration in seconds')
    args = parser.parse_args()
    main(args)
