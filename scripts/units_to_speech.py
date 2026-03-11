"""
Inference: Discrete Units → Speech for Vietnamese.

Given a sequence of discrete units (0-999), generate speech audio.

Input formats:
    - Text file: space-separated unit IDs (e.g., "50 120 340 120 50")
    - Text file with durations: tab-separated "unit\tduration" per line
    - JSON: {"units": [...], "durations": [...]}
    - PyTorch: .pt file with keys "unit" and optionally "duration"

Usage:
    # From a reference Vietnamese audio (extract units internally):
    python scripts/units_to_speech.py \
        --source_audio path/to/vietnamese.wav \
        --speaker_reference path/to/speaker_ref.wav \
        --output_path output.wav

    # From pre-extracted units file:
    python scripts/units_to_speech.py \
        --units_path units.txt \
        --speaker_reference path/to/speaker_ref.wav \
        --output_path output.wav

    # With fine-tuned model:
    python scripts/units_to_speech.py \
        --source_audio path/to/vietnamese.wav \
        --decoder_path outputs/train_unit2mel/best_decoder.pt \
        --speaker_reference path/to/speaker_ref.wav \
        --output_path output.wav
"""

import argparse
import json
import librosa
import os
import torch
import torchaudio
from scipy.io.wavfile import write

from unitspeech.unitspeech import UnitSpeech
from unitspeech.encoder import Encoder
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN_SMALL
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder
from unitspeech.util import HParams, fix_len_compatibility, process_unit, generate_path, sequence_mask
from unitspeech.vocoder.env import AttrDict
from unitspeech.vocoder.meldataset import mel_spectrogram
from unitspeech.vocoder.models import BigVGAN


def load_units_from_file(path):
    """Load units (and optionally durations) from various file formats."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.pt':
        data = torch.load(path, map_location='cpu', weights_only=False)
        units = data['unit'] if isinstance(data, dict) else data
        durations = data.get('duration', None) if isinstance(data, dict) else None
        return units, durations

    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        units = torch.LongTensor(data['units'])
        durations = torch.LongTensor(data['durations']) if 'durations' in data else None
        return units, durations

    # Plain text
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    lines = content.split('\n')
    if len(lines) == 1:
        # Single line, space or comma separated
        tokens = content.replace(',', ' ').split()
        units = torch.LongTensor([int(t) for t in tokens])
        return units, None
    else:
        # Multi-line: could be "unit\tduration" pairs
        units_list = []
        durations_list = []
        has_durations = '\t' in lines[0] or ',' in lines[0]
        for line in lines:
            parts = line.replace(',', '\t').split('\t')
            units_list.append(int(parts[0].strip()))
            if has_durations and len(parts) > 1:
                durations_list.append(int(parts[1].strip()))
        units = torch.LongTensor(units_list)
        durations = torch.LongTensor(durations_list) if durations_list else None
        return units, durations


def extract_units_from_audio(audio_path, unit_extractor, sampling_rate, hop_length):
    """Extract discrete units from audio file using mHuBERT."""
    wav, sr = librosa.load(audio_path, sr=None)
    wav = torch.FloatTensor(wav).unsqueeze(0)
    resample_fn = torchaudio.transforms.Resample(sr, 16000).cuda()
    wav_16k = resample_fn(wav.cuda())

    with torch.no_grad():
        encoded = unit_extractor(wav_16k)

    unit, duration = process_unit(encoded, sampling_rate, hop_length)
    return unit, duration


def extract_speaker_embedding(audio_path, spk_embedder):
    """Extract speaker embedding from reference audio."""
    wav, sr = librosa.load(audio_path, sr=None)
    wav = torch.FloatTensor(wav).unsqueeze(0)
    resample_fn = torchaudio.transforms.Resample(sr, 16000).cuda()
    wav_16k = resample_fn(wav.cuda())
    with torch.no_grad():
        spk_emb = spk_embedder(wav_16k)
        spk_emb = spk_emb / spk_emb.norm()
    return spk_emb


@torch.no_grad()
def synthesize(unit_encoder, decoder, vocoder, unit, duration, spk_emb,
               hps, diffusion_steps, text_gradient_scale, spk_gradient_scale):
    """Generate audio from units."""
    num_downsamplings = len(hps.decoder.dim_mults) - 1

    unit = unit.unsqueeze(0).cuda()
    unit_lengths = torch.LongTensor([unit.shape[-1]]).cuda()

    # Encode units
    cond_x, x, x_mask = unit_encoder(unit, unit_lengths)

    if duration is not None:
        duration = duration.unsqueeze(0).cuda()
        # Compute mel length from durations
        mel_length = duration.sum(-1).item()
    else:
        # Default: each unit = 1 mel frame
        mel_length = unit.shape[-1]
        duration = torch.ones_like(unit)

    mel_length = int(mel_length)
    mel_max_length = fix_len_compatibility(mel_length, num_downsamplings)

    # Build alignment from durations
    y_mask = sequence_mask(torch.LongTensor([mel_length]).cuda(), mel_max_length).unsqueeze(1).to(x_mask.dtype)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    attn = generate_path(duration, attn_mask.squeeze(1))

    # Align: expand cond_x to mel length
    cond_y = torch.matmul(
        attn.squeeze(1).transpose(1, 2).contiguous(),
        cond_x.transpose(1, 2).contiguous()
    ).transpose(1, 2).contiguous()

    # Pad to compatible length
    if cond_y.shape[-1] < mel_max_length:
        cond_y = torch.nn.functional.pad(cond_y, (0, mel_max_length - cond_y.shape[-1]))

    # Generate mel via reverse diffusion
    z = torch.randn_like(cond_y, device=cond_y.device)
    mel_out = decoder(
        z, y_mask, cond_y, spk_emb, diffusion_steps,
        text_gradient_scale=text_gradient_scale,
        spk_gradient_scale=spk_gradient_scale,
    )
    mel_out = mel_out[:, :, :mel_length]

    # Denormalize mel: from [-1, 1] range (need reference mel stats)
    # For now, use simple denorm that works with BigVGAN
    # The vocoder handles normalized mel directly
    audio = vocoder(mel_out)
    audio = audio.squeeze().cpu().numpy()
    return audio, mel_out


def main(args):
    with open(args.config_path, "r") as f:
        config = json.loads(f.read())
    hps = HParams(**config)

    # Initialize Unit Encoder
    print('Loading Unit Encoder...')
    unit_encoder = Encoder(
        n_vocab=hps.data.n_units,
        n_feats=hps.data.n_feats,
        **hps.encoder
    )
    encoder_dict = torch.load(args.encoder_path, map_location='cpu')
    unit_encoder.load_state_dict(encoder_dict['model'])
    _ = unit_encoder.cuda().eval()

    # Initialize Decoder
    print('Loading Decoder...')
    decoder = UnitSpeech(
        n_feats=hps.data.n_feats,
        **hps.decoder
    )
    decoder_dict = torch.load(args.decoder_path, map_location='cpu')
    decoder.load_state_dict(decoder_dict['model'])
    _ = decoder.cuda().eval()

    # Load speaker embedding from finetuned decoder if available
    spk_emb_from_ckpt = decoder_dict.get('spk_emb', None)
    mel_min = decoder_dict.get('mel_min', None)
    mel_max = decoder_dict.get('mel_max', None)

    # Initialize Vocoder
    print('Loading Vocoder...')
    with open(hps.train.vocoder_config_path) as f:
        h = AttrDict(json.load(f))
    vocoder = BigVGAN(h)
    vocoder.load_state_dict(
        torch.load(hps.train.vocoder_ckpt_path, map_location='cpu')['generator']
    )
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    # Get units
    unit, duration = None, None

    if args.units_path:
        print(f'Loading units from {args.units_path}...')
        unit, duration = load_units_from_file(args.units_path)
    elif args.source_audio:
        print(f'Extracting units from {args.source_audio}...')
        print('  Initializing Unit Extractor (mHuBERT)...')
        unit_extractor = SpeechEncoder.by_name(
            dense_model_name="mhubert-base-vp_en_es_fr",
            quantizer_model_name="kmeans",
            vocab_size=1000,
            deduplicate=True,
            need_f0=False
        )
        _ = unit_extractor.cuda().eval()
        unit, duration = extract_units_from_audio(
            args.source_audio, unit_extractor,
            hps.data.sampling_rate, hps.data.hop_length
        )
    else:
        raise ValueError("Must provide either --units_path or --source_audio")

    print(f'  Units: {unit.shape[0]} tokens, range [{unit.min()}, {unit.max()}]')
    if duration is not None:
        print(f'  Durations: sum={duration.sum().item()} mel frames')

    # Get speaker embedding
    if args.speaker_reference:
        print(f'Extracting speaker embedding from {args.speaker_reference}...')
        spk_embedder = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        state_dict = torch.load(args.speaker_encoder_path, map_location='cpu')
        spk_embedder.load_state_dict(state_dict['model'], strict=False)
        _ = spk_embedder.cuda().eval()
        spk_emb = extract_speaker_embedding(args.speaker_reference, spk_embedder)
        spk_emb = spk_emb.unsqueeze(1)
    elif args.speaker_embedding_path:
        print(f'Loading speaker embedding from {args.speaker_embedding_path}...')
        spk_emb = torch.load(args.speaker_embedding_path, map_location='cpu')
        if spk_emb.dim() == 1:
            spk_emb = spk_emb.unsqueeze(0).unsqueeze(0)
        elif spk_emb.dim() == 2:
            spk_emb = spk_emb.unsqueeze(1)
        spk_emb = spk_emb.cuda()
    elif spk_emb_from_ckpt is not None:
        print('Using speaker embedding from decoder checkpoint...')
        spk_emb = spk_emb_from_ckpt.cuda()
        if spk_emb.dim() == 2:
            spk_emb = spk_emb.unsqueeze(1)
    else:
        raise ValueError("Must provide --speaker_reference, --speaker_embedding_path, "
                         "or use a finetuned decoder that contains spk_emb")

    # Synthesize
    print('Generating audio...')
    audio, mel_out = synthesize(
        unit_encoder, decoder, vocoder,
        unit, duration, spk_emb, hps,
        args.diffusion_steps,
        args.text_gradient_scale,
        args.spk_gradient_scale,
    )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    write(args.output_path, hps.data.sampling_rate, audio)
    print(f'Audio saved to: {args.output_path}')
    print(f'  Duration: {len(audio) / hps.data.sampling_rate:.2f}s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Units → Speech synthesis")

    # Input: units
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--units_path', type=str,
                             help='Path to units file (.txt, .json, .pt)')
    input_group.add_argument('--source_audio', type=str,
                             help='Path to source audio (units will be extracted)')

    # Input: speaker
    spk_group = parser.add_mutually_exclusive_group()
    spk_group.add_argument('--speaker_reference', type=str,
                           help='Path to speaker reference audio for embedding extraction')
    spk_group.add_argument('--speaker_embedding_path', type=str,
                           help='Path to pre-computed speaker embedding .pt file')

    # Model paths
    parser.add_argument('--encoder_path', type=str, default='unitspeech/checkpoints/unit_encoder.pt')
    parser.add_argument('--decoder_path', type=str, default='unitspeech/checkpoints/pretrained_decoder.pt')
    parser.add_argument('--speaker_encoder_path', type=str,
                        default='unitspeech/speaker_encoder/checkpts/speaker_encoder.pt')
    parser.add_argument('--config_path', type=str,
                        default='unitspeech/checkpoints/finetune.json')

    # Output
    parser.add_argument('--output_path', type=str, default='outputs/synthesized.wav')

    # Synthesis params
    parser.add_argument('--diffusion_steps', type=int, default=50,
                        help='Number of diffusion steps (more = better quality, slower)')
    parser.add_argument('--text_gradient_scale', type=float, default=0.0,
                        help='Classifier-free guidance scale for conditioning')
    parser.add_argument('--spk_gradient_scale', type=float, default=0.0,
                        help='Classifier-free guidance scale for speaker')

    args = parser.parse_args()
    main(args)
