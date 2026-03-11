"""
Train units → mel mapper on Vietnamese dataset.

This script trains the Unit Encoder + Diffusion Decoder on preprocessed data
from prepare_dataset.py. It extends the single-file finetune.py into a proper
multi-file, multi-speaker training loop.

3 training modes:
    --mode decoder_only   : Freeze encoder, train decoder only (fast, needs less data)
    --mode encoder_only   : Freeze decoder, train encoder only (learn new unit→feature mapping)
    --mode full           : Train both encoder + decoder (best quality, needs most data)

Usage:
    python scripts/train_unit2mel.py \
        --data_dir data/preprocessed/ \
        --output_dir outputs/vietnamese_model/ \
        --mode full \
        --epochs 100 \
        --batch_size 8
"""

import argparse
import json
import os
import random
import torch
import torch.utils.data
from pathlib import Path
from tqdm import tqdm

from unitspeech.unitspeech import UnitSpeech
from unitspeech.encoder import Encoder
from unitspeech.util import (
    HParams, fix_len_compatibility, generate_path, sequence_mask
)


class UnitMelDataset(torch.utils.data.Dataset):
    """Dataset that loads preprocessed .pt files from prepare_dataset.py"""

    def __init__(self, data_dir, manifest_path=None):
        self.data_dir = data_dir
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            self.files = [item['file'] for item in manifest]
        else:
            self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

        self.files = sorted(self.files)
        print(f"Dataset: {len(self.files)} utterances")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(
            os.path.join(self.data_dir, self.files[idx]),
            map_location='cpu',
            weights_only=False,
        )
        return {
            'unit': data['unit'],              # [T_unit]
            'duration': data['duration'],       # [T_unit]
            'mel': data['mel'],                 # [80, T_mel]
            'spk_emb': data['spk_emb'],         # [256]
        }


def collate_fn(batch):
    """Collate variable-length sequences into padded batch tensors."""
    # Sort by mel length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: x['mel'].shape[-1], reverse=True)

    unit_lengths = torch.LongTensor([item['unit'].shape[0] for item in batch])
    mel_lengths = torch.LongTensor([item['mel'].shape[-1] for item in batch])

    max_unit_len = unit_lengths.max().item()
    max_mel_len = mel_lengths.max().item()
    n_feats = batch[0]['mel'].shape[0]

    # Pad units
    units_padded = torch.zeros(len(batch), max_unit_len, dtype=torch.long)
    durations_padded = torch.zeros(len(batch), max_unit_len, dtype=torch.long)
    mels_padded = torch.zeros(len(batch), n_feats, max_mel_len)
    spk_embs = torch.stack([item['spk_emb'] for item in batch])

    for i, item in enumerate(batch):
        u_len = item['unit'].shape[0]
        m_len = item['mel'].shape[-1]
        units_padded[i, :u_len] = item['unit']
        durations_padded[i, :u_len] = item['duration']
        mels_padded[i, :, :m_len] = item['mel']

    return {
        'unit': units_padded,
        'duration': durations_padded,
        'mel': mels_padded,
        'spk_emb': spk_embs,
        'unit_lengths': unit_lengths,
        'mel_lengths': mel_lengths,
    }


def compute_training_loss(
    unit_encoder, decoder, batch, segment_size, n_feats, num_downsamplings,
    train_encoder=False
):
    """Compute diffusion loss for one batch."""
    units = batch['unit'].cuda()
    durations = batch['duration'].cuda()
    mel = batch['mel'].cuda()
    spk_emb = batch['spk_emb'].cuda().unsqueeze(1)
    unit_lengths = batch['unit_lengths'].cuda()
    mel_lengths = batch['mel_lengths'].cuda()

    # Encode units → conditional features
    if train_encoder:
        cond_x, x, x_mask = unit_encoder(units, unit_lengths)
    else:
        with torch.no_grad():
            cond_x, x, x_mask = unit_encoder(units, unit_lengths)

    # Build alignment from durations
    mel_max_length = mel.shape[-1]
    mel_mask = sequence_mask(mel_lengths, mel_max_length).unsqueeze(1).to(x_mask)
    attn_mask = x_mask.unsqueeze(-1) * mel_mask.unsqueeze(2)

    # Adjust durations to match mel length per sample
    for i in range(durations.shape[0]):
        total_dur = durations[i, :unit_lengths[i]].sum()
        mel_len = mel_lengths[i]
        if total_dur != mel_len:
            diff = mel_len - total_dur
            # Distribute difference to the last valid unit
            last_idx = unit_lengths[i] - 1
            durations[i, last_idx] = max(1, durations[i, last_idx] + diff)

    attn = generate_path(durations, attn_mask.squeeze(1))

    # Random segment for training efficiency (same as finetune.py)
    if mel_max_length > segment_size:
        # Cut random segment
        max_offset = (mel_lengths - segment_size).clamp(0)
        offsets = torch.LongTensor([
            random.randint(0, max(0, off.item())) for off in max_offset
        ]).to(mel.device)

        attn_cut = torch.zeros(attn.shape[0], attn.shape[1], segment_size,
                               dtype=attn.dtype, device=attn.device)
        mel_cut = torch.zeros(mel.shape[0], n_feats, segment_size,
                              dtype=mel.dtype, device=mel.device)
        mel_cut_lengths = []

        for i in range(mel.shape[0]):
            cut_len = min(segment_size, mel_lengths[i].item())
            mel_cut_lengths.append(cut_len)
            lo, hi = offsets[i].item(), offsets[i].item() + cut_len
            mel_cut[i, :, :cut_len] = mel[i, :, lo:hi]
            attn_cut[i, :, :cut_len] = attn[i, :, lo:hi]

        mel_cut_lengths = torch.LongTensor(mel_cut_lengths).to(mel.device)
        mel_cut_mask = sequence_mask(mel_cut_lengths, segment_size).unsqueeze(1).to(mel.dtype)
        attn = attn_cut
        mel = mel_cut
        mel_mask = mel_cut_mask
    else:
        # Pad to segment_size if shorter
        if mel_max_length < segment_size:
            pad_size = segment_size - mel_max_length
            mel = torch.nn.functional.pad(mel, (0, pad_size))
            mel_mask = torch.nn.functional.pad(mel_mask, (0, pad_size))
            attn = torch.nn.functional.pad(attn, (0, pad_size))

    # Align encoded text with mel-spectrogram
    cond_y = torch.matmul(
        attn.squeeze(1).transpose(1, 2).contiguous(),
        cond_x.transpose(1, 2).contiguous()
    ).transpose(1, 2).contiguous()
    cond_y = cond_y * mel_mask

    # Compute diffusion loss
    diff_loss, _ = decoder.compute_loss(mel, mel_mask, cond_y, spk_emb=spk_emb)
    return diff_loss


def main(args):
    with open(args.config_path, "r") as f:
        config = json.loads(f.read())
    hps = HParams(**config)

    num_downsamplings = len(hps.decoder.dim_mults) - 1
    segment_size = fix_len_compatibility(
        hps.train.out_size_second * hps.data.sampling_rate // hps.data.hop_length,
        num_downsamplings
    )
    n_feats = hps.data.n_feats
    num_units = hps.data.n_units

    # Initialize models
    print('Initializing Unit Encoder...')
    unit_encoder = Encoder(
        n_vocab=num_units,
        n_feats=n_feats,
        **hps.encoder
    )
    encoder_dict = torch.load(args.encoder_path, map_location='cpu')
    unit_encoder.load_state_dict(encoder_dict['model'])
    unit_encoder = unit_encoder.cuda()

    print('Initializing Decoder...')
    decoder = UnitSpeech(
        n_feats=n_feats,
        **hps.decoder
    )
    decoder_dict = torch.load(args.decoder_path, map_location='cpu')
    decoder.load_state_dict(decoder_dict['model'])
    decoder = decoder.cuda()

    # Set training mode based on --mode
    train_encoder = args.mode in ('full', 'encoder_only')
    train_decoder = args.mode in ('full', 'decoder_only')

    if train_encoder:
        unit_encoder.train()
    else:
        unit_encoder.eval()

    if train_decoder:
        decoder.train()
    else:
        decoder.eval()

    # Collect trainable parameters
    params = []
    if train_encoder:
        params += list(unit_encoder.parameters())
        print(f"  Encoder params: {sum(p.numel() for p in unit_encoder.parameters()):,}")
    if train_decoder:
        params += list(decoder.parameters())
        print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    if not params:
        raise ValueError("No trainable parameters. Check --mode argument.")

    print(f"  Total trainable params: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # Dataset and DataLoader
    manifest_path = os.path.join(args.data_dir, "manifest.json")
    dataset = UnitMelDataset(args.data_dir, manifest_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    global_step = 0
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.fp16):
                loss = compute_training_loss(
                    unit_encoder, decoder, batch, segment_size, n_feats,
                    num_downsamplings, train_encoder=train_encoder
                )

            if args.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            # Log periodically
            if global_step % args.log_interval == 0:
                avg = epoch_loss / n_batches
                print(f"  Step {global_step}: avg_loss={avg:.4f}")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch} done. Avg loss: {avg_epoch_loss:.4f}")

        scheduler.step()

        # Save checkpoint
        if epoch % args.save_interval == 0 or avg_epoch_loss < best_loss:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss

            ckpt = {
                'epoch': epoch,
                'global_step': global_step,
                'loss': avg_epoch_loss,
                'mode': args.mode,
            }

            if train_encoder:
                encoder_path = os.path.join(args.output_dir, f"unit_encoder_ep{epoch}.pt")
                torch.save({'model': unit_encoder.state_dict()}, encoder_path)
                ckpt['encoder_path'] = encoder_path

            if train_decoder:
                decoder_path = os.path.join(args.output_dir, f"decoder_ep{epoch}.pt")
                torch.save({'model': decoder.state_dict()}, decoder_path)
                ckpt['decoder_path'] = decoder_path

            # Save best
            if avg_epoch_loss <= best_loss:
                if train_encoder:
                    torch.save({'model': unit_encoder.state_dict()},
                               os.path.join(args.output_dir, "best_unit_encoder.pt"))
                if train_decoder:
                    torch.save({'model': decoder.state_dict()},
                               os.path.join(args.output_dir, "best_decoder.pt"))

            meta_path = os.path.join(args.output_dir, f"checkpoint_ep{epoch}.json")
            with open(meta_path, "w") as f:
                json.dump(ckpt, f, indent=2)

            print(f"  Saved checkpoint at epoch {epoch}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Models saved in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train units → mel mapper")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with preprocessed .pt files from prepare_dataset.py')
    parser.add_argument('--output_dir', type=str, default='outputs/train_unit2mel/',
                        help='Directory to save checkpoints')
    parser.add_argument('--config_path', type=str, default='unitspeech/checkpoints/finetune.json')
    parser.add_argument('--encoder_path', type=str, default='unitspeech/checkpoints/unit_encoder.pt',
                        help='Path to pretrained unit encoder (init weights)')
    parser.add_argument('--decoder_path', type=str, default='unitspeech/checkpoints/pretrained_decoder.pt',
                        help='Path to pretrained decoder (init weights)')

    # Training mode
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'decoder_only', 'encoder_only'],
                        help='What to train: full (both), decoder_only, or encoder_only')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.999,
                        help='Exponential LR decay per epoch')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Print loss every N steps')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()
    main(args)
