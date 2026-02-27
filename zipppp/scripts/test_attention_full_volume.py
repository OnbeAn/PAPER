"""
Full-volume inference for nnUNetMultiTaskWithAttention models.
Features:
- Sliding window inference with optional TTA (mirror flips on specified axes)
- Optional automatic LCC decision (evaluate Dice with/without LCC and save decision)
- Save predictions and (if GT provided) metrics

Example:
python scripts/test_attention_full_volume.py \
  --experiment_dir /home/agr/DUOFENZHI_gujia/experiments/seg_dist_attention_v2_repro3 \
  --image_dir /home/agr/DUOFENZHI_gujia/data/processed/test_images \
  --seg_dir /home/agr/DUOFENZHI_gujia/data/processed/test_labels_seg \
  --dist_dir /home/agr/DUOFENZHI_gujia/data/processed/test_labels_dist \
  --patch_size 128 128 128 --stride 64 64 64 \
  --tta --mirror_axes 0 1 2 --auto_lcc_decide
"""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import nibabel as nib
from tqdm import tqdm

# project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yaml
from models.nnunet_multitask_attention import nnUNetMultiTaskWithAttention
from data.direction_codec import DirectionCodec

try:
    from scipy.ndimage import label as cc_label
    from scipy.ndimage import generate_binary_structure
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def normalize_image(image: np.ndarray) -> np.ndarray:
    p_low = np.percentile(image, 0.5)
    p_high = np.percentile(image, 99.5)
    image = np.clip(image, p_low, p_high)
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std
    return image


def flip_axes_to_dims_for_tensor(axes: tuple, base_dim_offset: int) -> tuple:
    """Map spatial axes (0,1,2) to tensor dims by adding base offset.
    For patch tensor (B,C,D,H,W) base_dim_offset=2 -> (2,3,4)
    For seg/flow logits (C,D,H,W) base_dim_offset=1 -> (1,2,3)
    For dist (D,H,W) base_dim_offset=0 -> (0,1,2)
    """
    if not axes:
        return tuple()
    return tuple(int(a) + base_dim_offset for a in axes)


def sliding_window_positions(D, H, W, pD, pH, pW, sD, sH, sW):
    d_starts = list(range(0, max(1, D - pD + 1), sD))
    h_starts = list(range(0, max(1, H - pH + 1), sH))
    w_starts = list(range(0, max(1, W - pW + 1), sW))
    if (len(d_starts) == 0) or (d_starts[-1] + pD < D):
        d_starts.append(max(0, D - pD))
    if (len(h_starts) == 0) or (h_starts[-1] + pH < H):
        h_starts.append(max(0, H - pH))
    if (len(w_starts) == 0) or (w_starts[-1] + pW < W):
        w_starts.append(max(0, W - pW))
    return d_starts, h_starts, w_starts


def run_model_single(model, patch_t: torch.Tensor):
    """Run model on a single (1,1,D,H,W) patch tensor on device, return torch tensors.
    Returns seg_logits (C,D,H,W), dist_pred (D,H,W), flow_logits (C,D,H,W)
    """
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            pred = model(patch_t)
        seg_logits = pred['seg'][0]          # (C, d,h,w)
        dist_pred = pred['dist'][0, 0]       # (d,h,w)
        flow_logits = pred['flow'][0]        # (C, d,h,w)
    return seg_logits, dist_pred, flow_logits


def sliding_window_inference(image, model, patch_size, stride, device, num_seg_classes=2, num_flow_classes=38, tta=False, mirror_axes=(0,1,2)):
    D, H, W = image.shape
    pD, pH, pW = patch_size
    sD, sH, sW = stride
    seg_sum = np.zeros((num_seg_classes, D, H, W), dtype=np.float32)
    dist_sum = np.zeros((D, H, W), dtype=np.float32)
    flow_sum = np.zeros((num_flow_classes, D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    d_starts, h_starts, w_starts = sliding_window_positions(D, H, W, pD, pH, pW, sD, sH, sW)
    positions = [(ds, hs, ws) for ds in d_starts for hs in h_starts for ws in w_starts]

    tta_flip_sets = [()]  # no flip
    if tta:
        # generate all non-empty combinations of specified axes
        axes = tuple(sorted(set(int(a) for a in mirror_axes)))
        tta_flip_sets = [()]  # include identity
        # add single flips
        for a in axes:
            tta_flip_sets.append((a,))
        # add double flips
        if len(axes) >= 2:
            for i in range(len(axes)):
                for j in range(i+1, len(axes)):
                    tta_flip_sets.append((axes[i], axes[j]))
        # add triple flips
        if len(axes) == 3:
            tta_flip_sets.append((axes[0], axes[1], axes[2]))

    pbar = tqdm(total=len(positions), desc="Sliding window inference")
    for (ds, hs, ws) in positions:
        patch = image[ds:ds+pD, hs:hs+pH, ws:ws+pW]
        seg_accum = np.zeros((num_seg_classes, pD, pH, pW), dtype=np.float32)
        dist_accum = np.zeros((pD, pH, pW), dtype=np.float32)
        flow_accum = np.zeros((num_flow_classes, pD, pH, pW), dtype=np.float32)
        n_augs = 0
        # Move base patch once to GPU (contiguous copy) and reuse for all TTA
        patch_t = torch.from_numpy(patch.copy()).float().unsqueeze(0).unsqueeze(0)
        # Using pinned memory can speed up H2D copy
        if torch.cuda.is_available():
            try:
                patch_t = patch_t.pin_memory()
            except Exception:
                pass
        patch_t = patch_t.to(device, non_blocking=True)

        for flips in tta_flip_sets:
            if len(flips) > 0:
                dims_in = flip_axes_to_dims_for_tensor(flips, base_dim_offset=2)   # for (B,C,D,H,W)
                patch_aug_t = torch.flip(patch_t, dims=dims_in)
            else:
                patch_aug_t = patch_t

            seg_logits_t, dist_pred_t, flow_logits_t = run_model_single(model, patch_aug_t)
            # invert flips on outputs back to original orientation
            if len(flips) > 0:
                dims_seg = flip_axes_to_dims_for_tensor(flips, base_dim_offset=1)  # (C,D,H,W)
                dims_dist = flip_axes_to_dims_for_tensor(flips, base_dim_offset=0) # (D,H,W)
                seg_logits_t = torch.flip(seg_logits_t, dims=dims_seg)
                dist_pred_t = torch.flip(dist_pred_t, dims=dims_dist)
                flow_logits_t = torch.flip(flow_logits_t, dims=dims_seg)

            seg_accum += seg_logits_t.detach().cpu().numpy()
            dist_accum += dist_pred_t.detach().cpu().numpy()
            flow_accum += flow_logits_t.detach().cpu().numpy()
            n_augs += 1
        seg_accum /= max(1, n_augs)
        dist_accum /= max(1, n_augs)
        flow_accum /= max(1, n_augs)

        seg_sum[:, ds:ds+pD, hs:hs+pH, ws:ws+pW] += seg_accum
        dist_sum[ds:ds+pD, hs:hs+pH, ws:ws+pW] += dist_accum
        flow_sum[:, ds:ds+pD, hs:hs+pH, ws:ws+pW] += flow_accum
        count_map[ds:ds+pD, hs:hs+pH, ws:ws+pW] += 1
        pbar.update(1)
    pbar.close()

    seg_avg = seg_sum / (count_map[None, ...] + 1e-8)
    dist_avg = dist_sum / (count_map + 1e-8)
    flow_avg = flow_sum / (count_map[None, ...] + 1e-8)

    pred_seg = np.argmax(seg_avg, axis=0).astype(np.uint8)
    pred_dist = dist_avg
    pred_flow = np.argmax(flow_avg, axis=0).astype(np.uint8)
    pred_flow[pred_seg == 0] = 255
    return pred_seg, pred_dist, pred_flow


def dice_score(pred, gt):
    pred = (pred == 1).astype(np.float32)
    gt = (gt == 1).astype(np.float32)
    inter = (pred * gt).sum()
    denom = pred.sum() + gt.sum() + 1e-8
    return float((2.0 * inter) / denom)


def apply_lcc(seg_bin: np.ndarray) -> np.ndarray:
    if not SCIPY_OK:
        return seg_bin  # fallback: no-op if scipy missing
    struct = generate_binary_structure(rank=3, connectivity=1)  # 6-connectivity
    labeled, n = cc_label(seg_bin.astype(bool), structure=struct)
    if n <= 1:
        return seg_bin
    # keep component with max voxels
    counts = np.bincount(labeled.ravel())
    if counts.size > 0:
        counts[0] = 0  # ignore background
    best_lab = int(np.argmax(counts)) if counts.size > 1 else 0
    out = (labeled == best_lab).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description='Full-volume inference with TTA and optional LCC decision')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, default=None, help='GT seg directory (optional)')
    parser.add_argument('--dist_dir', type=str, default=None, help='GT dist directory (optional, .npy or .nii.gz)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='best_model.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128,128,128])
    parser.add_argument('--stride', type=int, nargs=3, default=[64,64,64])
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--mirror_axes', type=int, nargs='*', default=[0,1,2])
    parser.add_argument('--auto_lcc_decide', action='store_true')
    parser.add_argument('--skip_lcc_eval', action='store_true', help='Skip per-case LCC Dice eval to save CPU time')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # Speed knobs: limit CPU threads and enable cuDNN autotuner
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    cudnn.benchmark = True

    # load config
    cfg_path = os.path.join(args.experiment_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'Config not found: {cfg_path}')
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # build model
    n_flow_classes = config['model'].get('num_flow_classes', config['model'].get('n_flow_classes', 38))
    pretrain_path = config.get('pretrain', {}).get('checkpoint_path', None)
    model = nnUNetMultiTaskWithAttention(
        num_flow_classes=n_flow_classes,
        pretrained_path=pretrain_path,
        freeze_encoder=False,
        use_spatial_attention=False,
        attention_reduction=16
    ).to(device)

    # load checkpoint weights
    ckpt_path = os.path.join(args.experiment_dir, 'checkpoints', args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # load direction codec
    codec_path = os.path.join(config['data']['processed_dir'], 'direction_codec.npy')
    codec = DirectionCodec.load(codec_path)

    # prepare IO
    out_dir = args.output_dir or os.path.join(args.experiment_dir, 'test_predictions_attention')
    os.makedirs(out_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith('.nii.gz')])

    per_case_metrics = []
    dices_no_lcc = []
    dices_lcc = []

    for img_file in tqdm(image_files, desc='Volumes'):
        case_id = img_file.replace('_0000.nii.gz', '').replace('.nii.gz', '')
        img_path = os.path.join(args.image_dir, img_file)
        nii = nib.load(img_path)
        image = nii.get_fdata()
        affine = nii.affine
        image = normalize_image(image)

        # inference
        pred_seg, pred_dist, pred_flow = sliding_window_inference(
            image=image,
            model=model,
            patch_size=tuple(args.patch_size),
            stride=tuple(args.stride),
            device=device,
            num_seg_classes=2,
            num_flow_classes=n_flow_classes,
            tta=args.tta,
            mirror_axes=tuple(args.mirror_axes)
        )

        # save preds
        case_out = os.path.join(out_dir, case_id)
        os.makedirs(case_out, exist_ok=True)
        nib.save(nib.Nifti1Image(pred_seg.astype(np.uint8), affine), os.path.join(case_out, f'{case_id}_pred_seg.nii.gz'))
        nib.save(nib.Nifti1Image(pred_dist.astype(np.float32), affine), os.path.join(case_out, f'{case_id}_pred_dist.nii.gz'))
        nib.save(nib.Nifti1Image(pred_flow.astype(np.uint8), affine), os.path.join(case_out, f'{case_id}_pred_flow.nii.gz'))

        # metrics (if GT provided)
        if args.seg_dir is not None:
            gt_seg_path = os.path.join(args.seg_dir, f'{case_id}.nii.gz')
            if os.path.exists(gt_seg_path):
                gt_seg = nib.load(gt_seg_path).get_fdata().astype(np.uint8)
                dice0 = dice_score(pred_seg, gt_seg)
                dices_no_lcc.append(dice0)
                dice_lcc = None
                if SCIPY_OK and (not args.skip_lcc_eval):
                    pred_lcc = apply_lcc((pred_seg == 1).astype(np.uint8))
                    dice_lcc = dice_score(pred_lcc, gt_seg)
                    dices_lcc.append(dice_lcc)
                per_case_metrics.append({
                    'case_id': case_id,
                    'seg_dice_no_lcc': dice0,
                    'seg_dice_lcc': dice_lcc
                })

        # also load and save GT dist if provided
        if args.dist_dir is not None:
            # support npy or nii.gz
            npy_path = os.path.join(args.dist_dir, f'{case_id}.npy')
            nii_path = os.path.join(args.dist_dir, f'{case_id}.nii.gz')
            if os.path.exists(npy_path):
                gt_dist = np.load(npy_path)
                nib.save(nib.Nifti1Image(gt_dist.astype(np.float32), affine), os.path.join(case_out, f'{case_id}_gt_dist.nii.gz'))
            elif os.path.exists(nii_path):
                gt_dist = nib.load(nii_path).get_fdata().astype(np.float32)
                nib.save(nib.Nifti1Image(gt_dist, affine), os.path.join(case_out, f'{case_id}_gt_dist.nii.gz'))

    # aggregate metrics
    if per_case_metrics:
        avg = {
            'avg_dice_no_lcc': float(np.mean(dices_no_lcc)) if dices_no_lcc else None,
            'avg_dice_lcc': float(np.mean(dices_lcc)) if dices_lcc else None,
        }
        metrics_path = os.path.join(out_dir, 'test_metrics_attention.json')
        with open(metrics_path, 'w') as f:
            json.dump({'average': avg, 'per_case': per_case_metrics}, f, indent=2)

        # auto lcc decide & persist
        if args.auto_lcc_decide and SCIPY_OK and dices_lcc:
            apply_lcc_flag = (avg['avg_dice_lcc'] is not None) and (avg['avg_dice_no_lcc'] is not None) and (avg['avg_dice_lcc'] > avg['avg_dice_no_lcc'])
            decision = {
                'apply_lcc': bool(apply_lcc_flag),
                'avg_dice_no_lcc': avg['avg_dice_no_lcc'],
                'avg_dice_lcc': avg['avg_dice_lcc'],
                'improvement': float(avg['avg_dice_lcc'] - avg['avg_dice_no_lcc'])
            }
            out_json = os.path.join(args.experiment_dir, 'postproc_lcc.json')
            with open(out_json, 'w') as f:
                json.dump(decision, f, indent=2)
            print(f"\nSaved LCC postproc decision to {out_json}: {decision}")
        elif args.auto_lcc_decide and not SCIPY_OK:
            print('\n[Warn] scipy not found, skip auto_lcc_decide')

    print(f"\n✓ All outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()
