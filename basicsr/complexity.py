# This code follows the same logic as profiling for RetinexFormer (ICCV 2023) and HVI-CIDNet (CVPR 2025)

#!/usr/bin/env python3
# complexity.py
import argparse
import logging
import pprint
import re
import importlib
from copy import deepcopy
import time
import statistics as stats
import math

import torch

from basicsr.utils.options import parse
from basicsr.utils import get_root_logger

# -----------------------------
# Pretty preview helper
# -----------------------------
def _safe_preview(obj, max_len=300):
    s = pprint.pformat(obj, compact=True, width=100)
    return s if len(s) <= max_len else s[:max_len] + " ..."

# -----------------------------
# network_g normalization
# -----------------------------
def _normalize_network_g(ng):
    """Return a dict that includes a 'type' key. Supports several YAML styles."""
    if ng is None:
        raise KeyError("Missing 'network_g' in options.")

    if isinstance(ng, list):
        if not ng:
            raise KeyError("'network_g' is an empty list.")
        ng = ng[0]

    if not isinstance(ng, dict):
        raise TypeError(f"'network_g' must be a dict (or list[dict]). Got: {type(ng)}")

    if 'type' in ng:
        return ng

    if 'which_model_G' in ng:
        out = dict(ng)
        out['type'] = out.pop('which_model_G')
        return out
    if 'name' in ng:
        out = dict(ng)
        out['type'] = out.pop('name')
        return out

    if len(ng) == 1:
        k, v = next(iter(ng.items()))
        if isinstance(v, dict):
            return {'type': k, **v}

    raise KeyError(
        "Could not find 'type' for network_g.\n"
        "Acceptable forms:\n"
        "  network_g:\n"
        "    type: ClassName\n"
        "    in_ch: 3\n"
        "    ...\n"
        "or\n"
        "  network_g:\n"
        "    ClassName:\n"
        "      in_ch: 3\n"
        "      ...\n"
        "or\n"
        "  network_g:\n"
        "    which_model_G: ClassName\n"
        "    in_ch: 3\n"
        "    ...\n"
    )

def _class_to_arch_module(class_name: str) -> str:
    """CamelCase -> snake_case + _arch"""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', class_name)
    snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return f"{snake}_arch"

# -----------------------------
# Build net_g from options
# -----------------------------
def _build_net_from_opt(opt):
    ng_raw = opt.get('network_g')
    print("[complexity] network_g (raw) =", _safe_preview(ng_raw))

    ng = _normalize_network_g(ng_raw)
    print("[complexity] network_g (normalized) =", _safe_preview(ng))

    ng_pristine = deepcopy(ng)

    # Try BasicSR dispatchers first — pass deep copies so they can't mutate ours
    try:
        from basicsr.models.archs import define_network
        return define_network(deepcopy(ng))
    except Exception:
        pass
    try:
        from basicsr.models.archs import build_network
        return build_network(deepcopy(ng))
    except Exception:
        pass

    # Fallback: dynamic import
    cls_name = ng_pristine['type']
    module_path = ng_pristine.get('arch_module')
    if not module_path:
        # try CamelCase_arch first
        module_path = f"basicsr.models.archs.{cls_name}_arch"
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        # fallback to snake_case_arch
        snake = _class_to_arch_module(cls_name)
        module_path = f"basicsr.models.archs.{snake}"
        mod = importlib.import_module(module_path)

    cls = getattr(mod, cls_name)
    kwargs = {k: v for k, v in ng_pristine.items() if k not in ('type', 'arch_module')}
    return cls(**kwargs)

# -----------------------------
# Complexity
# -----------------------------
def _count_params(model):
    return sum(p.numel() for p in model.parameters())

def _compute_complexity(model, input_res=(3, 256, 256), device='cpu'):
    """
    Returns a dict with available complexity metrics.
    Tries THOP then ptflops. May return subset if packages not installed.
    """
    results = {}
    model = model.to(device).eval()
    x = torch.randn(1, *input_res, device=device)

    # Try THOP
    try:
        from thop import profile
        with torch.no_grad():
            macs, _ = profile(model, inputs=(x,), verbose=False)
        results['gflops'] = macs / 1e9
    except Exception:
        pass

    return results

# -----------------------------
# Stats helpers
# -----------------------------
def _percentile(values, q):
    """Linear interpolation between closest ranks; q in [0,100]. Returns NaN on empty."""
    if not values:
        return float('nan')
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    data = sorted(values)
    k = (len(data) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(data[int(k)])
    d0 = data[f]
    d1 = data[c]
    return float(d0 + (d1 - d0) * (k - f))

# -----------------------------
# Inference-time benchmarking
# -----------------------------
def _benchmark_inference(model,
                         input_res=(3, 256, 256),
                         device='cpu',
                         warmup=10,
                         runs=50,
                         amp=False):
    """
    Returns timing dict with: mean, median, std, p95, p99, fps, and raw timings.
    Uses CUDA events for GPU, perf_counter for CPU.
    """
    model.eval().to(device)
    x = torch.randn(1, *input_res, device=device)
    use_cuda = (device == 'cuda' and torch.cuda.is_available())

    # CUDNN autotune can help for fixed shapes
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    timings = []

    @torch.no_grad()
    def _forward_once():
        if amp and use_cuda:
            with torch.cuda.amp.autocast():
                _ = model(x)
        else:
            _ = model(x)

    # Warmup
    for _ in range(max(0, warmup)):
        if use_cuda:
            torch.cuda.synchronize()
        _forward_once()
        if use_cuda:
            torch.cuda.synchronize()

    # Measured runs
    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            starter.record()
            _forward_once()
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))  # milliseconds
    else:
        for _ in range(runs):
            t0 = time.perf_counter()
            _forward_once()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    ms_mean = float(sum(timings) / len(timings)) if timings else float('nan')
    ms_median = float(stats.median(timings)) if timings else float('nan')
    ms_std = float(stats.pstdev(timings)) if len(timings) > 1 else 0.0
    ms_p95 = _percentile(timings, 95.0)
    ms_p99 = _percentile(timings, 99.0)
    fps_mean = 1000.0 / ms_mean if ms_mean > 0 else float('nan')

    return dict(
        ms_mean=ms_mean,
        ms_median=ms_median,
        ms_std=ms_std,
        ms_p95=ms_p95,
        ms_p99=ms_p99,
        fps=fps_mean,
        runs=runs,
        warmup=warmup,
        amp=amp,
        timings=timings
    )

# -----------------------------
# Resolution parsing
# -----------------------------
def _parse_resolutions(res_str):
    """
    Accepts formats like:
      "256x256,320x320,640x360"
      "256,256; 320,320 ; 1280,720"
      "256 256 | 320 320 | 1920 1080"
    Returns list of (H, W) tuples.
    """
    if not res_str:
        return [(320, 320), (416, 416), (640, 640)]
    # unify separators
    cleaned = res_str.replace(';', ',').replace('|', ',')
    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
    out = []
    for p in parts:
        if 'x' in p.lower():
            h, w = p.lower().split('x')
        elif ' ' in p:
            h, w = p.split()
        elif ':' in p:
            h, w = p.split(':')
        elif p.count('/') == 1:
            h, w = p.split('/')
        elif p.count('-') == 1:
            h, w = p.split('-')
        elif p.count('_') == 1:
            h, w = p.split('_')
        elif p.count('.') == 1:
            h, w = p.split('.')
        elif p.count(';') == 1:
            h, w = p.split(';')
        else:
            # "256,256" case
            h, w = p.split()
        try:
            h = int(h)
            w = int(w)
            out.append((h, w))
        except Exception:
            # fallback for "256,256"
            try:
                h, w = [int(t.strip()) for t in p.split()]
                out.append((h, w))
            except Exception:
                try:
                    h, w = [int(t.strip()) for t in p.split(',')]
                    out.append((h, w))
                except Exception:
                    raise ValueError(f"Could not parse resolution token: '{p}'")
    # dedupe while preserving order
    seen = set()
    uniq = []
    for hw in out:
        if hw not in seen:
            uniq.append(hw)
            seen.add(hw)
    return uniq if uniq else [(256, 256)]

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Model complexity & speed summary over multiple resolutions"
    )
    parser.add_argument('--opt', type=str, required=True,
                        help='Path to option YAML file (same as train).')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help='Device for profiling.')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations before timing.')
    parser.add_argument('--runs', type=int, default=50,
                        help='Timed iterations per resolution.')
    parser.add_argument('--amp', action='store_true',
                        help='Use torch.cuda.amp autocast for timing (CUDA only).')
    parser.add_argument('--resolutions', type=str, default='',
                        help=('Comma/semicolon separated list, e.g. '
                              '"256x256,320x320,640x360,1280x720,1920x1080"'))
    args = parser.parse_args()

    # Parse options like train.py but force non-distributed
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1
    opt.setdefault('num_gpu', 1)

    # Logger
    logger = get_root_logger(logger_name='complexity', log_level=logging.INFO)
    logger.info(f"Loaded options from: {args.opt}")

    # Build net_g
    if 'network_g' not in opt:
        raise KeyError("Options must contain 'network_g' to build the generator.")
    net_g = _build_net_from_opt(opt)
    net_g = net_g.to('cuda').eval()

    # Params
    n_params = _count_params(net_g)
    n_params_m = n_params / 1e6

    # Device
    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'

    # Resolutions list
    res_list = _parse_resolutions(args.resolutions)

    # Header
    print("\n" + "=" * 86)
    print(" Model Complexity & Speed (multiple input resolutions) ")
    print("=" * 86)
    print(f"Network type     : {opt['network_g'].get('type', 'Unknown')}")
    print(f"Params           : {n_params:,}  (~{n_params_m:.3f} M | ~{n_params_m*1000:.2f} K)")
    print(f"Device           : {device}   |   Inference (AMP): {'on' if (args.amp and device=='cuda') else 'off'}")
    print(f"Timing config    : warmup={args.warmup}  runs={args.runs}")

    # Per-resolution profiling
    rows = []
    for (h, w) in res_list:
        input_res = (3, h, w)

        # Complexity
        comp = _compute_complexity(net_g, input_res=input_res, device=device)

        # Timing
        timing = _benchmark_inference(
            net_g, input_res=input_res, device=device,
            warmup=args.warmup, runs=args.runs, amp=args.amp
        )

        if device == 'cuda':
            torch.cuda.empty_cache()

        rows.append({
            'res': f"{h}x{w}",
            'gflops': comp.get('gflops', None),
            'ms_mean': timing['ms_mean'],
            'ms_med': timing['ms_median'],
            'ms_p95': timing['ms_p95'],
            'ms_p99': timing['ms_p99'],
            'ms_std': timing['ms_std'],
            'fps': timing['fps'],
        })

    # Pretty print table
    print("\nPer-resolution results:")
    header = (
        f"{'Resolution':>12} | {'GFLOPs':>10} | "
        f"{'mean(ms)':>9} | {'median':>8} | {'p95':>8} | {'p99':>8} | {'std':>8} | {'FPS':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        gflops = f"{r['gflops']:.3f}" if r['gflops'] is not None else '--'
        print(f"{r['res']:>12} | {gflops:>10} | "
              f"{r['ms_mean']:>9.3f} | {r['ms_med']:>8.3f} | {r['ms_p95']:>8.3f} | {r['ms_p99']:>8.3f} | {r['ms_std']:>8.3f} | {r['fps']:>8.2f}")

    print("\n" + "=" * 86 + "\n")

if __name__ == '__main__':
    main()
