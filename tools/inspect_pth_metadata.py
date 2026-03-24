#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import torch


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def iso_time(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def sha256_head(path: str, num_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(num_bytes))
    return h.hexdigest()


def file_metadata(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size_bytes": st.st_size,
        "size_human": human_bytes(st.st_size),
        "modified": iso_time(st.st_mtime),
        "created_or_ctime": iso_time(st.st_ctime),
        "sha256_first_8mb": sha256_head(path),
    }


def is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def iter_children(x: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(x, dict):
        for k, v in x.items():
            yield str(k), v
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield str(i), v


def collect_tensors(root: Any, max_nodes: int = 2_000_000) -> List[Tuple[str, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor]] = []
    stack: List[Tuple[str, Any]] = [("", root)]
    visited = 0

    while stack:
        path, node = stack.pop()
        visited += 1
        if visited > max_nodes:
            break

        if torch.is_tensor(node):
            out.append((path or "<root>", node))
            continue

        if isinstance(node, (dict, list, tuple)):
            for k, child in iter_children(node):
                child_path = f"{path}.{k}" if path else k
                stack.append((child_path, child))

    return out


def summarize_top_level(obj: Any, max_keys: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"python_type": type(obj).__name__}
    if isinstance(obj, dict):
        keys = list(obj.keys())
        summary["num_top_level_keys"] = len(keys)
        summary["top_level_keys_preview"] = [str(k) for k in keys[:max_keys]]

        trainer = obj.get("trainer")
        if isinstance(trainer, dict):
            tkeys = list(trainer.keys())
            summary["trainer_keys_preview"] = [str(k) for k in tkeys[:max_keys]]

    return summary


def _is_state_dict_like(d: Any) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    vals = list(d.values())
    tensor_vals = sum(1 for v in vals if torch.is_tensor(v))
    return tensor_vals >= max(1, int(0.8 * len(vals)))


def _find_state_dict_candidates(root: Any, max_nodes: int = 2_000_000) -> List[Tuple[str, Dict[str, Any]]]:
    cands: List[Tuple[str, Dict[str, Any]]] = []
    stack: List[Tuple[str, Any]] = [("", root)]
    visited = 0

    while stack:
        path, node = stack.pop()
        visited += 1
        if visited > max_nodes:
            break

        if _is_state_dict_like(node):
            cands.append((path or "<root>", node))
            continue

        if isinstance(node, (dict, list, tuple)):
            for k, child in iter_children(node):
                child_path = f"{path}.{k}" if path else k
                stack.append((child_path, child))
    return cands


def _module_prefix(name: str, depth: int) -> str:
    parts = name.split(".")
    if len(parts) <= 1:
        return "<root_param>"
    end = min(max(1, depth), len(parts) - 1)
    return ".".join(parts[:end])


def summarize_architecture(
    state_dict: Dict[str, Any], depth: int, top_modules: int, param_preview: int
) -> Dict[str, Any]:
    total_params = 0
    num_tensors = 0
    module_param_counter: Counter = Counter()
    module_tensor_counter: Counter = Counter()
    sample_params: List[Dict[str, Any]] = []

    for idx, (name, value) in enumerate(state_dict.items()):
        if not torch.is_tensor(value):
            continue
        n = value.numel()
        total_params += n
        num_tensors += 1
        mod = _module_prefix(name, depth=depth)
        module_param_counter[mod] += n
        module_tensor_counter[mod] += 1
        if idx < param_preview:
            sample_params.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "numel": int(n),
                }
            )

    top = module_param_counter.most_common(top_modules)
    module_rows = []
    for mod, pcount in top:
        module_rows.append(
            {
                "module": mod,
                "params": int(pcount),
                "params_human": f"{pcount:,}",
                "tensor_count": int(module_tensor_counter[mod]),
            }
        )

    return {
        "num_state_dict_items": len(state_dict),
        "num_tensor_items": num_tensors,
        "total_params": int(total_params),
        "total_params_human": f"{total_params:,}",
        "depth_used": depth,
        "top_modules_by_params": module_rows,
        "param_name_preview": sample_params,
    }


def tensor_stats(tensors: List[Tuple[str, torch.Tensor]], top_n: int) -> Dict[str, Any]:
    dtype_counter = Counter()
    device_counter = Counter()
    total_numel = 0
    total_tensor_bytes = 0

    largest: List[Tuple[str, int, str, str, List[int]]] = []

    for name, t in tensors:
        numel = t.numel()
        nbytes = numel * t.element_size()
        total_numel += numel
        total_tensor_bytes += nbytes
        dtype_counter[str(t.dtype)] += 1
        device_counter[str(t.device)] += 1
        largest.append((name, nbytes, str(t.dtype), str(t.device), list(t.shape)))

    largest.sort(key=lambda x: x[1], reverse=True)
    largest = largest[:top_n]

    return {
        "num_tensors": len(tensors),
        "total_numel": int(total_numel),
        "total_tensor_bytes": int(total_tensor_bytes),
        "total_tensor_size_human": human_bytes(total_tensor_bytes),
        "dtype_histogram": dict(dtype_counter),
        "device_histogram": dict(device_counter),
        "largest_tensors": [
            {
                "name": n,
                "bytes": b,
                "size_human": human_bytes(b),
                "dtype": d,
                "device": dev,
                "shape": shp,
            }
            for n, b, d, dev, shp in largest
        ],
    }


def inspect_pth(
    path: str,
    max_keys: int,
    top_n: int,
    arch_depth: int,
    arch_top_modules: int,
    arch_param_preview: int,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "file": file_metadata(path),
    }

    obj = torch.load(path, map_location="cpu")
    report["checkpoint"] = summarize_top_level(obj, max_keys=max_keys)

    tensors = collect_tensors(obj)
    report["tensors"] = tensor_stats(tensors, top_n=top_n)

    state_dict_candidates = _find_state_dict_candidates(obj)
    if state_dict_candidates:
        chosen_path, chosen_sd = max(
            state_dict_candidates, key=lambda x: sum(v.numel() for v in x[1].values() if torch.is_tensor(v))
        )
        report["architecture"] = {
            "state_dict_candidates": [p for p, _ in state_dict_candidates],
            "selected_state_dict_path": chosen_path,
            "summary": summarize_architecture(
                chosen_sd,
                depth=arch_depth,
                top_modules=arch_top_modules,
                param_preview=arch_param_preview,
            ),
        }
    else:
        report["architecture"] = {
            "state_dict_candidates": [],
            "selected_state_dict_path": None,
            "summary": None,
        }
    return report


def print_human(label: str, report: Dict[str, Any]) -> None:
    print(f"\n=== {label} ===")

    f = report["file"]
    print(f"Path: {f['path']}")
    print(f"Size: {f['size_human']} ({f['size_bytes']} bytes)")
    print(f"Modified: {f['modified']}")
    print(f"Created/ctime: {f['created_or_ctime']}")
    print(f"SHA256(first 8MB): {f['sha256_first_8mb']}")

    c = report["checkpoint"]
    print(f"Checkpoint type: {c.get('python_type')}")
    if "num_top_level_keys" in c:
        print(f"Top-level keys: {c['num_top_level_keys']}")
        print(f"Top-level key preview: {c['top_level_keys_preview']}")
    if "trainer_keys_preview" in c:
        print(f"Trainer key preview: {c['trainer_keys_preview']}")

    t = report["tensors"]
    print(f"Tensors: {t['num_tensors']}")
    print(f"Total numel: {t['total_numel']}")
    print(f"Tensor payload size: {t['total_tensor_size_human']} ({t['total_tensor_bytes']} bytes)")
    print(f"Dtypes: {t['dtype_histogram']}")
    print(f"Devices: {t['device_histogram']}")

    print("Largest tensors:")
    for item in t["largest_tensors"]:
        print(
            f"  - {item['name']}: {item['size_human']} | {item['dtype']} | {item['device']} | shape={item['shape']}"
        )

    arch = report.get("architecture", {})
    print("Architecture:")
    print(f"  State-dict candidates: {arch.get('state_dict_candidates', [])}")
    print(f"  Selected state-dict path: {arch.get('selected_state_dict_path')}")
    s = arch.get("summary")
    if s is None:
        print("  No state_dict-like structure detected.")
        return

    print(f"  Total params: {s['total_params_human']}")
    print(f"  Tensor params: {s['num_tensor_items']}")
    print(f"  Depth used: {s['depth_used']}")
    print("  Top modules by params:")
    for item in s["top_modules_by_params"]:
        print(
            f"    - {item['module']}: {item['params_human']} params across {item['tensor_count']} tensors"
        )
    print("  Parameter name preview:")
    for item in s["param_name_preview"]:
        print(
            f"    - {item['name']}: shape={item['shape']} dtype={item['dtype']} numel={item['numel']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect metadata and tensor stats of .pth checkpoints.")
    parser.add_argument("--infinity", type=str, default=None, help="Path to Infinity .pth")
    parser.add_argument("--vae", type=str, default=None, help="Path to VAE .pth")
    parser.add_argument("--pth", nargs="*", default=[], help="Additional .pth files")
    parser.add_argument("--max-keys", type=int, default=20, help="Number of top-level keys to preview")
    parser.add_argument("--top-tensors", type=int, default=10, help="Number of largest tensors to print")
    parser.add_argument("--arch-depth", type=int, default=2, help="Hierarchy depth when grouping module params")
    parser.add_argument("--arch-top-modules", type=int, default=20, help="How many module groups to print")
    parser.add_argument("--arch-param-preview", type=int, default=30, help="How many parameter keys to preview")
    parser.add_argument("--json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    targets: List[Tuple[str, str]] = []
    if args.infinity:
        targets.append(("Infinity", args.infinity))
    if args.vae:
        targets.append(("VAE", args.vae))
    for i, p in enumerate(args.pth):
        targets.append((f"Extra-{i}", p))

    if not targets:
        parser.error("Provide at least one of --infinity, --vae, or --pth.")

    reports: Dict[str, Any] = {}
    for label, path in targets:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")
        if not path.endswith(".pth"):
            print(f"[Warning] {path} does not end with .pth")
        rep = inspect_pth(
            path,
            max_keys=args.max_keys,
            top_n=args.top_tensors,
            arch_depth=args.arch_depth,
            arch_top_modules=args.arch_top_modules,
            arch_param_preview=args.arch_param_preview,
        )
        reports[label] = rep
        print_human(label, rep)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"\nSaved JSON report to: {os.path.abspath(args.json)}")


if __name__ == "__main__":
    main()

# ------USAGE------

# python3 tools/inspect_pth_metadata.py \
#   --infinity weights/infinity_2b_reg.pth \
#   --json infinity_report_arch.json

# python3 tools/inspect_pth_metadata.py \
#   --vae weights/infinity_vae_d32reg.pth \
#   --json report_arch.json