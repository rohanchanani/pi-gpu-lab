#!/usr/bin/env python3
"""Validate VC4 M2 layout.json for lit tests."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"check_m2_layout.py: ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def region_list(data):
    raw = data.get('regions') or data.get('layout') or []
    if isinstance(raw, dict):
        out=[]
        for name, item in raw.items():
            if isinstance(item, dict):
                item=dict(item); item.setdefault('name', name); out.append(item)
        return out
    return raw if isinstance(raw, list) else []


def main(argv):
    ap=argparse.ArgumentParser()
    ap.add_argument('layout')
    ap.add_argument('--alignment', type=int, default=8)
    ap.add_argument('--require-heap', action='store_true')
    ap.add_argument('--min-heap-bytes', type=int, default=0)
    ns=ap.parse_args(argv)
    data=json.loads(Path(ns.layout).read_text())
    regions=region_list(data)
    if not regions:
        fail('layout has no regions')
    normalized=[]
    for r in regions:
        if not isinstance(r, dict): fail('region entry is not an object')
        name=str(r.get('name') or r.get('kind') or '')
        off=int(r.get('offset_bytes', r.get('offset', -1)))
        size=int(r.get('size_bytes', r.get('size', -1)))
        if not name or off < 0 or size < 0: fail(f'invalid region {r!r}')
        if off % ns.alignment != 0: fail(f'region {name} offset {off} is not {ns.alignment}-byte aligned')
        normalized.append((off, off+size, name, size))
    normalized.sort()
    for a,b in zip(normalized, normalized[1:]):
        if a[1] > b[0]:
            fail(f'regions overlap: {a[2]} and {b[2]}')
    heaps=[r for r in normalized if 'heap' in r[2].lower()]
    if ns.require_heap and not heaps: fail('heap region missing')
    if ns.min_heap_bytes and not any(size >= ns.min_heap_bytes for _,_,_,size in heaps):
        fail(f'no heap region at least {ns.min_heap_bytes} bytes')
    print(f"check_m2_layout.py: PASS {ns.layout} regions={len(normalized)}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
