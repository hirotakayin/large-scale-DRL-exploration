#!/usr/bin/env python3
"""The stage-1 world sets (docs/PLAN_3D.md): accepted ari3d worlds for training, a few flat ones mixed in, and a
held-out set from PNGs that training never sees. Each set is a folder of make_ari3d.py worlds plus worlds.txt,
the list env3d.py trains on (its order mixes flat and ari3d worlds).

  python make_world_sets.py --out /worlds --voxel /voxel_world      (in the ariadne-train image, see run_train.sh)

PNGs are drawn without overlap from maps/ with --seed; world seeds are make_ari3d.py's (--seed + k per call).
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np


def make(voxel, pngs, out, seed, flat=False):
    cmd = [sys.executable, os.path.join(voxel, 'scripts', 'make_ari3d.py'), '--maps', 'maps', '--seed', str(seed),
           '--out', out, '--pngs', *pngs] + (['--flat'] if flat else [])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return {r['folder']: r for r in json.load(open(os.path.join(out, 'index.json')))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--voxel', required=True, help='voxel_world folder')
    ap.add_argument('--train', type=int, default=200)
    ap.add_argument('--flat', type=int, default=20)
    ap.add_argument('--heldout', type=int, default=20)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    pngs = list(np.random.default_rng(a.seed).permutation(sorted(glob.glob('maps/*.png'))))

    def accepted_set(name, n, seed, take):
        out = os.path.join(a.out, name)
        chosen = []
        while len(chosen) < n:
            batch = [take.pop(0) for _ in range(min(len(take), (n - len(chosen)) * 13 // 10 + 2))]
            idx = make(a.voxel, batch, out, seed)
            stems = {os.path.splitext(os.path.basename(p))[0] for p in batch}
            new = [f for f, r in sorted(idx.items()) if r['accepted'] and f.rsplit('_s', 1)[0] in stems]
            chosen += new[:n - len(chosen)]
            seed += len(batch)
        return chosen

    heldout = accepted_set('ari3d_heldout', a.heldout, 9000, pngs)
    with open(os.path.join(a.out, 'ari3d_heldout', 'worlds.txt'), 'w') as f:
        f.write('\n'.join(heldout) + '\n')
    train = accepted_set('ari3d_train', a.train, 0, pngs)
    flat_pngs = [pngs.pop(0) for _ in range(a.flat)]
    idx = make(a.voxel, flat_pngs, os.path.join(a.out, 'ari3d_train'), 5000, flat=True)
    flat = sorted(f for f in idx if int(f.rsplit('_s', 1)[1]) >= 5000)
    worlds = list(np.random.default_rng(a.seed).permutation(train + flat))
    with open(os.path.join(a.out, 'ari3d_train', 'worlds.txt'), 'w') as f:
        f.write('\n'.join(worlds) + '\n')
    print('held out %d, train %d ari3d + %d flat' % (len(heldout), len(train), len(flat)))


if __name__ == '__main__':
    main()
