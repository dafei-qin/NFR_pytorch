import argparse
import os

import trimesh

from signature import SignatureExtractor

parser = argparse.ArgumentParser(description='Mesh signature visualization')
parser.add_argument('files', help='File to load', nargs='+')
parser.add_argument('--suffix', default='')
parser.add_argument('--n_basis', default='100', type=int, help='Number of basis used')
parser.add_argument('--f_size', default='128', type=int, help='Feature size used')
parser.add_argument('--approx', default='cotangens', help="Laplace approximation to use must be in ['beltrami', 'cotangens', 'mesh']")

args = parser.parse_args()

for f in args.files:
    if not os.path.exists(f):
        print(f"File \"{f}\" not found")
        continue
    print(f"Processing file {f}")

    mesh = trimesh.load(f, force='mesh')
    extractor = SignatureExtractor(
        mesh=mesh,
        n_basis=args.n_basis,
        approx=args.approx)
    filename = os.path.splitext(f)[0]
    out = f"{filename}.npz" if len(args.suffix) == 0 else f"{filename}_{args.suffix}.npz"
    extractor.save(out)
