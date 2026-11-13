"""Regenerate the submission into a temporary folder and compare it byte for byte.

    python scripts/check_reproducible.py --name "Team Claude"

RULES.md: "Running your repository against the provided data must reproduce your submission.
Not approximately -- the same file." Prints only whether the two files are identical.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="Team Claude")
    args = parser.parse_args()

    committed = os.path.join(HERE, "submissions", args.name + ".csv")
    with tempfile.TemporaryDirectory() as out:
        subprocess.run([sys.executable, os.path.join(HERE, "predict.py"), "--name", args.name,
                        "--out", out], check=True, cwd=HERE)
        same = sha256(committed) == sha256(os.path.join(out, args.name + ".csv"))
    print("reproducible:", "YES - byte-identical" if same else "NO - files differ")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
