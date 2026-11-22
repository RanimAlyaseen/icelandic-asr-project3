"""Download and verify the recovered Whisper Small checkpoint 1000."""

import argparse
import hashlib
from pathlib import Path

import gdown

FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1BKWu0PBXdNlp7pqeX7MCn9lhzFsay_Md"
)

EXPECTED_SHA256 = {
    "config.json": "572ff2d955ca237231a371e756e41b4a4c0c63df3c30e64538b90294f468e449",
    "generation_config.json": "2d3e60d85c13324649773e42932b2a9916edd3ff0f24434b17db657c61c57fd0",
    "model.safetensors": "d056c430a0b4c1d9006a9ddf68a0b8d92f1cf4828ccb4f5bd96f77d5f911f80b",
    "processor_config.json": "8c648fa85aafc59a07435ed751322499bd4238b819ca9200ddd2b5ba4df50e68",
    "recovery_metadata.json": "21e255d551be5020ca1c0304b8b25b6bf88c018928bc6768b0b5cd341bff7f09",
    "tokenizer.json": "783ab7faf82595b8abc49cef57865b055d763753e6383d0d98774abcfe16e57a",
    "tokenizer_config.json": "0940f4bfe73310f1fdc9f86242568178700211fde2d9b49b15dd1ad892552641",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_checkpoint(directory):
    for filename, expected_hash in EXPECTED_SHA256.items():
        path = directory / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint file: {filename}"
            )

        actual_hash = sha256_file(path)

        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

        print(f"PASS: {filename}")

    print("\nALL CHECKSUMS VERIFIED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/recovered_model_step1000"),
        help="Local directory for the recovered checkpoint",
    )

    args = parser.parse_args()
    output_dir = args.output

    if output_dir.exists() and all(
        (output_dir / filename).is_file()
        for filename in EXPECTED_SHA256
    ):
        print("Existing checkpoint found. Verifying...")

        try:
            verify_checkpoint(output_dir)
            print(f"\nCheckpoint ready: {output_dir}")
            return
        except (ValueError, FileNotFoundError):
            print("Existing checkpoint is invalid. Re-downloading...")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading recovered Whisper Small checkpoint 1000...")

    downloaded = gdown.download_folder(
        url=FOLDER_URL,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
    )

    if not downloaded:
        raise RuntimeError("Checkpoint download failed.")

    print("\nVerifying downloaded checkpoint...")
    verify_checkpoint(output_dir)

    print(f"\nCheckpoint ready: {output_dir}")


if __name__ == "__main__":
    main()
