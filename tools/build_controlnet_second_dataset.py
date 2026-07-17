"""Compatibility notice for the retired ImageFolder ControlNet adapter.

The old adapter stored ``conditioning_image_file_name`` as JSON metadata, but
Hugging Face ImageFolder only decodes ``file_name`` as an image.  The condition
therefore remained a string and failed at training time.  The maintained
ControlNet trainer reads the common SECOND JSONL manifest directly.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retired: use run_bash/controlnet_second_prepare.bash and the direct-manifest trainer."
    )
    parser.parse_args()
    raise SystemExit(
        "This ImageFolder adapter is retired. Run "
        "`bash run_bash/controlnet_second_prepare.bash`, then "
        "`bash run_bash/controlnet_second_train.bash`."
    )


if __name__ == "__main__":
    main()
