# Example image runs

This folder contains 14 selected scene runs from the RAGArtist experiments. The original scene directory names are preserved so they can be matched to prompt and evaluation records. See [manifest.json](manifest.json) for the exact files in each run.

Within a scene directory:

- `0.png` is the retrieval-grounded ImageRAG base image.
- Other numbered PNGs are recorded agentic outputs or intermediate images; numbering follows the original run and does **not** imply that the highest number was selected as the final output.
- `*_mask.png` files are editing masks.
- `0.txt` stores the prompt when present.
- `detection_results.json` and `correction_text.json` store agent outputs when present.

These 14 runs are qualitative examples. The [evaluation CSV](../data/evaluations_processed.csv) covers 562 prompts, so most rows in that CSV do not have a corresponding image in this folder.
