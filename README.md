#  Gesture-Language-Dataset-and-Fusion-v2 - Model Training and Testing 

This repository includes a script for training and testing various model types on datasets related to object detection and action-object detection.

## Requirements

Ensure you have the necessary Python dependencies installed. The script assumes the presence of the following modules:

- `train.py` — contains the training logic
- `test.py` — contains the evaluation logic

You can install any dependencies via:

```bash
pip install -r requirements.txt
```


---

## Script Overview

The main logic is handled by the script (e.g., `main.py`) which accepts the following arguments:

- `--mode`: Specifies whether to `train` or `test`
- `--model_type`: (optional) Restricts the run to a single model type (e.g., `'mix'`)
- `extra`: Required in `test` mode — provides a result folder name to test against

### Example Command: Training

```bash
python main.py --mode train
```

This will:

- Iterate through predefined model types (`nn-1`, `mix`, `max`, `mul`, `sum`)
- Use scene complexities `[0, 1, 2, 3]` and fixed action complexity `"x"`
- Train each combination on the datasets in `data/object_detection/`
- Save results in automatically created folders under `results/`

### Example Command: Testing

```bash
python main.py --mode test 2025-05-25_15-00
```

> Replace `2025-05-25_15-00` with the name of the folder containing trained models.

This will:

- Load test data from the dataset
- Use the specified folder to locate trained models
- Save evaluation results to subfolders under `results/`

---

## Parameters in Script

The script runs combinations of:

- **Model Types**: `nn-1`, `mix`, `max`, `mul`, `sum`
- **Dataset Sizes**: `small`
- **Modality Types**: `all`
- **Scene Complexities**: `0`, `1`, `2`, `3`
- **Action Complexities**: `"x"`

These can be edited directly in the script if needed.

---

## Dataset Structure

There are currently two dataset types prepared:

1. **Object Detection**  
   Structure:

   ```
   data/
   └── object_detection/
       ├── small_sc_0_x/
       │   ├── train/
       │   ├── val/
       │   └── test/
       ├── small_sc_1_x/
       └── ...
   ```

2. **Action-Object Detection**  
   *(can be added or adapted similarly)*

---

## Example Dataset (from Master's Thesis)

You can download a full example dataset used for a Master's thesis from this Google Drive link:

📁 [Download Dataset](https://drive.google.com/drive/folders/1dxnH4PlPj3nGefn1aiMtTDP7xH23uFDT?usp=sharing)

Unpack it into the `data/` directory to match the structure above.

---

## Output Structure

Results are saved under:

```
results/
└── object_detection/
    ├── small_m_mix_mod_all_sc_0_ac_x/
    ├── small_m_max_mod_all_sc_1_ac_x/
    └── ...
```

Each subfolder will contain logs, metrics, and any other outputs (e.g., trained models, test results).

---

## Example Results Path for Test Mode

When running:

```bash
python main.py --mode test 2025-05-25_15-00
```

Make sure a directory like:

```
results/2025-05-25_15-00/
```

exists and contains the saved models from a training run.

---

## Notes

- This script assumes all required folders and datasets exist and are correctly named.
- If `--model_type` is omitted, all model types will be run.
- In `test` mode, the final argument (`extra`) must be provided — it tells the script where to find models to test.

---

Let us know if you'd like templates for `train.py`, `test.py`, or configuration examples.
