# Neural Value Alignment

Code accompanying **Neural Value Alignment: Human–AI Collaboration Under Goal–Action Ambiguity**.

**Xin Xu, Yansen Wang, Dongqi Han, Dongsheng Li, and Sang Wan Lee**  
*IEEE Transactions on Cybernetics*, 2026.  
[Paper](https://ieeexplore.ieee.org/abstract/document/11663526) · [DOI: 10.1109/TCYB.2026.3722605](https://doi.org/10.1109/TCYB.2026.3722605) · [Code Ocean capsule, version 2](https://codeocean.com/capsule/6361644/tree/v2)

## Overview

Neural value alignment (NVA) investigates how reward prediction error (RPE) and state prediction error (SPE) can inform human–AI collaboration when goals cannot be unambiguously inferred from actions. This repository contains behavioral and EEG analysis notebooks, EEG decoder code, and visualization of simulation results associated with the paper.

This is a **code-only snapshot of Code Ocean capsule 6361644, version 2**. It does not include the capsule's Git history, research data, precomputed analysis results, or trained model files. The Code Ocean v2 link above identifies the published capsule version associated with this snapshot.

## Data availability

**Data are not included in this repository.** To request access, please email the first author or the corresponding author using the contact details in the publication.

- **First author:** Xin Xu — [sarahxu0811@kaist.ac.kr](mailto:sarahxu0811@kaist.ac.kr)
- **Corresponding author:** Sang Wan Lee — please refer to the publication for contact details.

When requesting data, please indicate which analyses or figures you would like to reproduce. The notebooks require processed inputs and, for several figures, precomputed decoder or simulation results. These inputs must be obtained separately; running the repository without them will produce missing-file errors.

## Repository contents

| Location | Purpose |
| --- | --- |
| `code/behavioral_analysis_fig3/fig_beh.ipynb` | Behavioral analyses and Figure 3 |
| `code/eeg_analysis_fig4/fig_psd_onlyForVis.ipynb` | EEG power-spectrum analyses and Figure 4 visualization |
| `code/eeg_decoding_analysis_fig5/fig_gradcam_onlyForVis.ipynb` | Decoder evaluation and Grad-CAM analyses for Figure 5 |
| `code/nva_simulation_fig6789/fig_model_comparison_onlyForVis.ipynb` | Visualization and comparison of precomputed simulation results for Figures 6–9 |
| `code/eeg_decoder/` | EEG decoder training script, dataset handling, configuration, and bundled supporting modules |
| `code/run` | Executes the four figure notebooks and exports HTML reports |
| `environment/` | Exported Code Ocean Dockerfile and post-install script |
| `metadata/metadata.yml` | Original capsule metadata |
| `REPRODUCING.md` | Original Code Ocean instructions and archived v2 image reference |

The simulation notebook loads saved simulation results; it is not a standalone implementation for regenerating all simulation inputs. The default `code/run` script executes the figure notebooks, not decoder training.

## Reproducing the figure notebooks

### 1. Obtain the code and required inputs

```bash
git clone https://github.com/Sarah-Xu-08/neural-value-alignment.git
cd neural-value-alignment
```

After obtaining the required inputs from the authors, place them under `data/` at the repository root, preserving the supplied filenames and directory structure. The notebooks reference these locations:

```text
neural-value-alignment/
├── code/
├── data/                         # obtained separately; not tracked by Git
│   ├── behavior/
│   ├── eeg/
│   ├── decoder_gradCAM/
│   ├── decoder_results/
│   └── simulation_results/
│       └── results_itrs_challenge/
├── environment/
└── results/                      # generated locally; not tracked by Git
```

Creating empty directories does not supply the required inputs. The notebooks use relative paths such as `../../data/eeg`; keep the directory layout intact.

### 2. Use the archived Code Ocean v2 environment

Install and start Docker. From the repository root, run the command supplied with the capsule:

```bash
mkdir -p results
docker run --platform linux/amd64 --rm \
  --workdir /code \
  --volume "$PWD/data":/data \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  registry.codeocean.com/published/6f230e9d-a2b3-4aa0-b6d3-4555c5ecfe73:v2 \
  bash run
```

This uses the archived Linux AMD64 environment and exports executed notebook HTML files to `results/`. On Apple Silicon, Docker must support AMD64 emulation. The archived environment is the reference for this snapshot; `environment/Dockerfile` records its exported build recipe.

**Check the generated reports for errors.** The original runner passes `--ExecutePreprocessor.allow_errors=True`, so a successful shell exit does not establish that every notebook cell completed successfully. This behavior is preserved from v2.

The exported `code/requirements.txt` uses single-equals package specifications. It is retained for provenance and should not be used directly with `pip install -r`. Consult the archived image and Dockerfile for the environment rather than assuming a fresh local installation will reproduce it.

### 3. EEG decoder training (separate workflow)

The training entry point is `code/eeg_decoder/run_binary_choice.py`. Before running it:

1. Obtain the required preprocessed EEG inputs.
2. Update `data.root_path` in `config_binary_choice.yml` to your local EEG data location; the exported value is `/root/capsule/data/eeg`.
3. Use a compatible Python environment with the required dependencies and a CUDA-capable GPU. The script calls `.cuda()` directly.
4. Review the subject, prediction-error condition, and hyperparameters in the script. The snapshot explicitly selects subject `03`, condition `srpe3`, and five folds; some YAML settings are overridden in the script.

From the decoder directory, the entry-point command is:

```bash
cd code/eeg_decoder
python run_binary_choice.py config_binary_choice.yml
```

Decoder training is not part of the default figure-notebook run. This code-only distribution has not been validated by retraining models or rerunning the data-dependent analyses.

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@article{xu2026neural,
  title   = {Neural Value Alignment: Human--AI Collaboration Under Goal--Action Ambiguity},
  author  = {Xu, Xin and Wang, Yansen and Han, Dongqi and Li, Dongsheng and Lee, Sang Wan},
  journal = {IEEE Transactions on Cybernetics},
  year    = {2026},
  doi     = {10.1109/TCYB.2026.3722605}
}
```

Machine-readable citation information is available in [CITATION.cff](CITATION.cff).

## License

The capsule includes the [MIT License](LICENSE), copyright © 2025 Xin Xu; the original copy is also retained at `code/LICENSE`. Retain applicable notices in bundled dependencies. The software license does not grant access to the research data.
