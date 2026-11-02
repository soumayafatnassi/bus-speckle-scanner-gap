# Speckle reduction and the scanner gap in breast ultrasound classification

Code for the article *"Speckle reduction does not close the scanner gap in breast ultrasound lesion classification: a leakage-controlled multi-scanner study"* (Soumaya Fatnassi, King Faisal University; submitted).

The study asks whether classical speckle reduction helps benign–malignant classifiers of breast lesions transfer to ultrasound scanners and hospitals not seen in training. Four filters (median, Lee, SRAD and BM3D for correlated noise), each tuned to the speckle grain of every image, are compared with unfiltered images. Two networks (ResNet-18 and EfficientNet-B0) are trained on one hospital, or on one scanner within BUS-BRA, and tested on the others under patient-grouped splits.

## Data

The images are not redistributed here. Notebook 01 downloads them from their public sources:

| Dataset | Scanner(s) | Source | Licence |
| --- | --- | --- | --- |
| BUSI (Al-Dhabyani et al., 2020) with the Curated BUSI mapping (Aumente-Maestro et al., 2025) | GE LOGIQ E9 | Kaggle `aryashah2k/breast-ultrasound-images-dataset`; mapping: [Zenodo 10.5281/zenodo.19047974](https://doi.org/10.5281/zenodo.19047974) | see source |
| BUS-BRA (Gómez-Flores et al., 2024) | GE Logiq 5, GE Logiq 7, Toshiba Aplio 300, U-Systems | [Zenodo 10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412) | CC BY 4.0 |
| BUS-UCLM (Vallez et al., 2025) | Siemens ACUSON S2000 | [Mendeley Data 10.17632/7fvgj4jsp7.3](https://doi.org/10.17632/7fvgj4jsp7.3) (Kaggle mirror `orvile/bus-uclm-breast-ultrasound-dataset`) | see source |

After exclusions the study uses 2488 lesion images: Curated BUSI 386, BUS-BRA 1875 and BUS-UCLM 227.

## How to run

The notebooks were written for Google Colab and save everything to Google Drive (`MyDrive/bus_crossscanner` by default; set `BUS_PROJECT_DIR` to change it). Run them in order. Each one resumes where it stopped if Colab disconnects.

| Notebook | Runtime | Approx. time | Main outputs |
| --- | --- | --- | --- |
| `01_prepare_data_and_speckle` | CPU | 15–30 min | `tables/metadata_all.csv`, lesion crops in `work/`, `tables/speckle_stats_raw.csv` (Table 2) |
| `02_filter_lesion_crops` | CPU | 1–2 h | `filtered_v2/{raw,median,lee,srad,bm3d}.npz`, `tables/nr_metrics_summary.csv` (Table 3), `figures/filters_examples.png` (Figure 1) |
| `03_train_resnet18` | T4 GPU | 1–2 h | `results_week2/predictions/`, `results_week2/splits.csv`, `results_week2/domain_gap.csv` |
| `04_train_efficientnet_b0` | T4 GPU | about 2 h | `results_week2_efficientnet_b0/predictions/` |
| `05_statistics_and_figures` | CPU | a few minutes | `results_week3/table_auc_ci.csv` (Tables S1–S2), `figures/forest_delta_auc.png` (Figure 2), `figures/dissociation.png` (Figure 3), `key_numbers.json` |

Each notebook is provided both as `.ipynb` and as a plain Python script in percent format (`.py`, readable with [jupytext](https://jupytext.readthedocs.io)). The scripts also run outside Colab with the packages in `requirements.txt`.

## Reproducibility notes

- Patient-grouped splits are created once (`results_week2/splits.csv`) and shared by all filters and both networks, so every comparison with raw images is paired.
- Training seeds are fixed per fold. The patient-level bootstrap in notebook 05 uses 10 000 resamples with a seed fixed for each target, so the reported intervals and p-values are exactly reproducible from the saved predictions.
- Lee and SRAD work in the intensity domain after undoing log compression with an assumed dynamic range of 50 dB (`DISPLAY_DR_DB`).
- Tested with Python 3.12–3.13, NumPy 2.x, PyTorch 2.x and bm3d 4.0.3. Recent NumPy removed `np.trapz`, which bm3d still calls; notebook 02 adds the needed alias.

## Citation

If you use this code, please cite the article (full reference to be added on publication).

## Licence

The code is released under the MIT licence (see `LICENSE`). The datasets keep their own licences and terms of use.
