# GROUP45_CENG313 – Scientific Image Forgery Detection

CENG313 – Introduction to Data Science

## Group Information

| | |
|---|---|
| **Group ID** | 45 |
| **Member 1** | Mehmet Mete Köksal – 24118080026 – [@metekoksal](https://github.com/metekoksal) |
| **Member 2** | Abdullah Oğuzhan Turgut – 181180754 – [@apo1274](https://github.com/apo1274) |

## Competition

**[Recod.ai/LUC – Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)**

The task is to detect and segment **copy-move forgeries** in biomedical research images. For each image the model must either predict *authentic* or output one binary mask per forged region (instance). The competition metric is **instance-level F1 (oF1)**, averaged over images. A false alarm on an authentic image or a missed forgery both score 0 for that image.

## Current Scores

`Competition closed — evaluated on a held-out split.`

| Method | oF1 (validation, held-out half) | Authentic | Forged |
|---|---|---|---|
| Baseline 0: all authentic | 0.464 | 1.000 | 0.000 |
| Baseline 1: SIFT + RANSAC | 0.542 | 0.971 | 0.172 |
| Baseline 2: U-Net (ResNet34) | 0.528 | 0.975 | 0.142 |

Scores are computed with our implementation of the competition metric on ≈513 validation images that were **not** used for hyperparameter tuning. Differences of about 0.03 are within noise at this sample size.



## Repository Structure

```
GROUP45_CENG313/
├── README.md
├── requirements.txt
├── .gitignore                         
├── assignment1_eda_baseline.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py                        # data loading, dataset class, preprocessing
│   ├── models.py                      # SIFT detector, U-Net, training, evaluation
│   └── utils.py                       # plotting, metric helpers
├── tests/                             # test methods for evaluation control
│   ├── test_metrics.py                
|   └── test_postprocess.py
├── data/                              # not committed (Kaggle rules + size)
└── submissions/                       # summary table and score screenshots
```

## How to Download the Data

The dataset is not stored in this repository due to Kaggle rules and file size.

1. Join the competition on Kaggle and **accept the competition rules** (required to download).
2. Create a Kaggle API token (Kaggle → Settings → API).
3. Download and unzip:

```bash
pip install -q --upgrade kaggle
export KAGGLE_API_TOKEN=<your_token>
kaggle competitions download -c recodai-luc-scientific-image-forgery-detection -p data
unzip -q data/*.zip -d data
```

* In Google Colab the notebook does this automatically, add your token as a Colab secret named `KAGGLE_API_TOKEN`.

## How to Run the Notebooks

### Google Colab

1. Open `assignment1_eda_baseline.ipynb` in Colab.
2. Add the Kaggle token under **Secrets** as `KAGGLE_API_TOKEN` and enable notebook access.
3. Select a GPU runtime (**Runtime → Change runtime type → GPU**). U-Net training takes about 40 minutes on a GPU and is impractically slow on CPU.
4. **Runtime → Run all.** The first cells clone this repository, install `requirements.txt` and download the data.


Note: If GPU time is not available, download the trained weights (`unet_resnet34.pth`) from the repository's **Releases** page. Upload the file to Colab and switch the `CKPT` path in section 4.3 as the comment in the notebook describes. The training cell is then skipped. The loss curve cell only works if training was run in the same session.

Last version of the notebook can be viewed through GitHub.
