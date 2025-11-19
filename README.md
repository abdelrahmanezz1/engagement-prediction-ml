\# 🎬 Engagement Prediction from Video \& Audio



This project builds a machine learning pipeline to predict \*\*user engagement\*\* from short video recordings using \*\*audio + video features\*\*.



---



\## 🚀 What this project does



\- Splits videos into labeled segments (high, mid, low, read, talk, idle)

\- Extracts features:

&nbsp; - 🎧 Audio (MFCCs, spectral, ZCR, chroma via Librosa)

&nbsp; - 🧠 Meta info (segment type, duration)

\- Trains ML models to predict engagement:

&nbsp; - 3-class: \*\*low / mid / high\*\*

&nbsp; - Binary: \*\*high vs non-high\*\* (best)



---



\## 📊 Results (Stratified 5-Fold CV)



| Task                     | Best Model     | Accuracy | Macro F1 |

|--------------------------|----------------|----------|----------|

| 3-Class (low/mid/high)   | Random Forest  | \*\*61.4%\*\* | 0.42     |

| Binary (high vs non-high)| XGBoost        | \*\*65.7%\*\* | \*\*0.62\*\* |



➡ Binary works better because \*\*“mid” engagement is subjective\*\* and hard to learn with few samples.



---



\## 🗂 Project structure



```text

notebooks/

├── 01-feature-extraction.ipynb          # Extract audio + video features

├── 02-training-evaluation-multiclass.ipynb

└── 03-training-evaluation-binary.ipynb  # Best-performing model





🔍 Key points



&nbsp;-Small real dataset: 6 subjects, ~34 labeled segments



&nbsp;-Engagement labels are self-reported  (1–5)



&nbsp;-Smart reframing: from noisy 3-class →  more reliable binary



&nbsp;-Shows full pipeline: data → features → models → evaluation



🧪Tech stack



Python · Scikit-Learn · XGBoost · Librosa · OpenCV · PCA · Jupyter

