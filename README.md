🎬 Engagement Prediction from Video & Audio

This project builds a machine learning pipeline to predict user engagement from short video recordings using audio + meta features.

🚀 What this project does

📌 Splits videos into labeled segments (high, mid, low, read, talk, idle)

🎧 Extracts audio features (MFCCs, spectral, ZCR, chroma — via Librosa)

🧠 Adds meta information (segment type, duration)

🤖 Trains ML models to predict engagement:

3-class → low / mid / high

Binary → high vs non-high (best-performing)

📥 Dataset Collection (Self-Recorded)

Unlike most projects that use public datasets, this dataset was collected manually.

6 real participants recorded ~3-minute videos each

Each recording contained structured segments (exciting video, boring video, reading aloud, personal talk, idle)

After every segment, participants self-reported their engagement on a 1–5 scale

📌 This made the project more realistic but also challenging due to:

Small dataset size (only 34 labeled segments)

Engagement is subjective, especially the mid class

Manual annotation and timestamp alignment effort

High variance between different individuals

➡ Despite these constraints, the ML pipeline successfully extracted meaningful engagement patterns.

📊 Results (Stratified 5-Fold CV)
Task	Best Model	Accuracy	Macro F1
3-Class (low/mid/high)	Random Forest	61.4%	0.42
Binary (high vs non-high)	XGBoost	65.7%	0.62

🎯 Binary classification performed better because the mid engagement level is highly subjective and difficult to learn from a limited dataset.

🗂 Project Structure
notebooks/
├── 01-feature-extraction.ipynb          # Extract audio + metadata features
├── 02-training-evaluation-multiclass.ipynb
└── 03-training-evaluation-binary.ipynb  # 🚀 Best-performing model

🔍 Key Points

✔ Small real dataset: 6 subjects, 34 total samples
✔ Labels based on self-reported engagement (1–5)
✔ Smart problem reframing: from noisy 3-class → more reliable binary
✔ Full ML pipeline demonstrated: data → features → training → evaluation

🧪 Tech Stack

Python · Scikit-Learn · XGBoost · Librosa · OpenCV · PCA · Jupyter