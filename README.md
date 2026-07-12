
# Data Classification Using AI — KNN on the Iris Dataset

A basic supervised learning project that builds, trains, and evaluates a K-Nearest Neighbors (KNN) classifier on the classic Iris flower dataset.

## Project Goal

Build a basic classification model using a small dataset, covering the full ML workflow: load data, understand it, split it, train a model, and evaluate results.

## Key Requirements Covered

- ✅ Load and understand a dataset
- ✅ Split data into training and testing sets
- ✅ Apply a simple classification algorithm (KNN)

## Key Skills Demonstrated

- Data handling with `pandas`
- Supervised learning basics
- Model training and evaluation with `scikit-learn`

## Dataset

The [Iris flower dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset), built into scikit-learn:
- 150 samples
- 4 numeric features (sepal length/width, petal length/width, in cm)
- 3 target classes (*setosa*, *versicolor*, *virginica*)

## Project Workflow

1. **Load & explore** — reads the dataset into a pandas DataFrame, checks shape, data types, summary statistics, class balance, and missing values. Generates a pairplot of feature relationships.
2. **Split & scale** — splits data 80/20 into train/test sets (stratified to keep class balance), then standardizes features with `StandardScaler` (fit only on training data to avoid data leakage).
3. **Train** — fits a `KNeighborsClassifier` (k=5) on the training set.
4. **Evaluate** — reports accuracy, a full classification report (precision/recall/F1), and a confusion matrix heatmap.
5. **Tune k (bonus)** — tests k values from 1–15 and plots accuracy vs. k to find the best-performing value.
6. **Predict** — classifies one new, unseen flower sample.

## Results

- Test accuracy at k=5: **93.3%**
- Best k found: **k=1** (96.7% accuracy)

## Project Structure

```
.
├── knn_iris_classification.py   # Main script
├── requirements.txt             # Python dependencies
├── 01_dataset_pairplot.png      # Feature relationship plot (generated on run)
├── 02_confusion_matrix.png      # Confusion matrix heatmap (generated on run)
├── 03_k_vs_accuracy.png         # Accuracy vs. k plot (generated on run)
└── README.md
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/data-classification-knn.git
   cd data-classification-knn
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:
   ```bash
   python knn_iris_classification.py
   ```

The script prints dataset exploration details, training/test split info, model accuracy, and a classification report to the console, and saves three plots to the project folder.

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Author

Roshan — B.Tech Information Technology, Guru Gobind Singh Indraprastha University
