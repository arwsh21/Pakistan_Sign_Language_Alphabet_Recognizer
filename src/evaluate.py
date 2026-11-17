"""Experiments tables and charts."""

import os
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import learning_curve, train_test_split

# Ensure root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collect import SIGNS
from src.config import RAW_DATA_PATH, SPLIT_SEEDS, TEST_SIZE
from src.features import load_dataset
from src.models import (
    BaselineRecognizer,
    ForestRecognizer,
    KNNRecognizer,
    SVMRecognizer,
)


def run_experiments(X, y, seeds=SPLIT_SEEDS):
  model_names = ["Baseline", "KNN", "SVM", "Random Forest"]
  raw_results = {
      name: {"acc": [], "f1": [], "train_t": [], "pred_t": []}
      for name in model_names
  }

  print(f"Running experiments across {len(seeds)} seeded splits...")

  for seed in seeds:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    models = {
        "Baseline": BaselineRecognizer(),
        "KNN": KNNRecognizer(),
        "SVM": SVMRecognizer(),
        "Random Forest": ForestRecognizer(),
    }

    for name, clf in models.items():
      # Time training
      t0 = time.perf_counter()
      clf.fit(X_train, y_train)
      train_time = time.perf_counter() - t0

      # Time prediction (run once)
      t0 = time.perf_counter()
      y_pred = clf.predict(X_test)
      pred_time = (time.perf_counter() - t0) / len(X_test)

      # Evaluate directly from y_pred
      acc = accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred, average="macro")

      raw_results[name]["acc"].append(acc)
      raw_results[name]["f1"].append(f1)
      raw_results[name]["train_t"].append(train_time)
      raw_results[name]["pred_t"].append(pred_time)

  # Compute sample statistics (ddof=1)
  summary_rows = []
  for name in model_names:
    acc_mean = np.mean(raw_results[name]["acc"]) * 100
    acc_std = np.std(raw_results[name]["acc"], ddof=1) * 100

    f1_mean = np.mean(raw_results[name]["f1"]) * 100
    f1_std = np.std(raw_results[name]["f1"], ddof=1) * 100

    train_mean = np.mean(raw_results[name]["train_t"]) * 1000
    pred_mean = np.mean(raw_results[name]["pred_t"]) * 1000000

    summary_rows.append({
        "Model": name,
        "Accuracy (Mean ± Std %)": f"{acc_mean:.2f} ± {acc_std:.2f}%",
        "Macro F1 (Mean ± Std %)": f"{f1_mean:.2f} ± {f1_std:.2f}%",
        "Train Time (ms)": f"{train_mean:.2f}",
        "Pred Time (μs/sample)": f"{pred_mean:.2f}",
    })

  summary_df = pd.DataFrame(summary_rows)

  os.makedirs("outputs/results", exist_ok=True)
  output_csv = "outputs/results/model_comparison.csv"
  summary_df.to_csv(output_csv, index=False)
  print(f"Saved results table to: {output_csv}\n")

  return summary_df, raw_results


def plot_accuracy_comparison(
    raw_results, output_path="outputs/figures/model_accuracy_comparison.png"
):
  """Chart 1: Bar chart comparing model accuracies with sample std error bars."""
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  models = list(raw_results.keys())
  means = [np.mean(raw_results[m]["acc"]) * 100 for m in models]
  stds = [np.std(raw_results[m]["acc"], ddof=1) * 100 for m in models]

  plt.figure(figsize=(8, 5))
  bars = plt.bar(
      models,
      means,
      yerr=stds,
      capsize=6,
      color=["#95a5a6", "#3498db", "#2ecc71", "#e67e22"],
      edgecolor="black",
  )
  plt.ylabel("Accuracy (%)", fontsize=12)
  plt.title(
      "Model Accuracy Comparison over 20 Seeded Splits",
      fontsize=14,
      fontweight="bold",
  )
  plt.ylim(0, 110)
  plt.grid(axis="y", linestyle="--", alpha=0.7)

  for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        h + 2,
        f"{h:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

  plt.tight_layout()
  plt.savefig(output_path, dpi=300)
  plt.close()
  print(f"Chart 1 saved to: {output_path}")


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    labels,
    output_path="outputs/figures/confusion_matrix.png",
):
  """Chart 2: Heatmap showing predictions across all 10 PSL signs."""
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  y_pred = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred, labels=labels)

  plt.figure(figsize=(9, 7))
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      xticklabels=labels,
      yticklabels=labels,
  )
  *
  plt.xlabel("Predicted Sign", fontsize=12, fontweight="bold")
  plt.ylabel("True Sign", fontsize=12, fontweight="bold")
  plt.title("Confusion Matrix (SVM)", fontsize=14, fontweight="bold")

  plt.tight_layout()
  plt.savefig(output_path, dpi=300)
  plt.close()
  print(f"Chart 2 saved to: {output_path}")


def plot_learning_curve(
    model, X, y, output_path="outputs/figures/learning_curve.png"
):
  """Chart 3: Shows learning progression vs dataset size."""
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  train_sizes, train_scores, test_scores = learning_curve(
      model.model,
      X,
      y,
      cv=5,
      train_sizes=np.linspace(0.2, 1.0, 5),
      random_state=42,
  )
  train_mean = np.mean(train_scores, axis=1) * 100
  test_mean = np.mean(test_scores, axis=1) * 100

  plt.figure(figsize=(8, 5))
  plt.plot(train_sizes, train_mean, "o-", color="#e74c3c", label="Training Score")
  plt.plot(
      train_sizes,
      test_mean,
      "s-",
      color="#2ecc71",
      label="Cross-Validation Score",
  )

  plt.xlabel("Training Examples Used", fontsize=12, fontweight="bold")
  plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
  plt.title(
      "Learning Curve (Generalization vs Sample Count)",
      fontsize=14,
      fontweight="bold",
  )
  plt.legend(loc="lower right")
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.ylim(0, 105)

  plt.tight_layout()
  plt.savefig(output_path, dpi=300)
  plt.close()
  print(f"Chart 3 saved to: {output_path}")


if __name__ == "__main__":
  try:
    X, y = load_dataset(RAW_DATA_PATH)
  except FileNotFoundError as err:
    print(f"\n[ERROR] {err}")
    print(
        "Hint: Run 'python src/collect.py' to record dataset before running"
        " evaluation."
    )
    sys.exit(1)

  # 1. Run 20-seed experiment
  df_results, raw_results = run_experiments(X, y)
  print(df_results.to_string(index=False))

  # 2. Chart 1: Accuracy Bar Chart
  plot_accuracy_comparison(raw_results)

  # 3. Chart 2: Confusion Matrix
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=TEST_SIZE, random_state=42, stratify=y
  )
  svm = SVMRecognizer()
  svm.fit(X_train, y_train)
  plot_confusion_matrix(svm, X_test, y_test, labels=SIGNS)

  # 4. Chart 3: Learning Curve
  plot_learning_curve(svm, X, y)

  print("\nDone! All 3 figures generated in outputs/figures/")