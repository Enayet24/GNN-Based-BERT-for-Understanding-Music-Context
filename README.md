# 🎵 GNN-Based BERT: Understanding Context from Music

> A Supervised Neural Network Project 

## 📘 Overview

This project implements a **hybrid BERT + Graph Neural Network (GNN) system** for understanding musical context, rather than generating it. The system combines two complementary neural architectures: **BERT**, for interpreting text such as mood tags and captions, and a **GNN**, for interpreting the structural relationships between segments of an audio track — how different parts of a song relate to each other over time and in terms of sound similarity.

The project follows a **task based roadmap**: a text-only BERT baseline, a graph-only GNN model, a GNN-BERT fusion model with a multi-task classification and a emotion-regression objective task.

The task specification and structure were provided as part of the course assignment. Dataset selection and the full implementation, training, and evaluation pipeline were done by our team.

---

## 🎯 Objectives

The main objectives of this project are:

- To classify a track's mood using text (tags/captions) via BERT
- To classify a track's mood using audio structure via a GNN over segment graphs
- To fuse text and audio representations and jointly predict mood tags and continuous emotion scores
- To align audio and text in a shared embedding space using contrastive learning
- To compare all of the above against non-graph and non-fusion baselines

---

## 🗂️ Dataset

- **Primary dataset: DEAM** — ~1,800 song excerpts annotated with continuous valence and arousal scores. Since DEAM has no discrete tags, we derived **4 mood-quadrant labels** (`happy_energetic`, `calm_content`, `sad_depressed`, `angry_tense`) by bucketing the valence-arousal plane, following the same idea used by the EmoMusic dataset.
- **Additional datasets: MusicCaps + FMA-small** — MusicCaps supplies real natural-language captions paired with audio; FMA-small supplies audio with genre labels used as weaker pseudo-captions, expanding the contrastive training pool.

---

## 🧹 Preprocessing

- Audio resampled to **22,050 Hz**
- **Chroma features** (12 bins) extracted per track using `librosa`
- Each track split into **8 fixed segments**, chroma-averaged per segment
- A **segment graph** built per track: edges connect temporally adjacent segments, plus segments above a cosine-similarity threshold
- DEAM's valence/arousal bucketed into **4 mood quadrants** to form classification labels and pseudo-captions

---

## 🧠 Tasks Implemented

### 🔹 Task 1 — BERT Baseline (Easy)

- `distilbert-base-uncased` encodes the mood-quadrant pseudo-caption
- `[CLS]` embedding → linear layer → sigmoid, trained with BCE loss
- Text-only, no audio input at all

### 🔹 Task 2 — GNN on Segment Graphs (Medium)

- Hand-written **GraphSAGE-style** layer (no external graph library): neighbor mean → concat with self → linear → ReLU, stacked twice, mean-pooled into a graph embedding
- Compared against a **CNN baseline** (2D conv net over the segment-feature grid) with no graph structure at all

### 🔹 Task 3 — GNN-BERT Fusion (Hard)

- Two fusion variants compared: **concatenation** and **cross-attention** (graph embedding attends over text embedding)
- Multi-task loss: mood-tag classification (BCE) + valence/arousal regression (MSE)

---

## 📈 Results Summary

| Model | Macro F1 | AUC-PR |
|---|---|---|
| Random baseline | ~0.05 | ~0.15 |
| CNN baseline (no graph) | 0.258 | 0.384 |
| BERT only (Task 1) | 1.000 | 1.000 |
| GNN only (Task 2) | 0.296 | 0.384 |
| Fusion, concatenation (Task 3) | 1.000 | 1.000 |
| Fusion, attention (Task 3) | 1.000 | 1.000 |

**Emotion regression (Task 3):** MAE **0.114**, R² **0.107**

📌 The GNN clearly outperforms the CNN baseline (0.296 vs. 0.258), which is the fairest evidence that graph structure helps. The perfect BERT-only / fusion scores are flagged, not celebrated — see Limitations below.

---

## ⚠️ Limitations & Known Issues

- **Label leakage in the text pathway** — the pseudo-captions are built directly from the same mood-quadrant label being predicted (e.g. caption = `"happy energetic"`, label = `happy_energetic`). This lets BERT solve the task by matching words rather than genuinely understanding mood, which explains the perfect Task 1 / Task 3 scores and means the fusion-vs-baseline comparison isn't fair yet. The GNN-only vs. CNN-baseline comparison doesn't have this shortcut and is more trustworthy.
- **Task 4 data loading issue** — our FMA-small and MusicCaps loaders returned 0 usable tracks due to metadata/audio-path mismatches in our Colab setup, so Task 4 fell back to an unintended dataset. Contrastive loss and Recall@K are not reported. This is an open issue to fix before final submission.

---

## 🛠 Tech Stack

- **PyTorch**
- **Hugging Face Transformers** (`distilbert-base-uncased`)
- **librosa** (audio feature extraction)
- **scikit-learn** (metrics, t-SNE)
- **pandas / numpy**
- Developed and run on **Google Colab**

---

## 📄 Project Report

A complete **IEEE-format report** covering methodology, results, and limitations is included in this repository.

📌 **Read the full report here:** [`report/CSE425_report.tex`](./report/CSE425_report.tex)

---

## 🎓 Key Learning Outcomes

Through this project, the following were achieved:

- Implementing a GNN from scratch (no external graph library)
- Combining transformer-based text encoders with graph-based audio encoders
- Multi-task learning across classification and regression objectives
- Contrastive representation learning for cross-modal retrieval
- Diagnosing label leakage and dataset-loading issues in a real pipeline

---

## 👥 Project Team

This project was developed as part of our **Neural Networks (CSE425)** course.

Team Members:

- [Enayet Ali Labib](https://github.com/Enayet24)
- Iffat Islam Aria
