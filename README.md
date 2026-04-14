# kgsemembed

## 🔍 Overview

**kgsemembed** is a research-oriented Python framework for studying **semantic embedding-based entity matching in Knowledge Graphs (KGs)**.

This repository represents **Phase 2** of a multi-stage KG matching pipeline, focusing on **dense retrieval and verbalisation strategies** for improving semantic alignment.

It builds upon the findings of Phase 1 (lexical candidate generation) and investigates how embedding models and textual representations influence matching performance.

---

## 🎯 Objectives

- Evaluate embedding models for KG entity matching
- Study the impact of verbalisation strategies
- Analyse cross-domain robustness
- Measure efficiency vs performance trade-offs
- Provide a reproducible experimental framework

---

## 🧭 Pipeline Context

This work is part of a multi-stage KG matching pipeline:

1. Candidate Generation (Phase 1) → Character n-gram retrieval
2. Dense Retrieval (Phase 2 — This Repository) → Embedding-based filtering
3. LLM-based Re-ranking (Phase 3 — Future Work)

---

## 🧠 Core Research Questions

- How does embedding model choice affect performance?
- How does verbalisation impact embedding quality?
- What are the trade-offs between accuracy and efficiency?
- Which configurations generalise across domains?

---

## 🧾 Verbalisation Strategies

- V0: Label only
- V1: Label + normalized variants
- V2: Label + annotations
- V3: Local graph context
- V4: Structured template

---

## ⚙️ Installation

```bash
git clone https://github.com/swntsvt/kgsemembed.git
cd kgsemembed

python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install sentence-transformers transformers torch pandas numpy rdflib hydra-core pytest matplotlib seaborn
```

---

## 🚀 Usage

```bash
python -m kgsemembed.pipeline.run_experiment --config configs/experiment.yaml
```

---

## 📊 Evaluation Metrics

- Recall@k
- MRR
- Precision@k
- Runtime metrics

---

## 📁 Project Structure

```
kgsemembed/
├── src/kgsemembed/
├── configs/
├── tests/
├── data/
├── logs/
├── notebooks/
├── scripts/
```

---

## ⚡ Hardware Acceleration

```python
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

---

## 🧪 Testing

```bash
pytest
```

---

## 🧾 Logging

- Console + file logging
- Logs stored in logs/

---

## 📈 Expected Contributions

- Verbalisation impact analysis
- Cross-domain robustness
- Efficiency vs accuracy trade-offs

---

## 📄 License

MIT License

---

## 🧑‍💻 Author

Savita Kiran Sawant
