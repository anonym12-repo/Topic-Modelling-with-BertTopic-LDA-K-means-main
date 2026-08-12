# 🦠 Topic Modelling with BERTopic, LDA & K-Means
### Comparing Transformer-Based and Classical NLP Approaches on Noisy, Short-Form Social Media Text

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![BERTopic](https://img.shields.io/badge/BERTopic-NLP-orange.svg)](https://maartengr.github.io/BERTopic/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

## 📌 Overview

This project analyzes **~2,600 tweets from December 2021** discussing the Omicron COVID-19 variant to uncover the dominant themes in public discourse. Rather than applying a single modelling technique, I built a **comparative NLP pipeline** benchmarking three distinct approaches — **BERTopic**, **Latent Dirichlet Allocation (LDA)**, and **K-Means clustering** — to evaluate how each handles the unique challenges of short, informal, noisy text (slang, emojis, sparse word co-occurrence).

The goal: determine which method best extracts coherent, interpretable topics from social media data, and quantify *why*.

**🏆 Result:** BERTopic, powered by transformer-based sentence embeddings and UMAP dimensionality reduction, significantly outperformed the classical baselines — achieving the highest topic coherence score and producing the most semantically distinct, human-interpretable clusters.

## 🎯 Why This Project Matters

Social media analysis is a high-value, real-world NLP problem faced across marketing, public health, and crisis-response teams. This project demonstrates:

- **End-to-end ML pipeline design** — from raw, messy data to evaluated, visualized insights
- **Model comparison rigor** — not just building one model, but designing a fair, evidence-based benchmark across three methodologically different techniques
- **Data-driven decision-making** — using coherence scores, intertopic distance maps, and hierarchical clustering visualizations to justify hyperparameter choices (e.g., reducing topic count from ambiguous overlapping clusters down to 5 well-separated topics)
- **Communication of technical results** — findings are backed by a full written report with cited academic sources

## 🧠 Methodology

**Pipeline:** `Raw Tweets → Preprocessing → Feature Extraction → Topic Modelling → Clustering → Evaluation`

| Stage | Techniques Used |
|---|---|
| **Data Sourcing** | Kaggle "Omicron Tweets" dataset (8,000+ tweets), filtered to Dec 2021 for temporal relevance |
| **Preprocessing** | Deduplication, null removal, lowercasing, URL/emoji/mention/hashtag/punctuation stripping, lemmatization |
| **Feature Extraction** | TF-IDF (for LDA & K-Means) and `SentenceTransformer` embeddings — `all-mpnet-base-v2` / `all-MiniLM-L6-v2` (for BERTopic) |
| **Dimensionality Reduction** | UMAP (2D projection, tuned `n_neighbors` and `min_dist` for cluster separation) |
| **Clustering** | HDBSCAN (density-based, for BERTopic), K-Means (distance-based baseline) |
| **Topic Refinement** | Bigram n-gram range (1,2) to preserve meaningful phrases (e.g. *"vaccination rate"*, *"travel ban"*); topic count reduced to 5 based on coherence and redundancy analysis |
| **Evaluation** | Quantitative: **Coherence Score (C_v)** across all three models. Qualitative: intertopic distance maps, hierarchical dendrograms, word clouds, PCA scatter plots |

## 📊 Key Findings

- **BERTopic** achieved the highest coherence score and cleanest topic separation, correctly isolating themes like vaccine discourse (~71% of tweets), international travel restrictions, and geopolitical debate — validated against peer-reviewed literature on transformer-based topic modelling for short text.
- **LDA** struggled with overlapping, poorly separated topics — a known limitation on sparse, short-text data.
- **K-Means** (TF-IDF-based) provided an interpretable but shallow baseline; switching its input to BERT embeddings noticeably improved semantic grouping, reinforcing the value of contextual embeddings over raw word frequency.

*Full write-up, figures (intertopic distance maps, word clouds, coherence tables), and academic citations are available in [`Documentation.docx`](./Documentation.docx).*

## 🛠️ Tech Stack

`Python` · `BERTopic` · `Sentence-Transformers` · `UMAP` · `HDBSCAN` · `Gensim (LDA)` · `Scikit-learn (K-Means, TF-IDF, PCA)` · `NLTK` · `Matplotlib` / `Seaborn` · `WordCloud` · `pyLDAvis`

## 📁 Repository Structure

```
├── code.ipynb              # Full analysis notebook (preprocessing → modelling → evaluation)
├── Documentation.docx      # Written report with methodology, figures, and references
├── omicron_tweets.csv      # Source dataset (Omicron-related tweets, Dec 2021)
└── README.md
```

## 🚀 Running the Project

```bash
git clone https://github.com/<your-username>/Topic-Modelling-with-BertTopic-LDA-K-means.git
cd Topic-Modelling-with-BertTopic-LDA-K-means
pip install pandas numpy matplotlib seaborn nltk scikit-learn bertopic sentence-transformers wordcloud gensim pycountry pyLDAvis
jupyter notebook code.ipynb
```

## 🔮 Future Work

- Extend to **multilingual tweet analysis** for cross-cultural insight
- Integrate **real-time topic tracking** to monitor emerging public health narratives as they develop
- Explore **sentiment-aware topic modelling** to pair *what* people are discussing with *how* they feel about it

## 📄 License

This project is available under the MIT License.

---

*This project was built as an independent research exercise in applied NLP and unsupervised learning.*
