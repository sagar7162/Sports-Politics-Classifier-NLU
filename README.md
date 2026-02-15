# Sport vs Politics Text Classifier (Custom Features, 9 Experiments)

This project solves binary text classification:
- `sports`
- `politics`

The assignment constraint is respected:
- machine learning models can come from libraries,
- feature extraction (`BoW`, `TF-IDF`, `n-grams`) is implemented from scratch.

## Repository structure
- `code.ipynb`: main experiment pipeline (custom vectorizers + 9 model-feature combinations)
- `docs/index.md`: detailed report for GitHub Pages submission
- `toi_politics/headlines_politics.csv`: scraped politics headlines
- `indianexpress_sports/sports_headlines.csv`: scraped sports headlines
- `outputs/`: generated metrics, confusion matrices, top terms, and error analysis
- `requirements.txt`: dependencies

## Experimental matrix (3 x 3)
Feature methods:
1. `bow_unigram` (count-based unigram Bag of Words)
2. `tfidf_unigram` (unigram TF-IDF)
3. `ngram_count_1_2` (unigram + bigram counts)

Classifiers:
1. `decision_tree`
2. `random_forest`
3. `knn`

Total combinations: `9`

## Quick start (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook code.ipynb
```

Run all cells in order.

## Output artifacts
Running `code.ipynb` generates:
- `outputs/model_comparison_9combos.csv`
- `outputs/dataset_stats_custom.csv`
- `outputs/best_model_summary.csv`
- `outputs/best_model_errors.csv`
- `outputs/all_misclassifications.csv`
- `outputs/cm_<feature>__<classifier>.csv` (9 files)
- `outputs/top_terms_<feature>__<classifier>.csv` (9 files)

## Anti-plagiarism notes
- No usage of `CountVectorizer` / `TfidfVectorizer`.
- Vocabulary building, document frequency, TF-IDF weighting, and sparse matrix assembly are written manually in `code.ipynb`.
- Report text is tailored to the scraped dataset used in this repository.

## GitHub Pages
1. Push this folder to GitHub.
2. Open `Settings -> Pages`.
3. Choose `Deploy from a branch`.
4. Select `main` branch and `/docs` folder.
5. Save and wait for publish.
