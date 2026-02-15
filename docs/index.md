# Sport vs Politics Text Classifier
## A custom-feature binary text classification system (NLU Assignment)

## 1. Abstract
This project builds a binary text classifier that predicts whether a news headline belongs to `sports` or `politics`. The system uses data scraped from two real news sources and compares three machine learning classifiers over three feature representation methods, creating a full 3 x 3 matrix of experiments.

The complete implementation is in `code.ipynb`, and generated quantitative outputs are stored in `outputs/` for direct use in submission and GitHub Pages.

## 2. Problem Statement
Given a text document (here, a news headline), classify it into one of two classes:
1. `sports`
2. `politics`

The assignment requires:
1. At least three machine learning techniques.
2. Feature representation using n-grams / TF-IDF / Bag of Words.
3. Detailed analysis including data collection, dataset description, quantitative comparison, and limitations.
4. A GitHub page with complete project details.

This submission addresses all four requirements.

## 3. Data Collection
### 3.1 Sources
The dataset is scraped from two websites:
1. Times of India Politics section (politics headlines)
2. Indian Express Sports section (sports headlines)

Stored files:
1. `toi_politics/headlines_politics.csv`
2. `indianexpress_sports/sports_headlines.csv`

Each file has columns:
1. `headline`
2. `category`

### 3.2 Collection approach
Scraping code (already present in earlier notebook work) collects headline text from listing pages and stores each headline with a category label. This creates a directly supervised dataset suitable for binary classification.

### 3.3 Why this dataset is appropriate
1. Labels are explicit and semantically clear.
2. Headlines are short, realistic, and noisy enough to test practical NLP behavior.
3. Sports and politics have overlapping vocabulary at times, making the task non-trivial.

## 4. Dataset Description and Analysis
### 4.1 Raw dataset size
Based on current CSV files in this repository:
1. Politics rows: 1450
2. Sports rows: 1475
3. Total rows: 2925

### 4.2 Data quality observations
1. Missing headlines: none observed in raw files.
2. Duplicate headlines exist in both classes (more in sports than politics).
3. Some sports text contains mojibake artifacts (`â€™`, `Â`, etc.), handled in preprocessing.

### 4.3 Cleaning decisions
1. Lowercase normalization.
2. Basic encoding artifact replacement.
3. Tokenization with regex.
4. Stopword removal.
5. Remove empty text rows.
6. Remove duplicate `(clean_text, category)` pairs.

### 4.4 Post-cleaning analysis (from notebook output)
`code.ipynb` computes and saves:
1. Class counts after cleaning.
2. Token length statistics by class.
3. Average and median tokens per headline.

These values are exported to:
1. `outputs/dataset_stats_custom.csv`

## 5. Feature Representation
A custom `ScratchVectorizer` class is implemented in `code.ipynb`.

### 5.1 Tokenization
Regex token pattern:
1. alphanumeric tokens
2. optional apostrophe continuation (for words like `don't`)

Stopwords are removed using a manually defined set.

### 5.2 Bag of Words (unigram counts)
For each document:
1. Generate unigram tokens.
2. Count term frequencies.
3. Build sparse vector over learned vocabulary.

### 5.3 TF-IDF (unigram)
Same unigram vocabulary, with TF-IDF weighting:
1. TF: raw count or sublinear term frequency (`1 + log(tf)`).
2. IDF: `log((1 + N)/(1 + df)) + 1`.
3. Final weight: `TF * IDF`.

### 5.4 n-gram counts (1,2)
Features include:
1. Unigrams
2. Bigrams (joined with underscore)

This representation captures short phrase patterns that pure unigrams can miss.

### 5.5 Vocabulary control
To reduce noise and dimensional explosion:
1. `min_df` threshold is applied.
2. `max_features` cap is applied.
3. Terms are sorted by document frequency then lexicographic order.

## 6. Machine Learning Techniques
Three classifiers are imported from libraries (allowed by assignment):
1. Decision Tree (`DecisionTreeClassifier`)
2. Random Forest (`RandomForestClassifier`)
3. K-Nearest Neighbors (`KNeighborsClassifier`)

### 6.1 Why these models
1. They are standard, strong baselines for sparse text features.
2. They train quickly on moderate-size text datasets.
3. They allow interpretable feature-level analysis.

## 7. Experimental Setup
### 7.1 Train-test split
1. Stratified split: 80% train, 20% test.
2. Random seed fixed (`SEED = 50`) for reproducibility.

### 7.2 Full comparison matrix
Feature methods:
1. `bow_unigram`
2. `tfidf_unigram`
3. `ngram_count_1_2`

Classifiers:
1. `decision_tree`
2. `random_forest`
3. `knn`

Total runs:
1. 9 combinations

### 7.3 Evaluation metrics
1. Accuracy
2. Precision (sports positive class)
3. Recall (sports positive class)
4. F1-score (sports positive class)
5. Training time
6. Inference time
7. Confusion matrix

## 8. Quantitative Results
After running all notebook cells, the main table is generated at:
1. `outputs/model_comparison_9combos.csv`

| Feature Method | Classifier | Accuracy | Precision (sports) | Recall (sports) | F1 (sports) | Train Time (s) | Inference Time (s) | Vocab Size |
|---|---|---:|---:|---:|---:|---:|---:|---:|
ngram_count_1_2|knn|0.9671848013816926|0.9651567944250871|0.9685314685314685|0.9668411867364747|0.0013229000005594571|0.033539899999595946|4848
bow_unigram|knn|0.9671848013816926|0.968421052631579|0.965034965034965|0.9667250437828371|0.0011353999998391373|0.035032400000091|2975
tfidf_unigram|random_forest|0.9671848013816926|0.989010989010989|0.9440559440559441|0.9660107334525939|0.45828200000050856|0.0616016000003583|2975
ngram_count_1_2|random_forest|0.9671848013816926|0.992619926199262|0.9405594405594405|0.9658886894075404|0.3798679000001357|0.03618629999982659|4848
tfidf_unigram|knn|0.9654576856649395|0.9683098591549296|0.9615384615384616|0.9649122807017544|0.0011830000003101304|0.04190839999955642|2975
bow_unigram|random_forest|0.9654576856649395|0.9889705882352942|0.9405594405594405|0.96415770609319|0.2814732999995613|0.03646059999937279|2975
bow_unigram|decision_tree|0.8981001727115717|0.8388059701492537|0.9825174825174825|0.9049919484702094|0.03507550000085757|0.0004144000004089321|2975
tfidf_unigram|decision_tree|0.8981001727115717|0.8388059701492537|0.9825174825174825|0.9049919484702094|0.03730770000038319|0.0005271999998512911|2975
ngram_count_1_2|decision_tree|0.8946459412780656|0.8338278931750742|0.9825174825174825|0.9020866773675762|0.04205950000050507|0.0005076000006738468|4848

### 8.1 Best model summary
Generated automatically:
1. `outputs/best_model_summary.csv`

| Best Feature Method | Best Classifier | Accuracy | Precision_Sports| Recall_Sports |F1_Sports |
n-gram_count_1_2|knn|0.9671848013816926|0.9651567944250871|0.9685314685314685|0.9668411867364747
| Best Feature Method | Best Classifier | Accuracy | Precision_Sports | Recall_Sports | F1_Sports |
|---|---|---:|---:|---:|---:|
| n-gram_count_1_2 | knn | 0.9671848013816926 | 0.9651567944250871 | 0.9685314685314685 | 0.9668411867364747 |

### 8.2 Confusion matrices
Generated per combination:
1. `outputs/cm_<feature>__<classifier>.csv`

### 8.3 Error analysis files
1. `outputs/all_misclassifications.csv`
### 8.4 Best model errors
Generated for the best-performing combination:
1. `outputs/best_model_errors.csv`

| Feature Method | Classifier | Actual | Predicted | Headline |
|---|---|---|---|---|
| ngram_count_1_2 | knn | politics | sports | Sri Lanka: SJB's Sajith Premadasa nominated for interim presidency |
| ngram_count_1_2 | knn | sports | politics | 'If they call, I will be ready': Sprint legend Usain Bolt might return at the 2028 LA Olympics for cricket |
| ngram_count_1_2 | knn | politics | sports | Explained: Why majority governments are holding floor tests |
| ngram_count_1_2 | knn | sports | politics | Used soft drink bottles being refilled as waste-management procedure, not for re-sale, says DDCA in response to viral video |
| ngram_count_1_2 | knn | sports | politics | Javelin thrower Sachin Yadav included in RTP of Athletics Integrity Unit |
| ngram_count_1_2 | knn | sports | politics | Tearful LeBron James keeps retirement talk simmering in possible Cleveland farewell |
| ngram_count_1_2 | knn | sports | politics | Daniel Naroditsky's autopsy reveals cause of death: popular streamer died of abnormal heartbeat |
| ngram_count_1_2 | knn | sports | politics | Daniel Naroditsky had multiple drugs in system at time of death, toxicology report says |
| ngram_count_1_2 | knn | sports | politics | How the Chinese have approached the puzzle of the unbeatable An Se-young |
| ngram_count_1_2 | knn | politics | sports | Bangladesh sports advisor: 'Team will not play in India in any circumstances' |
| ngram_count_1_2 | knn | politics | sports | SP-BSP splitsville on the anvil, who'll cut loose first |
| ngram_count_1_2 | knn | sports | politics | Why does Delhi monopolise major sporting events even as pollution and player welfare are always genuine concerns |
| ngram_count_1_2 | knn | politics | sports | Why 'Mullah Mulayam' tag didn't hurt |
| ngram_count_1_2 | knn | politics | sports | Modi mantra now resonates world over: Yogi Adityanath |


## 9. Interpretation and Analysis
### 9.1 Expected feature behavior
1. Bag of Words is usually fast and strong on topical headlines.
2. TF-IDF helps reduce overemphasis on common words.
3. Unigram+bigram can improve disambiguation for short headline phrases.

### 9.2 Model behavior notes
1. Decision Tree gives a simple rule-based baseline and is easy to explain.
2. Random Forest improves robustness by averaging many decision trees.
3. KNN is a distance-based non-parametric baseline that is straightforward to understand.

### 9.3 Top terms
The notebook exports weighted terms per combination:
1. `outputs/top_terms_all_models.csv`


## 10. Limitations of the System
1. Headline-only input lacks full article context.
2. Data comes from two sources only; publisher-style bias may exist.
3. Temporal drift can reduce performance as news language changes.
4. Mixed-topic headlines can confuse binary classification.
5. Manual stopword list is limited and language-specific.
6. No probability calibration for confidence-aware decisions.
7. Single holdout split gives limited uncertainty estimation.

## 11. Conclusion
The project delivers a complete, assignment-compliant binary classifier for sports vs politics headlines with:
1. Custom feature extraction from scratch (BoW, TF-IDF, n-grams),
2. Three machine learning classifiers,
3. Full 9-combination quantitative comparison,

