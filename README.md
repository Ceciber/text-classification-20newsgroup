# Text Classification: From Bag-of-Words to PyTorch
**Academic Project | Final Assessment - Télécom Paris**

This repository contains the work completed for the final laboratory assessment in Natural Language Processing at Télécom Paris. The project serves as a deep dive into the various ways textual data can be represented and classified, transitioning from classical statistical methods to modern deep learning.

## 1. Project Overview
The project evaluates how different text representation strategies impact the performance of a classifier using the **20Newsgroups** dataset. The objective was to move beyond simple word counts to capture semantic meaning and evaluate the efficiency of neural architectures.

## 2. Features & Methodology

### Data Preprocessing
* Custom cleaning pipeline: lowercasing, regex-based noise removal (numbers/punctuation), and short-word filtering.
* Label aggregation: Mapping 20 granular categories into 6 broad themes (Comp, Sci, Rec, Pol, Rel, Misc).

### Models & Representations
* **Baseline:** Logistic Regression with `CountVectorizer` and `TfidfVectorizer`.
* **Topic Modeling:** Dimensionality reduction using `TruncatedSVD` (LSA) to 300 latent dimensions.
* **Word Embeddings:**
    * **Word2Vec:** Trained from scratch on the corpus.
    * **GloVe:** Integration of pre-trained global vectors.
* **Neural Networks (PyTorch):**
    * Implementation of a custom `Dataset` and `DataLoader` with padding/truncation.
    * **Deep Averaging Network (DAN):** A neural model that learns task-specific embeddings and aggregates them for classification.

## 3. Results Summary
| Model | Representation | Test Accuracy |
| :--- | :--- | :--- |
| Logistic Regression | TF-IDF | ~80% |
| Logistic Regression | LSA (300 Topics) | ~75% |
| PyTorch (Learned) | Averaged Embeddings | ~76% |
| PyTorch (GloVe) | Pre-trained Vectors | ~63% |

*Key finding: Task-specific learned embeddings outperformed frozen pre-trained GloVe vectors, likely due to the specific technical jargon found in the newsgroup archives.*

## 4. Technologies Used
* **Python 3.12**
* **PyTorch:** Neural network modeling and training.
* **Scikit-Learn:** Baseline models and evaluation metrics.
* **Gensim:** Word2Vec training and GloVe loading.
* **NLTK:** Tokenization and text processing.
