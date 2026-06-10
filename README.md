# AI SMS Spam Filter From Scratch (Bayes' Theorem)

A Python-based Natural Language Processing (NLP) machine learning engine that classifies text messages as **Spam** or **Ham (Normal)**. Built entirely from scratch using raw mathematics, without using high-level ML libraries like Scikit-Learn or TensorFlow.

---

## The Mathematics Behind the Project

This project implements a **Naive Bayes Classifier** directly from your core foundation of probability and calculus:

1. **Bayes' Theorem ($P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$):** Calculates the probability that a message is spam *given* the specific words it contains.
2. **Conditional & Joint Distributions:** The model maps out the frequencies of words across 5,500+ real-world SMS messages to calculate $P(\text{Word} \mid \text{Spam})$ and $P(\text{Word} \mid \text{Ham})$.
3. **Numerical Stability & Log-Transformations ($\log(A \cdot B) = \log A + \log B$):** Multiplying thousands of tiny decimal probabilities causes *Arithmetic Underflow* (the computer rounds the number to 0.0, breaking the model). To solve this, I applied a NumPy logarithmic transformation to **add** log-probabilities instead of multiplying raw probabilities, ensuring 100% numerical stability.

---

## Tech Stack & Concepts Used

* **Python 3** (Core logic, string manipulation, and tokenization)
* **Pandas** (Data cleaning, column-wise processing via `.apply()`, and frequency mapping)
* **NumPy** (Mathematical log-scaling for stability)
* **Google Colab** (Cloud development environment)

---

## How It Works (Step-by-Step)

1. **Data Ingestion:** Loads the famous SMS Spam Collection dataset (~5,572 messages) directly from the internet.
2. **Text Cleaning & Tokenization:** Converts strings to lowercase, strips punctuation, and breaks sentences down into unique word arrays.
3. **Probability Mapping:** Builds a dictionary counting word frequencies separately for the Spam pile and the Ham pile.
4. **Stable Inference Engine:** Evaluates new, unseen text input by calculating and comparing log-probability scores.

### Sample Prediction Output:
```text
Input: "URGENT! Claim your free cash prize right now!"

Stable Log-Spam Score: -18.134
Stable Log-Ham Score:  -44.412
Result: 🚨 SPAM DETECTED!
