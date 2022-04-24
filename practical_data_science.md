---
marp: true
title: Practical Data Science
theme: uncover
---

<!-- _class: invert -->
<!-- ![bg](images/title-background.png) -->

## Practical Data Science

Bauke Brenninkmeijer

<!-- footer: Kaggle Competition for Jr. Data Science 2022 -->

---

<!-- paginate: true -->

## Goal

Tell you what `works`
and `doesn't work`
specifically when starting out
with data science

<!-- footer: Practical Data Science • Bauke Brenninkmeijer -->

---

## whois:`Bauke Brenninkmeijer`

- MSc in CS and Data Science @Nijmegen
- Data Scientist @ABNAMRO since 2019
  - 1.5 years in Data Management
  - ~1 years in Global Markets
- Co-founder of DSFC

- [![github_logo](images/GitHub-Mark-32px.png)](https://github.com/Baukebrenninkmeijer) [@baukebrenninkmeijer](https://github.com/Baukebrenninkmeijer)

---

## Feature engineering

It always looks simple

But mostly isn't

---

### Feature Engineering [1/x]

Ordinal features

---

## Class Imbalance
Techniques used include:
- Oversampling
- Undersampling
- Smote

All methods fail to reach high levels of recall while creating undue complexity.

But there is a more intuitive way.

---

<!-- _paginate: False -->

# SMOTE
- `Synthetic Minority Oversampling Technique`
- Creates synthetic points to increase the number of observations in minority class

![bg right vertical](images/smote_theory.png)

![bg right](images/smote_actually.png)

---

# Upsampling

- Fix class imbalance by looking more at the minority class
- I.e., duplicate minority data points

---

# Undersampling

- Only look at the same number of data points in the majority class, as there are in the minority class.
- I.e., drop part of the data

---

# Class Weights

- Default part of sklearn
- Manages tradeoff between precision and recall
- Allows one to penalise minority errors more
- Assign weights to classes such that the weighted sums is equal.

$$
n_{minority} \times w_{minority} = n_{majority} \times w_{majority}
$$

---

# Order of pre-processing

- Using transformation steps before doing the train/test split
- You should split as a first step
- Target and train distribution leakage through the transformation
- e.g. with StandardScaler, the scaler knows the mean and std