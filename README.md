# CSC311 Final Project (Summer 2025)

**Course:** CSC311 - Introduction to Machine Learning  
**Instructor:** Amanjit Singh Kainth  
**Institution:** University of Toronto  
**Project:** Final Group Project (Option 1, Part A & B)
**Group Member:** Haichuan Qi, Xing Xu, Linghao Wang

## Repository URL
    HTTPS: https://github.com/riverqqq/csc311-group-project.git
    SSH: git@github.com:riverqqq/csc311-group-project.git

## Overview

This repository contains the starter Python code for the CSC311 Final Project (Option 1, Part A & B). The project focuses on applying and improving machine learning algorithms to predict student responses to diagnostic questions, using real-world educational data from Eedi.  
The code is structured to help students:

- Experiment with and implement K-Nearest Neighbors, Item Response Theory, Matrix Factorization, Neural Networks, and Ensemble methods.
- Analyze and compare algorithm performance.
- Extend one of the methods for improved prediction (Part B).

## Starter Files

The following Python modules are included:
- `knn.py` — Collaborative filtering using KNN (user- and item-based).
- `item_response.py` — One-parameter Item Response Theory (IRT) model.
- `matrix_factorization.py` — Matrix factorization (SVD/ALS/SGD) methods.
- `neural_network.py` — Neural network (autoencoder) for student-question prediction.
- `ensemble.py` — Bagging ensemble methods.
- `majority_vote.py` — Example baseline (majority vote, not part of the core deliverables).
- `utils.py` — Helper functions for data loading, saving, and evaluation.
- `data` - Primary Data

**Project instructions and requirements:** see `Project.pdf` in this repository.

## Current Project Structure

```
csc311-group-project/
├── ensemble.py
├── item_response.py
├── knn.py
├── majority_vote.py
├── matrix_factorization.py
├── neural_network.py
├── utils.py
├── Project.pdf
├── README.md
├── LICENSE
├── .gitignore
└── data/
    ├── train_data.csv
    ├── valid_data.csv
    ├── test_data.csv
    ├── private_test_data.csv
    └── train_sparse.npz
```

## Data

The project includes the following data files in the `./data/` directory:
- `train_data.csv`
- `valid_data.csv`
- `test_data.csv`
- `private_test_data.csv`
- `train_sparse.npz`
- (and optionally) metadata files

## Usage

Starter code is provided as templates. You must:
- Complete all `TODO` sections in the code.
- Tune hyperparameters and implement required plots/analyses.
- Write a report as described in the project instructions.
- See the provided comments and docstrings for specific instructions in each file.

## Attribution

This project is for educational purposes only.  
**Based on coursework from:**  
University of Toronto, CSC311 - Introduction to Machine Learning, Summer 2025  
**Instructor:** Amanjit Singh Kainth

See [Project.pdf](Project.pdf) for detailed requirements and attribution of the dataset ([Wang et al., 2020](https://arxiv.org/abs/2007.12061)).

---

## License

See [LICENSE](LICENSE.txt) for terms.

---

## Acknowledgements

- [University of Toronto Department of Computer Science](https://www.cs.toronto.edu/)
- [Eedi Education Challenge dataset](https://arxiv.org/abs/2007.12061)

