# TODO: complete this file.
import numpy as np
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)
from matrix_factorization import als
from item_response import irt, sigmoid
from sklearn.impute import KNNImputer

np.random.seed(2025)

def bootstrap_data(data):
    N = len(data['question_id'])
    idxs = np.random.randint(0, N, size=N)
    print(idxs)
    return {
        'user_id':     [data['user_id'][i]     for i in idxs],
        'question_id': [data['question_id'][i] for i in idxs],
        'is_correct':  [data['is_correct'][i]  for i in idxs]
    }


def build_sparse_matrix(data, num_users, num_items):
    mat = load_train_sparse("./data").toarray()
    for u, q, c in zip(data['user_id'], data['question_id'], data['is_correct']):
        mat[u, q] = c
    return mat


def ensemble_accuracy(models, data):
    preds = []
    for u, q, true in zip(data['user_id'], data['question_id'], data['is_correct']):
        scores = [m[u, q] for m in models]
        avg = np.mean(scores)
        preds.append(int(avg >= 0.5))
    return np.mean(np.array(preds) == np.array(data['is_correct']))


def main():
    # Load raw data
    train_data = load_train_csv("./data")
    val_data   = load_valid_csv("./data")
    test_data  = load_public_test_csv("./data")


    num_users = max(train_data['user_id']) + 1
    num_items = max(train_data['question_id']) + 1

    # Hyperparameters
    k_als    = 40; lr_als    = 0.02; iters_als = 50000
    lr_irt   = 0.01; iters_irt = 50
    k_knn    = 11

    base_models = []

    # 1) ALS ensemble members
    for i in range(3):
        boot = bootstrap_data(train_data)
        mat_als = als(boot, k_als, lr_als, iters_als)
        base_models.append(mat_als)
        print(f"Trained ALS model {i+1}")

    # 2) IRT ensemble members
    for i in range(3):
        boot = bootstrap_data(train_data)
        theta, beta, _, _ = irt(boot, val_data, lr_irt, iters_irt)
        mat_irt = np.zeros((num_users, num_items))
        for u in range(num_users):
            for q in range(num_items):
                mat_irt[u, q] = sigmoid(theta[u] - beta[q])
        base_models.append(mat_irt)
        print(f"Trained IRT model {i+1}")

    # 3) KNN ensemble members
    for i in range(3):
        boot = bootstrap_data(train_data)
        mat_boot = build_sparse_matrix(boot, num_users, num_items)
        imputer = KNNImputer(n_neighbors=k_knn)
        mat_knn = imputer.fit_transform(mat_boot)
        base_models.append(mat_knn)
        print(f"Trained KNN model {i+1}")

    # Evaluate individual models
    print("\nIndividual model performance:")
    idx = 1
    for mat in base_models:
        v = sparse_matrix_evaluate(val_data, mat)
        t = sparse_matrix_evaluate(test_data, mat)
        print(f" Model #{idx}: val_acc={v:.4f}, test_acc={t:.4f}")
        idx += 1

    # Ensemble (average all 9 models)
    ens_val = ensemble_accuracy(base_models, val_data)
    ens_test= ensemble_accuracy(base_models, test_data)
    print(f"\nEnsemble of 9 models: val_acc={ens_val:.4f}, test_acc={ens_test:.4f}")




if __name__ == "__main__":
    main()
