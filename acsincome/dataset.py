import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from folktables import ACSDataSource, ACSIncome
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir="./acsincome/data")


class ACSIncomeDataset(Dataset):
    def __init__(self, X: List[float], Y: List[bool]):
        self.X = torch.from_numpy(X).to(dtype=torch.float)
        self.Y = torch.from_numpy(Y).to(dtype=torch.long)
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


def remove_feature(removed_feature_indices: List[int], X: np.ndarray) -> np.ndarray:
    removed = set(removed_feature_indices)
    keep_indices = [i for i in range(X.shape[1]) if i not in removed]
    return X[:, keep_indices]


def acsincome_load_train_val(REMOVED_FEATURE_INDICES: List[int], state: str, VAL_RATE: int) -> Tuple[ACSIncomeDataset, ACSIncomeDataset]:
    # load
    acs_data = data_source.get_data(states=[state], download=True)
    X_train_val, Y_train_val, _ = ACSIncome.df_to_numpy(acs_data)
    X_train_val = remove_feature(REMOVED_FEATURE_INDICES, X_train_val)

    # shuffle
    indices = np.arange(X_train_val.shape[0])
    np.random.shuffle(indices)
    X_train_val, Y_train_val = X_train_val[indices], Y_train_val[indices]

    # split
    VAL_IDX = int(VAL_RATE * len(indices))
    X_train, X_val = X_train_val[VAL_IDX:], X_train_val[:VAL_IDX]
    Y_train, Y_val = Y_train_val[VAL_IDX:], Y_train_val[:VAL_IDX]

    train = ACSIncomeDataset(X_train, Y_train)
    val = ACSIncomeDataset(X_val, Y_val)
    return train, val


def acsincome_load_tests(REMOVED_FEATURE_INDICES: List[int], states: List[str]) -> List[ACSIncomeDataset]:
    tests = []
    for state in states:
        acs_data = data_source.get_data(states=[state], download=True)
        X_test, Y_test, _ = ACSIncome.df_to_numpy(acs_data)
        X_test = remove_feature(REMOVED_FEATURE_INDICES, X_test)
        tests.append(ACSIncomeDataset(X_test, Y_test))

    return tests
