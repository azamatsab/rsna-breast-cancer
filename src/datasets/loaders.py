import os
import logging

import torch
import pandas as pd
import numpy as np
from catalyst.data import BalanceClassSampler, BatchBalanceClassSampler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

import src.datasets.datasets as datasets


def get_loaders(config, train_transform, test_transform):
    dataset = getattr(datasets, config.dataset)
    batch_size = config.batch_size
    num_workers = config.num_workers

    df_all = pd.read_csv(config.csv_path)
    df_all = df_all.fillna(config.default_age)
    
    if False:
        pid_to_tar = {}
        targets = df_all.cancer.tolist()
        pids = df_all.patient_id.tolist()
        for tar, pid in zip(targets, pids):
            pid_to_tar[pid] = tar

        pids = set(pids)
        pids = sorted(list(pids))
        targets = [pid_to_tar[pid] for pid in pids]
        length = len(pids) // 10
        train, test = pids[:-length], pids[-length:]
        skf = StratifiedKFold(n_splits=5, random_state=None)
        logging.info(f"Training on fold {config.fold}")
        for i, (train_index, test_index) in enumerate(skf.split(pids, targets)):
            if i == config.fold:
                break
        train = np.array(pids)[train_index]
        test = np.array(pids)[test_index]

        df_train = df_all[df_all.patient_id.isin(set(train))]
        df_test = df_all[df_all.patient_id.isin(set(test))]
    else:
        num_bins = 5
        df_all["age_bin"] = pd.cut(df_all["age"].values.reshape(-1), bins=num_bins, labels=False)
        strat_cols = [
            "laterality", "view", "biopsy","invasive", "BIRADS", "age_bin",
            "implant", "density","machine_id", "difficult_negative_case",
            "cancer",
        ]

        df_all["stratify"] = ""
        for col in strat_cols:
            df_all["stratify"] += df_all[col].astype(str)

        skf = StratifiedGroupKFold(n_splits=5)
        for fold_, (train_idx, test_idx) in enumerate(skf.split(df_all, df_all["stratify"].values, df_all["patient_id"].values)):
            df_all.loc[test_idx, "fold"] = fold_
        df_all.fold = df_all.fold.astype(int)

        logging.info(f"Training on fold {config.fold}")
        df_train = df_all[df_all.fold != config.fold]
        df_test = df_all[df_all.fold == config.fold]

    train_dataset = dataset(df_train, config, train_transform, train=True)
    if config.balanced_batch:
        train_sampler = BatchBalanceClassSampler(
                train_dataset.targets, num_classes=2, num_samples=batch_size // 2)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=num_workers,
            batch_sampler=train_sampler
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=num_workers, batch_size=batch_size,
            shuffle=True
        )

    test_dataset = dataset(df_test, config, test_transform, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_loader, test_loader