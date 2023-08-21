import os
from typing import List, Dict

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


def list_files_with_extension(folder_path:  str, extension: str) -> List[str]:
    file_list = []
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(extension):
                file_list.append(os.path.join(folder_path, entry.name))
    return file_list


class ApolloDataset(Dataset):
    def __init__(self, folder: str, transform: callable(Dict)=None):
        self.file_paths = list_files_with_extension(folder, ".h5")
        self.files = [h5py.File(file_path, "r") for file_path in self.file_paths]
        self.lengths = []
        for file in self.files:
            self.lengths.append(len(file["tokens"]))
        self.transform = transform

    def __len__(self) -> int:
        return sum(self.lengths)

    def __getitem__(self, idx) -> Dict:
        file_idx = 0
        while idx >= self.lengths[file_idx]:
            idx -= self.lengths[file_idx]
            file_idx += 1
        file = self.files[file_idx]
        row = {
            "person_id": file["person_id"][idx],
            "tokens": np.array(file["tokens"][str(idx)]),
            "visit_segments": np.array(file["visit_segments"][str(idx)]),
            "dates": np.array(file["dates"][str(idx)]),
            "ages": np.array(file["ages"][str(idx)]),
            "visit_concept_orders": np.array(file["visit_concept_orders"][str(idx)]),
            "visit_concept_ids": np.array(file["visit_concept_ids"][str(idx)]),
            "orders": np.array(file["orders"][str(idx)]),
            "num_of_concepts": file["num_of_concepts"][idx],
            "num_of_visits": file["num_of_visits"][idx],
        }
        if self.transform:
            row = self.transform(row)
        return row

    def __del__(self):
        for file in self.files:
            file.close()


if __name__ == "__main__":
    dataset = ApolloDataset("d:/GPM_Sim/pretraining/patient_sequence")
    for i in range(len(dataset)):
        x = dataset[i]
        if i % 100 == 0:
            print(i)
