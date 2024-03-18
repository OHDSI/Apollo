import csv
import json
import os
from typing import List, Optional


class Results:

    def __init__(self):
        self._fields: List[str] = []
        self._values: List[any] = []

    def put_value(self, field: str, value: any):
        self._fields.append(field)
        self._values.append(value)

    def write_to_csv(self, file_name: str):
        with open(file_name, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._fields)
            if isinstance(self._values[0], list):
                for i in range(len(self._values[0])):
                    writer.writerow([row[i] for row in self._values])
            else:
                writer.writerow(self._values)


class JsonWriter:
    def __init__(self, file_name: Optional[str] = None):
        self._file_name = file_name
        self.previous = 0
        self.metrics_list = []

    def add_metrics(self, epochs: int, metrics: dict):
        if self.metrics_list:
            for metric in self.metrics_list:
                if metric["epochs"] == epochs:
                    metric["metrics"].update(metrics)
                    return
            self.metrics_list.append({"epochs": epochs, "metrics": metrics})
        else:
            self.metrics_list.append({"epochs": epochs, "metrics": metrics})

    def flush(self):
        if self._file_name:
            if os.path.isfile(self._file_name):
                self.append_metrics()
            self.write_metrics()
        else:
            raise ValueError("No file name specified for json writer")

    def append_metrics(self):
        previous_metrics = self.load_metrics()
        self.metrics_list = previous_metrics + self.metrics_list

    def load_metrics(self):
        with open(self._file_name, "r", encoding="UTF8", newline="") as f:
            return json.load(f)

    def write_metrics(self):
        with open(self._file_name, "w", encoding="UTF8", newline="") as f:
            f.write(json.dumps(self.metrics_list, indent=4))