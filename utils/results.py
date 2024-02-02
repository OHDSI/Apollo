import csv
import json
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
        self.metrics = {}

    def add_metrics(self, epochs: int, metrics: dict):
        if epochs == self.previous:
            # update self.metrics dict
            self.metrics["metrics"].update(metrics)
            with open(self._file_name, "a", encoding="UTF8", newline="") as f:
                f.write(json.dumps(self.metrics) + "\n")
        else:
            self.metrics = {"epochs": epochs, "metrics": metrics}
            self.previous = epochs

