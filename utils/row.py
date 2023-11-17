import csv
from typing import List


class Row:

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
            writer.writerow(self._values)