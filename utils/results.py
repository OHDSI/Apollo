import csv
from typing import List


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