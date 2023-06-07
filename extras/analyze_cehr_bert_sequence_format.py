# Some code to analyze the sequence format used by cehr-bert

import pandas as pd
import pyarrow.parquet as pq
import os

# folder = "D:/omopSynthea/cehr-bert/patient_sequence"
# file = pq.read_table(os.path.join(folder, "part-00000-c0fda67a-757c-41ba-8c31-a69d1f7bf530-c000.snappy.parquet"))
folder = "D:/GPM_CCAE/patient_sequence"
file = "part0001.parquet"
folder = "D:/GPM_CCAE/patient_sequence_2"
file = "part0001.parquet"
pfile = pq.read_table(os.path.join(folder, file))

x = pfile.to_pandas()
print(x.dtypes)
for column in x.columns:
  print(f"column: {column}")
  print(x[column].iat[0])
  print(x[column].iat[0].dtype)

len(x)


full = pd.read_parquet(folder)
cids = full["concept_ids"]
cids = full["visit_concept_ids"]
cids = full["dates"]
x = [j for i in cids for j in i]
x = set(x)
x = list(x)
x.sort()
x[:25]
x[-25:-1]

for i in range(len(full)):
  cidLen = full["num_of_concepts"].iat[i]
  if len(full["concept_ids"].iat[i]) != cidLen:
    print(f"Issue with concept_ids for {i}")
  if len(full["visit_segments"].iat[i]) != cidLen:
    print(f"Issue with visit_segments for {i}")
  if len(full["orders"].iat[i]) != cidLen:
    print(f"Issue with orders for {i}")
  if len(full["dates"].iat[i]) != cidLen:
    print(f"Issue with dates for {i}")
  if len(full["ages"].iat[i]) != cidLen:
    print(f"Issue with ages for {i}")
  if len(full["visit_concept_orders"].iat[i]) != cidLen:
    print(f"Issue visit_concept_orders ages for {i}")

  max([max(x) for x in full["visit_concept_orders"]])
  min([len(x) for x in full["visit_concept_orders"]])


total_visits = 0
shady_visits = 0
for i in range(len(full)):
  dates = full["dates"].iat[i]
  days = (max(dates) - min(dates)) * 7
  vcos = full["visit_concept_orders"].iat[i]
  visits = max(vcos)
  total_visits += visits
  if visits > days / 2:
    shady_visits += visits

print(f"Total visits: {total_visits}")
print(f"Shady visits: {shady_visits}")
print(f"Shady visits: {shady_visits / total_visits}")



