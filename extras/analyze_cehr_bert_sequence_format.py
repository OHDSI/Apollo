import pandas as pd
import pyarrow.parquet as pq
import pyarrow
import os

# folder = 'D:/omopSynthea/cehr-bert/patient_sequence'
# pfile = pq.read_table(os.path.join(folder, 'part-00000-c0fda67a-757c-41ba-8c31-a69d1f7bf530-c000.snappy.parquet'))
folder = 'D:/GPM_MDCD/patient_sequence'
pfile = pq.read_table(os.path.join(folder, 'part0001.parquet'))
x = pfile.to_pandas()
print(x.dtypes)
for column in x.columns:
  print(f"column: {column}")
  print(x[column].iat[0])
  print(x[column].iat[0].dtype)

v= x["concept_ids"].iat[0]


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

max_seq_len = 512
idx = []
for i in range(len(full)):
  vcos = full["visit_concept_orders"].iat[i]
  seq_length = len(vcos)
  half_window_size = int(max_seq_len / 2)
  for cursor in range(seq_length):
    start_index = max(0, cursor - half_window_size)
    end_index = min(cursor + half_window_size, seq_length)
    sub_sequence = vcos[start_index:end_index]
    if max(sub_sequence) - min(sub_sequence) > 512:
      print(i)
