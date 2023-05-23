# Some code to see if we can speed up concept mapping
import random
import pandas as pd
import timeit

random.seed(0)

ids_to_map = [random.randint(0, int(1e7)) for i in range(1000)]
source_ids = [random.randint(0, int(1e7)) for i in range(100000)]
target_ids = [random.randint(0, int(1e7)) for i in range(100000)]


# Using pandas
mapping = pd.DataFrame({"source_id": source_ids, "target_id": target_ids})
mapping.set_index("source_id", drop=True, inplace=True)
df = pd.DataFrame({"source_id": ids_to_map})
stmt = '''
result = df.merge(mapping, how="inner", left_on="source_id", right_index=True)
'''
timeit.timeit(stmt=stmt, globals=globals(), number=10)
# 0.13781319199999942

# Using a dictionary
mapping = dict(zip(source_ids, target_ids))


def do_map(x):
    if x in mapping:
        return mapping[x]
    else:
        return -1


stmt = '''
result = [do_map(x) for x in ids_to_map]
'''
timeit.timeit(stmt=stmt, globals=globals(), number=10)
# 0.0017271530000471103
