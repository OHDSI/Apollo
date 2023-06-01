# For my own sanity, a documentation of issues I've had with pandas:
import pandas as pd

# Storing and retrieving integers --------------------------------------------------------------------------------------
# By default, pandas stores integers as int64, which can't store nulls. This leads to silent conversion to float64:
df = pd.DataFrame({"int": [1, 2]})
df["int"].dtype
# dtype('int64')
df.iloc[0, 0] = None
df["int"].dtype
# dtype('float64')

# To avoid this, we can use the nullable integer type:
df = pd.DataFrame({"int": pd.array([1, 2], dtype=pd.Int64Dtype())})
df["int"].dtype
# Int64Dtype()
df.iloc[0, 0] = None
df["int"].dtype
# Int64Dtype()

# Slicing and indexing -------------------------------------------------------------------------------------------------
# When slicing a DataFrame, the index is preserved, even when creating a copy:
df = pd.DataFrame({"int": [1, 2, 3, 4, 5]})
part_df = df[df.int > 3].copy()
part_df.int[0]
# Key error

# Instead, we need to reset the index:
part_df = df[df.int > 3].copy().reset_index(drop=True)
part_df.int[0]
# 4
