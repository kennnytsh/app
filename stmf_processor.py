import pandas as pd

def stmf_to_dataframe(file_path):
    headers = pd.read_csv(file_path, sep='\t', skiprows=28, nrows=1, header=None).iloc[0]
    dataset = pd.read_csv(file_path, sep='\t', skiprows=30, header=None, names=headers)
    return dataset


