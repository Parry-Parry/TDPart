from typing import List
import ir_datasets as irds 
import pandas as pd

def create_synthetic(datasets : List[str], ratio : List[str], mode : str):
    datasets = {}
    for dataset in datasets:
