# Cleans the SuperKart dataset and creates train/test splits on HF hub.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/iamsubha/superkart/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print("Loaded:", df.shape)

# ---- cleaning ----
# fix the 'reg' typo in the sugar content column
df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace({'reg': 'Regular'})

# pull Food / Non-Consumable / Drinks out of the product id as a feature
df['Product_Category'] = df['Product_Id'].str[:2].map(
    {'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'}
)

# derive store age instead of keeping the raw year
CURRENT_YEAR = 2025
df['Store_Age'] = CURRENT_YEAR - df['Store_Establishment_Year']

# drop columns we don't need for modelling
df = df.drop(columns=['Product_Id', 'Store_Id', 'Store_Establishment_Year'])

# ---- target + features ----
target = 'Product_Store_Sales_Total'
X = df.drop(columns=[target])
y = df[target]

# ---- split ----
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# push splits back to the dataset repo
for f in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id="iamsubha/superkart",
        repo_type="dataset",
    )
    print("uploaded", f)
