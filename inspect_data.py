import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\SIH\BigEarthNet.txt")

files = list(ROOT.glob("*.parquet"))
print("Parquet files:", files)

if not files:
    raise FileNotFoundError("No parquet file found.")

df = pd.read_parquet(files[0])

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nTask types:")
print(df["type"].value_counts())

print("\nCategories:")
print(df["category"].value_counts())

print("\nSplits:")
print(df["split"].value_counts())

print("\nExample VQA rows:")
vqa = df[df["type"].isin(["binary", "mcq"])]

print("\nExample VQA rows:")
print(
    vqa[
        ["patch_id", "input", "output", "type", "category"]
    ].head(10).to_string(index=False)
)
print("\nExample grounding rows:")
print(
    df[df["type"] == "bounding box"][
        ["patch_id", "input", "output", "type", "category"]
    ].head(5).to_string(index=False)
)

print("\nExample captioning rows:")
print(
    df[df["type"] == "captioning"][
        ["patch_id", "input", "output", "type", "category"]
    ].head(5).to_string(index=False)
)