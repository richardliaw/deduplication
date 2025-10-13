"""
Quick script to inspect the schema of the dataset.
"""

import ray

# Initialize Ray
ray.init()

# Read the dataset
print("Reading dataset...")
ds = ray.data.read_parquet("gs://anyscale-example-datasets/HuggingFaceFW/fineweb-edu/data/")

# Print schema
print("\nDataset Schema:")
print(ds.schema())

# Print column names
print("\nColumn Names:")
print(ds.schema().names)

# Take a few samples to see the data
print("\nSample Data (first 3 rows):")
samples = ds.take(3)
for i, sample in enumerate(samples):
    print(f"\nRow {i}:")
    for key, value in sample.items():
        # Print first 100 chars of text fields
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

# Print dataset stats
print(f"\nTotal rows: {ds.count()}")
