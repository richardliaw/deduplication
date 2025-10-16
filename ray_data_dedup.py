"""
Large-scale deduplication using Ray Data with Spark-like operations.

Based on the approach from: https://huggingface.co/blog/dedup

This implementation uses Ray Data's native operations (map_batches, groupby, etc.)
to implement MinHash + LSH deduplication, similar to the Spark approach.

Architecture:
1. MinHash signature generation (map_batches)
2. LSH banding to generate candidate pairs (flatmap + groupby)
3. Connected components to find duplicate clusters (iterative map-reduce)
"""

import argparse
import hashlib
import logging
import struct
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
from pyarrow import fs as pafs
import pandas as pd
import ray
from scipy import integrate
import os

logger = logging.getLogger(__name__)

# Constants
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint32((1 << 32) - 1)


def list_gcs_parquet_files(path: str) -> List[str]:
    """
    List all parquet files in a GCS directory recursively using PyArrow.

    Args:
        path: GCS path (e.g., gs://bucket/path/)

    Returns:
        List of full GCS paths to parquet files
    """
    # Create GCS filesystem
    gcs_fs = pafs.GcsFileSystem()

    # Remove gs:// prefix if present
    if path.startswith("gs://"):
        path = path[5:]

    # Remove trailing slash
    path = path.rstrip("/")

    logger.info(f"Listing parquet files in gs://{path}")

    # Use FileSelector with recursive=True
    selector = pafs.FileSelector(path, recursive=True)
    file_infos = gcs_fs.get_file_info(selector)

    # Filter for parquet files
    parquet_files = []
    for file_info in file_infos:
        if file_info.type == pafs.FileType.File and file_info.path.endswith('.parquet'):
            parquet_files.append(f"gs://{file_info.path}")

    logger.info(f"Found {len(parquet_files)} parquet files")

    return parquet_files


def sha1_hash32(data: bytes) -> int:
    """Generate 32-bit hash from SHA1."""
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
) -> Tuple[int, int]:
    """
    Compute optimal LSH parameters (bands, rows) that minimize weighted sum
    of false positive and false negative probabilities.

    Returns:
        (num_bands, rows_per_band)
    """
    def false_positive_probability(th: float, band: int, rows: int) -> float:
        def proba(s: float) -> float:
            return 1 - (1 - s ** float(rows)) ** float(band)
        a, _ = integrate.quad(proba, 0.0, th)
        return a

    def false_negative_probability(th: float, band: int, rows: int) -> float:
        def proba(s: float) -> float:
            return 1 - (1 - (1 - s ** float(rows)) ** float(band))
        a, _ = integrate.quad(proba, th, 1.0)
        return a

    min_error = float('inf')
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashGenerator:
    """Generates MinHash signatures from text using n-grams."""

    def __init__(
        self,
        num_perm: int = 128,
        ngram_size: int = 5,
        seed: int = 42,
        lowercase: bool = True,
    ):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.lowercase = lowercase

        # Generate permutations for MinHash
        gen = np.random.RandomState(seed=seed)
        self.perm_a, self.perm_b = np.array(
            [
                (
                    gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

    def _ngrams(self, text: str) -> Set[bytes]:
        """Generate character n-grams from text."""
        if self.lowercase:
            text = text.lower()
        return {
            text[i:i + self.ngram_size].encode('utf-8')
            for i in range(len(text) - self.ngram_size + 1)
        }

    def compute_minhash(self, text: str) -> np.ndarray:
        """
        Compute MinHash signature for a single text.

        Returns:
            Array of shape (num_perm,) with uint32 values
        """
        tokens = self._ngrams(text)

        if len(tokens) == 0:
            # Empty text gets max hash values
            return np.full(self.num_perm, MAX_HASH, dtype=np.uint32)

        # Hash all tokens
        hashes = np.array([sha1_hash32(token) for token in tokens], dtype=np.uint64)

        # Apply permutations: (h * a + b) % c
        # Broadcasting: hashes[:, None] has shape (num_tokens, 1)
        # perm_a[None, :] has shape (1, num_perm)
        phv = ((hashes[:, None] * self.perm_a[None, :] + self.perm_b) % MERSENNE_PRIME).astype(np.uint32)

        # Take minimum across all tokens for each permutation
        return phv.min(axis=0)


def generate_minhash_signatures(
    batch: Dict[str, np.ndarray],
    text_column: str,
    num_perm: int,
    ngram_size: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Ray Data UDF to generate MinHash signatures for a batch of documents.

    This function is called by map_batches and processes documents in parallel.
    """
    generator = MinHashGenerator(num_perm=num_perm, ngram_size=ngram_size, seed=seed)

    texts = batch[text_column]
    signatures = np.array([generator.compute_minhash(text) for text in texts])

    # Add signatures to batch
    batch['minhash'] = signatures
    return batch


def generate_lsh_bands(
    batch: Dict[str, np.ndarray],
    num_bands: int,
    rows_per_band: int,
) -> Dict[str, np.ndarray]:
    """
    Generate LSH bands from MinHash signatures.

    This creates multiple (band_id, band_hash) pairs per document,
    which will be used to find candidate duplicate pairs.

    Returns a flattened batch where each row represents one band of one document.

    Example:
    {
        'minhash': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        'id': np.array([1, 2, 3]),
    }
    ->
    {
        'doc_id': np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
        'band_id': np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        'band_hash': np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']),
    }
    """
    minhashes = batch['minhash']
    num_docs = len(minhashes)

    # For each document, generate num_bands rows
    output_size = num_docs * num_bands

    # Replicate document IDs for each band
    doc_ids = np.repeat(batch['id'], num_bands)

    # np.tile repeats the entire array a given number of times.
    # [0, 1, ..., num_bands-1] * num_docs times
    band_ids = np.tile(np.arange(num_bands), num_docs)
    band_hashes = []

    for doc_idx, minhash in enumerate(minhashes):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_values = minhash[start:end]
            # Create a hash of the band
            band_hash = hashlib.sha256(band_values.tobytes()).hexdigest()[:16]
            band_hashes.append(band_hash)

    return {
        'doc_id': doc_ids,
        'band_id': band_ids,
        'band_hash': np.array(band_hashes),
    }


def create_edges_from_collisions(batch: pd.DataFrame) -> pd.DataFrame:
    """Create edges from a batch of candidate pairs that collide."""
    # schema: band_id, band_hash, doc_id
    # all band_id and band_hash are the same
    # Generate all pairs of doc_id

    # batch['doc_id'] can be a pandas Series or 1D array-like
    a = np.asarray(batch['doc_id'])
    n = a.shape[0]
    if n < 2:
        return pd.DataFrame(columns=['src', 'dst'])

    # indices for all i < j
    i, j = np.triu_indices(n, k=1)
    return pd.DataFrame({'src': a[i], 'dst': a[j]})


def distinct(current_ds: ray.data.Dataset, columns: List[str]) -> ray.data.Dataset:
    current_ds = current_ds.groupby(columns).count()
    current_ds = current_ds.drop_columns(['count()'])
    current_ds = current_ds.materialize()
    return current_ds

def large_star_emit(row):
    u, v = row["node"], row["parent"]
    if u == v:
        return [{"node": u, "parent": v}]
    return [{"node": u, "parent": v}, {"node": v, "parent": u}]

def small_star_emit(row):
    """Emit (u, v) if u >= v else (v, u) -> (node, parent)"""
    u, v = row["node"], row["parent"]
    if u >= v:
        return [{"node": u, "parent": v}]
    else:
        return [{"node": v, "parent": u}]


def large_star_map_groups(batch: pd.DataFrame) -> pd.DataFrame:
    """Map groups for large-star.

    take the minimum parent (mp)
    Return dataframe (v, mp) where v > mp for v in neighborhood of node."""
    node = batch['node'].tolist()[0]
    assert len(set(batch["node"])) == 1
    neighbors = batch['parent'].tolist()
    neighbors += [node]
    mp = min(neighbors)
    large_neighbors = [n for n in neighbors if n > node]
    return pd.DataFrame({'node': large_neighbors, 'parent': mp})

def small_star_map_groups(batch: pd.DataFrame) -> pd.DataFrame:
    """Map groups for small-star.

    take the minimum parent (mp)
    let N be all neighbors less than or equal to node
    Return dataframe (v, mp) for all v in N"""
    node = batch['node'].tolist()[0]
    assert len(set(batch["node"])) == 1
    neighbors = batch['parent'].tolist()
    small_neighbors = [n for n in neighbors if n <= node]
    small_neighbors += [node]
    mp = min(small_neighbors)
    return pd.DataFrame({'node': small_neighbors, 'parent': mp})


def compute_connected_components_distributed(
    current_ds: ray.data.Dataset,
    max_iterations: int = 100,
    parallelism: int = 200,
    verbose=False
) -> ray.data.Dataset:
    """
    Compute connected components using distributed large-star/small-star algorithm.

    This iterative algorithm is suitable for large-scale graphs (billions of edges).
    Based on the paper: "Connected Components in MapReduce and Beyond"

    Algorithm:
    1. Large-star: For each edge (u,v), point the larger node to the smaller
        Emit (u, v) and (v, u) -> (node, parent)
        Group by node and take the minimum parent (mp)
        Emit (v, mp) where v > mp for v in neighborhood of node
    2. Small-star: Propagate parent pointers transitively
        Emit (u, v) if u >= v else (v, u) -> (node, parent)
        Group by node and take the minimum parent (mp)
        Emit (v, mp) for all v in neighborhood of node
    3. Repeat until convergence

    Returns:
        Dataset with columns: node, parent (where parent is the component root)
    """
    logger.info("Computing connected components with distributed algorithm...")


    num_components = current_ds.count()
    print(f"Initial count: {num_components}")
    convergence_counter = 3
    for i in range(max_iterations):
        # Step 1: Large-star
        current_ds = current_ds.flat_map(large_star_emit)
        current_ds = current_ds\
            .groupby(['node'], num_partitions=parallelism)\
            .map_groups(large_star_map_groups, batch_format="pandas")
        current_ds = current_ds.materialize()
        current_ds = distinct(current_ds, ['node', 'parent'])
        current_ds = current_ds.materialize()
        if verbose:
            print(current_ds.to_pandas())

        # Step 2: Small-star
        current_ds = current_ds.flat_map(small_star_emit)
        current_ds = current_ds\
            .groupby(['node'], num_partitions=parallelism)\
            .map_groups(small_star_map_groups, batch_format="pandas")

        current_ds = current_ds.materialize()
        current_ds = distinct(current_ds, ['node', 'parent'])
        current_ds = current_ds.materialize()

        new_num_components = len(current_ds.unique("parent"))

        if num_components == new_num_components:
            convergence_counter -= 1
        else:
            convergence_counter = 3
        if convergence_counter <= 0:
            break

        num_components = new_num_components

        print("-" * 10)
        print(f"Iteration {i}: {current_ds.count()}")
        print(f"number of components: {num_components}")
        print(f"convergence_counter: {convergence_counter}")
        print("-" * 10)

    return current_ds

def deduplicate_dataset(
    ds: ray.data.Dataset,
    text_column: str = 'text',
    threshold: float = 0.7,
    num_perm: int = 128,
    ngram_size: int = 5,
    seed: int = 42,
    max_cc_iterations: int = 100,
    validate_local: bool = False,
    hash_parallelism: int = 100,
) -> ray.data.Dataset:
    """
    Perform large-scale fuzzy deduplication using MinHash + LSH.

    Designed for 3TB+ scale using distributed algorithms throughout.

    Args:
        ds: Input Ray dataset with text documents
        text_column: Name of the column containing text
        threshold: Jaccard similarity threshold (0.7 = 70% similar)
        num_perm: Number of MinHash permutations
        ngram_size: Size of character n-grams
        seed: Random seed
        max_cc_iterations: Maximum iterations for connected components
        limit: If set, only process first N documents (for testing)

    Returns:
        Deduplicated dataset
    """
    logger.info(f"Starting large-scale deduplication with threshold={threshold}")
    # Need to materialize first if limiting, or else the limit could be non-deterministic
    ds = ds.materialize()
    masterset_uuids = set(x["id"] for x in ds.select_columns("id").take_all())
    assert "Materialized" in str(type(ds))

    # Compute optimal LSH parameters
    num_bands, rows_per_band = optimal_param(threshold, num_perm)
    logger.info(f"LSH parameters: {num_bands} bands, {rows_per_band} rows per band")

    # Step 1: Generate MinHash signatures
    logger.info("Step 1: Generating MinHash signatures...")
    # Schema: dict_keys(['*', 'minhash'])
    ds_with_minhash = ds.map_batches(
        generate_minhash_signatures,
        fn_kwargs={
            'text_column': text_column,
            'num_perm': num_perm,
            'ngram_size': ngram_size,
            'seed': seed,
        },
        batch_format='numpy',
    )

    # Step 2: Generate LSH bands (creates multiple rows per document)
    logger.info("Step 2: Generating LSH bands...")
    # Schema: ['doc_id', 'band_id', 'band_hash'], non are unique
    bands_ds = ds_with_minhash.map_batches(
        generate_lsh_bands,
        fn_kwargs={
            'num_bands': num_bands,
            'rows_per_band': rows_per_band,
        },
        batch_format='numpy',
    )
    bands_ds = bands_ds.materialize()
    # Step 3: Group by band to find candidate pairs
    logger.info("Step 3: Grouping by bands to find candidate pairs...")
    edges_ds = bands_ds.groupby(
        ['band_id', 'band_hash'], num_partitions=hash_parallelism).map_groups(create_edges_from_collisions, batch_format="pandas")
    edges_ds = edges_ds.materialize()
    print("Length of edges_ds", edges_ds.count())

    # Deduplicate edges (same pair might appear in multiple bands)
    logger.info("Step 4: Deduplicating edges...")
    # Use groupby to deduplicate across all batches
    # Group by (src, dst) and keep just one of each unique edge
    edges_ds = distinct(edges_ds, ['src', 'dst'])
    print("Length of edges_ds after distinct", edges_ds.count())

    # Step 6: Compute connected components (distributed algorithm)
    logger.info("Step 6: Computing connected components (distributed)...")
    edges_ds = edges_ds.rename_columns(
        {"src": "node", "dst": "parent"}
    ).materialize()
    components_ds = compute_connected_components_distributed(
        edges_ds,
        max_iterations=max_cc_iterations,
        parallelism=hash_parallelism,
    )

    # check local version
    if validate_local:
        compute_connected_components_pandas(edges_ds.to_pandas())

    # Step 7: Filter duplicates
    logger.info("Step 7: Filtering duplicates...")

    # Keep only documents where node != parent (extraneous components)
    duplicate_components = components_ds.filter(
        lambda row: row['node'] != row['parent']
    ).materialize()

    logger.info(f"Duplicated components count: {duplicate_components.count()}")
    # Join with original dataset to get full document content
    deduplicated_ds = ds.join(
        duplicate_components,
        on=('id',),
        right_on=('node',),
        join_type='left_anti',
        num_partitions=hash_parallelism)

    return deduplicated_ds


def main():
    """CLI for large-scale deduplication (3TB+)."""
    parser = argparse.ArgumentParser(
        description='Large-scale deduplication with Ray Data (designed for 3TB+)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input path (parquet files, can use wildcards or GCS paths)',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=os.path.join(os.environ["ANYSCALE_ARTIFACT_STORAGE"], "rliaw-scratch"),
        help='Output path for deduplicated data',
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of text column',
    )
    parser.add_argument(
        '--id-column',
        type=str,
        default='id',
        help='Name of ID column (must be unique for each document)',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Jaccard similarity threshold (0.0-1.0)',
    )
    parser.add_argument(
        '--num-perm',
        type=int,
        default=128,
        help='Number of MinHash permutations',
    )
    parser.add_argument(
        '--ngram-size',
        type=int,
        default=5,
        help='Character n-gram size',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--max-cc-iterations',
        type=int,
        default=100,
        help='Maximum iterations for connected components convergence',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process (for testing on subset)',
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--disable-progress-bars",
        action="store_true",
        default=False,
        help="Disable progress bars",
    )

    args = parser.parse_args()
    if args.disable_progress_bars:
        ray.data.DataContext.get_current().enable_progress_bars = False

    # Read input data
    logger.info(f"Reading data from {args.input}")
    input_path = args.input

    # List all parquet files in the directory
    list_of_all_input_files = list_gcs_parquet_files(input_path)
    logger.info(f"Reading {len(list_of_all_input_files)} parquet files")

    ds = ray.data.read_parquet(list_of_all_input_files)

    input_count = ds.count()
    print(f"Original input count {input_count}")
    if args.limit is not None:
        logger.info(f"Limiting input to {args.limit} documents")
        assert input_count >= args.limit
        ds = ds.limit(args.limit)
        input_count = args.limit
    logger.info(f"Input dataset: {input_count} documents")

    # Deduplicate
    deduplicated_ds = deduplicate_dataset(
        ds,
        text_column=args.text_column,
        threshold=args.threshold,
        num_perm=args.num_perm,
        ngram_size=args.ngram_size,
        seed=args.seed,
        max_cc_iterations=args.max_cc_iterations,
        hash_parallelism=args.parallelism
    )

    # Write output
    logger.info(f"Writing deduplicated data to {args.output}")
    deduplicated_ds.write_parquet(args.output)

    output_count = deduplicated_ds.count()
    logger.info(f"Output dataset: {output_count} documents")
    logger.info(f"Removed {input_count - output_count} duplicates ({100*(input_count - output_count)/input_count:.1f}%)")



def compute_connected_components_pandas(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute connected components for a local dataframe using scipy's DisjointSet.

    This is an efficient local algorithm suitable for graphs that fit in memory.

    Args:
        edges_df: DataFrame with columns ['node', 'parent'] representing edges

    Returns:
        DataFrame with columns ['node', 'parent'] where parent is the component root
    """
    from scipy.cluster.hierarchy import DisjointSet
    edges = edges_df[['node', 'parent']].values

    # Create DisjointSet with all unique nodes
    all_nodes = set()
    for node, parent_node in edges:
        all_nodes.add(node)
        all_nodes.add(parent_node)

    ds = DisjointSet(all_nodes)

    # Merge edges
    for node, parent_node in edges:
        ds.merge(node, parent_node)
    # Create result dataframe
    result = pd.DataFrame({
        'node': list(all_nodes),
        'parent': [ds[node] for node in all_nodes]
    })
    print("Number of subsets:", ds.n_subsets)
    return result


def test_connected(local: bool = False):
    # Generate random graph
    n = 10000
    # Generate n random edges between 0 and n-1 (uniformly)
    edges = np.random.randint(0, n, size=(n * 2, 2))
    edges = pd.DataFrame(edges, columns=["node", "parent"])

    print("running distributed")
    ray.data.DataContext.get_current().enable_progress_bars = False
    edges_ds = ray.data.from_pandas(edges)
    edges_ds = edges_ds.map_batches(lambda batch: batch, batch_format="pyarrow")
    result = compute_connected_components_distributed(
        edges_ds, max_iterations=10, parallelism=10)

    print("Running local")
    result = compute_connected_components_pandas(edges)

    print(result)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    main()
    # test_connected()
    print("finished")
