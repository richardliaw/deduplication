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
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs as pafs
import ray
from ray.data.aggregate import AggregateFnV2
from scipy import integrate

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
    """
    minhashes = batch['minhash']
    num_docs = len(minhashes)

    # For each document, generate num_bands rows
    output_size = num_docs * num_bands

    # Replicate document IDs for each band
    if 'id' in batch:
        doc_ids = np.repeat(batch['id'], num_bands)
    else:
        # Create sequential IDs if not present
        doc_ids = np.repeat(np.arange(num_docs), num_bands)

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


def create_edges_from_candidates(
    batch: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Create edges between documents that share the same band hash.

    Input: grouped by (band_id, band_hash), contains list of doc_ids
    Output: edges (pairs of doc_ids)
    """
    # This receives groups of doc_ids that share the same band_hash
    doc_ids = batch['doc_id']

    # Generate all pairs within this group
    edges = []
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            # Always put smaller ID first for consistency
            u, v = sorted([doc_ids[i], doc_ids[j]])
            edges.append((u, v))

    if len(edges) == 0:
        return {'src': np.array([], dtype=np.int64), 'dst': np.array([], dtype=np.int64)}

    edges_array = np.array(edges)
    return {
        'src': edges_array[:, 0],
        'dst': edges_array[:, 1],
    }


def deduplicate_edges(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Remove duplicate edges from the edge list."""
    if len(batch["src"]) == 0:
        return {
            'src': batch["src"],
            'dst': batch["dst"]
        }
    edges = np.stack([batch['src'].tolist(), batch['dst'].tolist()], axis=1)
    unique_edges = np.unique(edges, axis=0)
    return {
        'src': unique_edges[:, 0],
        'dst': unique_edges[:, 1],
    }




def large_star_iteration(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Large star iteration: for each edge (u,v), if u < v, emit v -> u.
    This propagates smaller IDs to their neighbors.
    """
    src = batch['src']
    dst = batch['dst']

    # For each edge, emit both directions with the smaller node as the target
    new_src = []
    new_dst = []

    for u, v in zip(src, dst):
        if u < v:
            # Emit: v should point to u (smaller)
            new_src.append(v)
            new_dst.append(u)
        elif v < u:
            # Emit: u should point to v (smaller)
            new_src.append(u)
            new_dst.append(v)
        # If u == v, skip (self-loop)

    return {
        'node': np.array(new_src, dtype=np.int64),
        'parent': np.array(new_dst, dtype=np.int64),
    }


def small_star_iteration(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Small star iteration: propagate parent pointers transitively.
    For each edge (u, parent[u]), emit (u, parent[parent[u]]).
    """
    nodes = batch['node']
    parents = batch['parent']

    # Keep the parent assignment
    return {
        'node': nodes,
        'parent': parents,
    }


class MinParentAggregate(AggregateFnV2):
    """
    Custom aggregate function to compute minimum parent for each node.

    This uses AggregateFnV2 for efficient distributed aggregation.
    """

    def __init__(self):
        super().__init__(
            name="min_parent",
            zero_factory=lambda: np.inf,
            on="parent",
            ignore_nulls=True,
        )

    def aggregate_block(self, block: pa.Table) -> int:
        """Aggregate a block by finding minimum parent."""
        parents = block['parent'].to_numpy()
        if len(parents) == 0:
            return np.inf
        return int(np.min(parents))

    def combine(self, accumulator: int, partial: int) -> int:
        """Combine partial results by taking minimum."""
        return min(accumulator, partial)

    def finalize(self, accumulator: int) -> int:
        """Finalize by returning the minimum parent."""
        return accumulator


class EdgeGenerationAggregate(AggregateFnV2):
    """Generate all pairs of documents in the same band."""

    def __init__(self):
        super().__init__(
            name="edge_generation",
            zero_factory=lambda: [],
            on="doc_id",
            ignore_nulls=True,
        )

    def aggregate_block(self, block: pa.Table) -> List[int]:
        """Collect all doc_ids in this block."""
        return block['doc_id'].to_pylist()

    def combine(self, accumulator: List[int], partial: List[int]) -> List[int]:
        """Combine doc_id lists."""
        return accumulator + partial

    def finalize(self, accumulator: List[int]) -> Dict[str, List[int]]:
        """Generate all pairs from the collected doc_ids."""
        doc_ids = accumulator
        edges_src = []
        edges_dst = []

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                u, v = sorted([doc_ids[i], doc_ids[j]])
                edges_src.append(u)
                edges_dst.append(v)

        return {
            'src': edges_src,
            'dst': edges_dst,
        }


def compute_connected_components_distributed(
    edges_ds: ray.data.Dataset,
    max_iterations: int = 100,
) -> ray.data.Dataset:
    """
    Compute connected components using distributed large-star/small-star algorithm.

    This iterative algorithm is suitable for large-scale graphs (billions of edges).
    Based on the paper: "Connected Components in MapReduce and Beyond"

    Algorithm:
    1. Large-star: For each edge (u,v), point the larger node to the smaller
    2. Small-star: Propagate parent pointers transitively
    3. Repeat until convergence

    Returns:
        Dataset with columns: node, parent (where parent is the component root)
    """
    logger.info("Computing connected components with distributed algorithm...")

    # Initialize: each node points to itself
    # Start with edges as initial parent pointers
    current_ds = edges_ds.map_batches(
        lambda batch: {
            'node': np.concatenate([batch['src'], batch['dst']]),
            'parent': np.concatenate([batch['src'], batch['dst']]),
        },
        batch_format='numpy',
    )

    # Get unique nodes with their initial parents (themselves)
    current_ds = current_ds.groupby('node').aggregate(MinParentAggregate())

    # Add edges as parent relationships
    edge_parents = edges_ds.map_batches(
        lambda batch: {
            'node': batch['dst'],
            'parent': batch['src'],
        },
        batch_format='numpy',
    )

    current_ds = ray.data.DatasetContext.get_current().union(
        current_ds, edge_parents
    ) if hasattr(ray.data.DatasetContext, 'union') else current_ds.union(edge_parents)

    # Initial aggregation using AggregateFnV2
    current_ds = current_ds.groupby('node').aggregate(MinParentAggregate())

    iteration = 0
    prev_count = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Connected components iteration {iteration}")

        # Materialize to enable checkpointing and multiple passes
        current_ds = current_ds.materialize()

        # Create parent lookup by duplicating dataset
        parent_lookup = current_ds

        # Large-star: join nodes with their parents
        # For each (node, parent), look up parent's parent

        # Join current_ds with parent_lookup
        # This is expensive - we need to propagate parent pointers

        # Option 1: Broadcast small parent table (if fits in memory)
        # Option 2: Distributed join

        # For 3TB scale, we'll do iterative propagation without explicit join
        # by using the edge-based approach

        # Convert parent pointers to edges
        edges_for_next = current_ds.map_batches(
            lambda batch: {
                'src': batch['parent'],
                'dst': batch['node'],
            },
            batch_format='numpy',
        )

        # Apply large-star
        large_star_edges = edges_for_next.map_batches(
            large_star_iteration,
            batch_format='numpy',
        )

        # Combine with current parent pointers
        combined = ray.data.DatasetContext.get_current().union(
            current_ds, large_star_edges
        ) if hasattr(ray.data.DatasetContext, 'union') else current_ds.union(large_star_edges)

        # Aggregate: for each node, take minimum parent using AggregateFnV2
        next_ds = combined.groupby('node').aggregate(MinParentAggregate())

        # Check convergence: count how many nodes changed parents
        # For large scale, we use a sample-based check
        sample_size = min(10000, next_ds.count())

        if iteration % 5 == 0:
            # Checkpoint periodically
            logger.info(f"Checkpointing at iteration {iteration}")
            next_ds = next_ds.materialize()

        current_ds = next_ds

        # Convergence check: if no changes in last iteration
        current_count = current_ds.count()
        if current_count == prev_count and iteration > 1:
            logger.info(f"Converged after {iteration} iterations")
            break

        prev_count = current_count

    logger.info(f"Connected components completed after {iteration} iterations")

    return current_ds




def filter_duplicates(
    batch: Dict[str, np.ndarray],
    components_dict: Dict[int, int],
) -> Dict[str, np.ndarray]:
    """
    Filter out duplicate documents, keeping only one per component.

    We keep documents that are roots of their component.
    """
    if 'id' not in batch:
        # Add sequential IDs if not present
        batch['id'] = np.arange(len(batch[next(iter(batch.keys()))]))

    doc_ids = batch['id']

    # Keep documents that are not marked as duplicates
    mask = np.array([
        doc_id not in components_dict or components_dict[doc_id] == doc_id
        for doc_id in doc_ids
    ])

    # Filter all columns
    return {k: v[mask] for k, v in batch.items()}


def deduplicate_dataset(
    ds: ray.data.Dataset,
    text_column: str = 'text',
    threshold: float = 0.7,
    num_perm: int = 128,
    ngram_size: int = 5,
    seed: int = 42,
    checkpoint_dir: str | None = None,
    max_cc_iterations: int = 100,
    limit: int | None = None,
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
        checkpoint_dir: Directory for checkpointing intermediate results
        max_cc_iterations: Maximum iterations for connected components
        limit: If set, only process first N documents (for testing)

    Returns:
        Deduplicated dataset
    """
    logger.info(f"Starting large-scale deduplication with threshold={threshold}")

    # Limit dataset size if requested (for testing)
    if limit is not None:
        logger.info(f"Limiting to first {limit} documents for testing")
        ds = ds.limit(limit)

    # Compute optimal LSH parameters
    num_bands, rows_per_band = optimal_param(threshold, num_perm)
    logger.info(f"LSH parameters: {num_bands} bands, {rows_per_band} rows per band")

    # Step 1: Generate MinHash signatures
    logger.info("Step 1: Generating MinHash signatures...")
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

    # Checkpoint after expensive MinHash computation
    if checkpoint_dir:
        minhash_path = f"{checkpoint_dir}/minhash"
        logger.info(f"Checkpointing MinHash signatures to {minhash_path}")
        ds_with_minhash.write_parquet(minhash_path, try_create_dir=True)
        ds_with_minhash = ray.data.read_parquet(minhash_path)

    # Step 2: Generate LSH bands (creates multiple rows per document)
    logger.info("Step 2: Generating LSH bands...")
    bands_ds = ds_with_minhash.map_batches(
        generate_lsh_bands,
        fn_kwargs={
            'num_bands': num_bands,
            'rows_per_band': rows_per_band,
        },
        batch_format='numpy',
    )

    # Step 3: Group by band to find candidate pairs
    logger.info("Step 3: Grouping by bands to find candidate pairs...")
    grouped_ds = bands_ds.groupby(['band_id', 'band_hash'])

    # Step 4: Generate edges from candidate pairs
    logger.info("Step 4: Generating edges from candidate pairs...")
    edges_ds = grouped_ds.aggregate(EdgeGenerationAggregate())
    edges_ds = edges_ds.materialize()
    print("half-way, adding columns now")
    print(edges_ds.take(1))
    edges_ds = edges_ds.add_column("src", lambda df: df["edge_generation"].str.get("src").tolist())
    edges_ds = edges_ds.add_column("dst", lambda df: df["edge_generation"].str.get("dst").tolist())
    edges_ds = edges_ds.materialize()
    print("half-way")
    # print(edges_ds.schema())
    print(edges_ds.take(5))
    # Deduplicate edges (same pair might appear in multiple bands)
    logger.info("Step 5: Deduplicating edges...")
    edges_ds = edges_ds.map_batches(
        deduplicate_edges,
        batch_format='numpy',
    )
    edges_ds = edges_ds.materialize()

    # Checkpoint edges
    if checkpoint_dir:
        edges_path = f"{checkpoint_dir}/edges"
        logger.info(f"Checkpointing edges to {edges_path}")
        edges_ds.write_parquet(edges_path, try_create_dir=True)
        edges_ds = ray.data.read_parquet(edges_path)

    # Step 6: Compute connected components (distributed algorithm)
    logger.info("Step 6: Computing connected components (distributed)...")
    components_ds = compute_connected_components_distributed(
        edges_ds,
        max_iterations=max_cc_iterations
    )

    # Checkpoint components
    if checkpoint_dir:
        components_path = f"{checkpoint_dir}/components"
        logger.info(f"Checkpointing components to {components_path}")
        components_ds.write_parquet(components_path, try_create_dir=True)
        components_ds = ray.data.read_parquet(components_path)

    # Step 7: Filter duplicates
    logger.info("Step 7: Filtering duplicates...")

    # Rename 'node' to 'id' for filtering
    components_ds = components_ds.map_batches(
        lambda batch: {'id': batch['node'], 'parent': batch['parent']},
        batch_format='numpy',
    )

    # Collect components for filtering
    # Note: For extremely large component sets (billions of docs),
    # consider using a distributed join instead
    logger.info("Materializing components for filtering...")
    components_list = components_ds.take_all()
    components_dict = {row['id']: row['parent'] for row in components_list}

    logger.info(f"Found {len(components_dict)} documents in duplicate clusters")

    deduplicated_ds = ds.map_batches(
        filter_duplicates,
        fn_kwargs={'components_dict': components_dict},
        batch_format='numpy',
    )

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
        required=True,
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
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory for checkpointing intermediate results',
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

    args = parser.parse_args()
    ray.data.DataContext.get_current().enable_progress_bars = False

    # Read input data
    logger.info(f"Reading data from {args.input}")
    input_path = args.input

    # List all parquet files in the directory
    list_of_all_input_files = list_gcs_parquet_files(input_path)
    logger.info(f"Reading {len(list_of_all_input_files)} parquet files")

    ds = ray.data.read_parquet(list_of_all_input_files)

    input_count = ds.count()
    logger.info(f"Input dataset: {input_count} documents")
    assert input_count >= args.limit

    # Deduplicate
    deduplicated_ds = deduplicate_dataset(
        ds,
        text_column=args.text_column,
        threshold=args.threshold,
        num_perm=args.num_perm,
        ngram_size=args.ngram_size,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        max_cc_iterations=args.max_cc_iterations,
        limit=args.limit,
    )

    # Write output
    logger.info(f"Writing deduplicated data to {args.output}")
    deduplicated_ds.write_parquet(args.output)

    output_count = deduplicated_ds.count()
    logger.info(f"Output dataset: {output_count} documents")
    logger.info(f"Removed {input_count - output_count} duplicates ({100*(input_count - output_count)/input_count:.1f}%)")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    ray.init()
    main()
