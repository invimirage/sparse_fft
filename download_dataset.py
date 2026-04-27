#!/usr/bin/env python3
"""
Download matrices from SuiteSparse using ssget and clean them.
Only keeps COO indices, makes values all ones, converts to real-general format.
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix
import tempfile
import shutil


def list_matrix_names(directory):
    """List matrix names (without extension) in a directory."""
    if not directory or not os.path.isdir(directory):
        return set()
    names = set()
    for filename in os.listdir(directory):
        if filename.endswith('.mtx'):
            names.add(os.path.splitext(filename)[0])
    return names


def list_matrix_paths(directory):
    """List matrix paths keyed by matrix name (without extension)."""
    if not directory or not os.path.isdir(directory):
        return {}
    paths = {}
    for filename in os.listdir(directory):
        if filename.endswith('.mtx'):
            paths[os.path.splitext(filename)[0]] = os.path.join(directory, filename)
    return paths


def format_range(min_val, max_val):
    """Format numeric range for display."""
    if min_val is None and max_val is None:
        return "any"
    if min_val is None:
        return f"<= {max_val}"
    if max_val is None:
        return f">= {min_val}"
    return f"{min_val} - {max_val}"


def mat_value(mat, *names, default=None):
    """Safely read ssgetpy Matrix or dict fields."""
    if isinstance(mat, dict):
        for name in names:
            if name in mat and mat[name] is not None:
                return mat[name]
        return default
    for name in names:
        if hasattr(mat, name):
            value = getattr(mat, name)
            if value is not None:
                return value
    return default


def select_mtx_file(root_dir, mat_name):
    """Select the best .mtx file for a matrix."""
    candidates = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.mtx'):
                path = os.path.join(root, filename)
                base = os.path.splitext(filename)[0]
                candidates.append((path, base))

    if not candidates:
        return None

    for path, base in candidates:
        if base == mat_name:
            return path

    preferred = [item for item in candidates if item[1].startswith(mat_name)]
    if preferred:
        return max(preferred, key=lambda item: os.path.getsize(item[0]))[0]

    return max(candidates, key=lambda item: os.path.getsize(item[0]))[0]


def is_valid_matrix_file(path, min_nnz=1):
    """Check if a matrix file is readable and non-empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        matrix = mmread(path)
        if not isinstance(matrix, coo_matrix):
            matrix = coo_matrix(matrix)
        return matrix.nnz >= min_nnz
    except Exception:
        return False


def install_ssgetpy():
    """Install ssgetpy if not available."""
    try:
        import ssgetpy
        return True
    except ImportError:
        print("ssgetpy not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ssgetpy"])
            import ssgetpy
            return True
        except Exception as e:
            print(f"Failed to install ssgetpy: {e}")
            return False


def combine_bounds(min_a, max_a, min_b, max_b):
    """Combine row/col bounds for the initial SuiteSparse search."""
    lower_bounds = [value for value in (min_a, min_b) if value is not None]
    upper_bounds = [value for value in (max_a, max_b) if value is not None]
    lower = min(lower_bounds) if lower_bounds else None
    upper = max(upper_bounds) if upper_bounds else None
    return (lower, upper)


def download_matrices(output_dir, min_rows=1000, max_rows=10000,
                      min_cols=None, max_cols=None,
                      min_nnz=None, max_nnz=None,
                      max_sparsity=None, min_sparsity=None,
                      limit=None, exclude_names=None, min_count=None,
                      reuse_paths=None):
    """
    Download matrices from SuiteSparse Collection.

    Args:
        output_dir: Directory to save downloaded matrices
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows
        min_cols: Minimum number of columns
        max_cols: Maximum number of columns
        min_nnz: Minimum number of nonzeros
        max_nnz: Maximum number of nonzeros
        max_sparsity: Maximum sparsity (1 - density)
        min_sparsity: Minimum sparsity (1 - density)
        limit: Maximum number of matrices to download (None = no limit)
        exclude_names: Set of matrix names to skip
        min_count: Minimum number of matrices to download
        reuse_paths: Mapping of existing matrix names to local file paths to symlink
    """
    try:
        import ssgetpy
    except ImportError:
        print("Error: ssgetpy is required but could not be installed.")
        return []

    print(f"Searching SuiteSparse Collection...")
    print(f"  Rows: {min_rows} - {max_rows}")
    print(f"  Cols: {format_range(min_cols, max_cols)}")
    print(f"  NNZ: {format_range(min_nnz, max_nnz)}")
    print(f"  Sparsity: {format_range(min_sparsity, max_sparsity)}")
    if exclude_names:
        print(f"  Excluding {len(exclude_names)} existing matrices by name")
    if reuse_paths:
        print(f"  Reusing {len(reuse_paths)} existing local matrices when available")

    # Get the matrix collection index
    search_limit = 100000
    search_rowbounds = combine_bounds(min_rows, max_rows, min_cols, max_cols)
    index = ssgetpy.search(
        rowbounds=search_rowbounds,
        nzbounds=(min_nnz, max_nnz),
        limit=search_limit
    )

    # Filter matrices based on criteria
    filtered = []
    excluded_existing = 0
    for mat in index:
        nrows = mat_value(mat, 'nrows', 'rows', default=0)
        ncols = mat_value(mat, 'ncols', 'cols', default=0)
        nnz = mat_value(mat, 'nnz', default=0)

        if nrows <= 0 or ncols <= 0:
            continue

        mat_name = mat_value(mat, 'name', default=None)
        if exclude_names and mat_name in exclude_names and not (reuse_paths and mat_name in reuse_paths):
            excluded_existing += 1
            continue

        if min_cols is not None and ncols < min_cols:
            continue
        if max_cols is not None and ncols > max_cols:
            continue

        if min_nnz is not None and nnz < min_nnz:
            continue
        if max_nnz is not None and nnz > max_nnz:
            continue

        if min_sparsity is None and max_sparsity is None:
            sparsity = None
        else:
            density = nnz / (nrows * ncols)
            sparsity = 1.0 - density

        # Check if matrix meets criteria
        if (min_rows <= nrows <= max_rows and
            (min_cols is None or min_cols <= ncols) and
            (max_cols is None or ncols <= max_cols) and
            (min_sparsity is None or sparsity >= min_sparsity) and
            (max_sparsity is None or sparsity <= max_sparsity) and
            mat_value(mat, 'kind', default='') != 'sequence'):  # Skip non-sparse formats
            filtered.append(mat)

    print(f"Found {len(filtered)} matrices matching criteria")
    if excluded_existing:
        print(f"Filtered out {excluded_existing} matrices already present")

    if min_count is not None and limit is not None and limit < min_count:
        print(f"Limit {limit} is less than min_count {min_count}; using {min_count}")
        limit = min_count

    if limit and len(filtered) > limit:
        filtered = filtered[:limit]
        print(f"Limiting to {limit} matrices")

    os.makedirs(output_dir, exist_ok=True)
    downloaded = []
    linked = []

    if limit is not None:
        target_count = limit
    elif min_count is not None and min_count > 0:
        target_count = min_count
    else:
        target_count = None
    for idx, mat in enumerate(filtered, 1):
        mat_name = mat_value(mat, 'name', default='')
        mat_group = mat_value(mat, 'group', default='')
        print(f"\n[{idx}/{len(filtered)}] Downloading {mat_group}/{mat_name}...")

        output_path = os.path.join(output_dir, f"{mat_name}.mtx")
        existing_path = reuse_paths.get(mat_name) if reuse_paths else None

        if existing_path and os.path.exists(existing_path):
            if os.path.lexists(output_path):
                if os.path.islink(output_path) and os.path.realpath(output_path) == os.path.realpath(existing_path):
                    print(f"  Symlink already exists at {output_path}, skipping")
                    linked.append(output_path)
                    if target_count is not None and len(downloaded) + len(linked) >= target_count:
                        print(f"Reached target of {target_count} matrices; stopping early")
                        break
                    continue
                print(f"  Output path already exists and will not be replaced: {output_path}")
            elif is_valid_matrix_file(existing_path):
                os.symlink(existing_path, output_path)
                linked.append(output_path)
                print(f"  Reused existing matrix via symlink: {output_path}")
                if target_count is not None and len(downloaded) + len(linked) >= target_count:
                    print(f"Reached target of {target_count} matrices; stopping early")
                    break
                continue
            else:
                print(f"  Existing local matrix invalid, downloading fresh copy: {existing_path}")

        if os.path.exists(output_path):
            if is_valid_matrix_file(output_path):
                print(f"  Already exists at {output_path}, skipping")
                downloaded.append(output_path)
                if target_count is not None and len(downloaded) + len(linked) >= target_count:
                    print(f"Reached target of {target_count} matrices; stopping early")
                    break
                continue
            print(f"  Existing file invalid, re-downloading: {output_path}")
            try:
                os.remove(output_path)
            except OSError as e:
                print(f"  Warning: failed to remove {output_path}: {e}")

        try:
            # Download to temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                mat_data = ssgetpy.fetch(
                    f"{mat_group}/{mat_name}",
                    location=tmpdir
                )

                # Find the .mtx file
                mtx_file = select_mtx_file(tmpdir, mat_name)

                if not mtx_file:
                    print(f"  Warning: No .mtx file found for {mat_name}")
                    continue

                # Clean and save the matrix
                if clean_matrix(mtx_file, output_path):
                    downloaded.append(output_path)
                    print(f"  Saved to {output_path}")
                else:
                    print(f"  Failed to clean {mat_name}")

        except Exception as e:
            print(f"  Error downloading {mat_name}: {e}")
            continue

        if target_count is not None and len(downloaded) + len(linked) >= target_count:
            print(f"Reached target of {target_count} matrices; stopping early")
            break

    print(f"\n{'='*60}")
    print(f"Found {len(filtered)} matrices matching criteria in SuiteSparse")
    print(f"Reused {len(linked)} matrices via symlink")
    print(f"Downloaded {len(downloaded)} new matrices to {output_dir}")
    if min_count is not None and len(downloaded) + len(linked) < min_count:
        print(f"Warning: only {len(downloaded) + len(linked)} matrices collected, fewer than {min_count}")
    print(f"{'='*60}")

    return {
        'downloaded': downloaded,
        'linked': linked,
        'matched': len(filtered),
    }


def clean_matrix(input_path, output_path):
    """
    Clean a matrix file:
    1. Keep only COO indices
    2. Make all values = 1.0
    3. Convert to real-general format

    Args:
        input_path: Path to input .mtx file
        output_path: Path to output .mtx file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the matrix
        matrix = mmread(input_path)

        # Convert to COO format
        if not isinstance(matrix, coo_matrix):
            matrix = coo_matrix(matrix)

        # Make all values = 1.0
        ones_data = np.ones(len(matrix.data), dtype=np.float64)

        # Create new matrix with all values = 1.0
        cleaned_matrix = coo_matrix(
            (ones_data, (matrix.row, matrix.col)),
            shape=matrix.shape,
            dtype=np.float64
        )

        # Ensure it's in real-general format by converting to float64
        # Remove duplicates and sort
        cleaned_matrix.sum_duplicates()

        # Write the cleaned matrix
        mmwrite(output_path, cleaned_matrix,
                comment=f"Cleaned matrix: all values = 1.0, real-general format",
                field='real', symmetry='general')

        return True

    except Exception as e:
        print(f"Error cleaning matrix: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download and clean matrices from SuiteSparse Collection'
    )
    parser.add_argument('output_dir', type=str,
                        help='Directory to save downloaded matrices')
    parser.add_argument('--min-rows', type=int, default=1000,
                        help='Minimum number of rows (default: 1000)')
    parser.add_argument('--max-rows', type=int, default=10000,
                        help='Maximum number of rows (default: 10000)')
    parser.add_argument('--min-cols', type=int, default=None,
                        help='Minimum number of columns (default: same as min-rows)')
    parser.add_argument('--max-cols', type=int, default=None,
                        help='Maximum number of columns (default: same as max-rows)')
    parser.add_argument('--min-nnz', type=int, default=None,
                        help='Minimum number of nonzeros (default: no limit)')
    parser.add_argument('--max-nnz', type=int, default=None,
                        help='Maximum number of nonzeros (default: no limit)')
    parser.add_argument('--min-sparsity', type=float, default=None,
                        help='Minimum sparsity (default: no limit)')
    parser.add_argument('--max-sparsity', type=float, default=None,
                        help='Maximum sparsity (default: no limit)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of matrices to download (default: no limit)')
    parser.add_argument('--min-count', type=int, default=1000,
                        help='Minimum number of matrices to download (default: 1000)')
    parser.add_argument('--exclude-dir', type=str,
                        default='/home/rzhang38/VenomTileSkipping/FFT-Graph-Sparse/dataset_combined',
                        help='Directory of existing matrices to exclude by name')
    parser.add_argument('--clean-only', type=str, default=None,
                        help='Only clean an existing matrix file (provide input path)')
    parser.add_argument('--clean-dir', type=str, default=None,
                        help='Clean all .mtx files in a directory')

    args = parser.parse_args()

    # Install ssgetpy if needed (unless we're only cleaning)
    if args.clean_only or args.clean_dir:
        pass  # Don't need ssgetpy for cleaning only
    else:
        if not install_ssgetpy():
            print("Error: Cannot proceed without ssgetpy")
            return 1

    # Clean single file mode
    if args.clean_only:
        print(f"Cleaning {args.clean_only}...")
        output_path = os.path.join(args.output_dir,
                                    os.path.basename(args.clean_only))
        os.makedirs(args.output_dir, exist_ok=True)
        if clean_matrix(args.clean_only, output_path):
            print(f"Cleaned matrix saved to {output_path}")
            return 0
        else:
            print("Failed to clean matrix")
            return 1

    # Clean directory mode
    if args.clean_dir:
        print(f"Cleaning all .mtx files in {args.clean_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)
        count = 0
        for filename in os.listdir(args.clean_dir):
            if filename.endswith('.mtx'):
                input_path = os.path.join(args.clean_dir, filename)
                output_path = os.path.join(args.output_dir, filename)
                print(f"  Cleaning {filename}...")
                if clean_matrix(input_path, output_path):
                    count += 1
        print(f"Cleaned {count} matrices")
        return 0

    # Download mode
    min_cols = args.min_cols if args.min_cols is not None else args.min_rows
    max_cols = args.max_cols if args.max_cols is not None else args.max_rows
    exclude_names = list_matrix_names(args.exclude_dir)
    reused_paths = list_matrix_paths(args.exclude_dir)
    results = download_matrices(
        args.output_dir,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_cols=min_cols,
        max_cols=max_cols,
        min_nnz=args.min_nnz,
        max_nnz=args.max_nnz,
        min_sparsity=args.min_sparsity,
        max_sparsity=args.max_sparsity,
        limit=args.limit,
        exclude_names=exclude_names,
        min_count=args.min_count,
        reuse_paths=reused_paths
    )

    collected = results['linked'] + results['downloaded']
    if collected:
        print(f"\nCollected matrices ({len(collected)} total, {results['matched']} matched):")
        for path in collected:
            print(f"  - {os.path.basename(path)}")
        return 0
    else:
        print("No matrices downloaded")
        return 1


if __name__ == "__main__":
    sys.exit(main())
