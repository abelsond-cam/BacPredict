#!/usr/bin/env python3
"""
Explore the structure of ESMC embedding parquet files.
This script samples a small number of files (1-10) and examines their structure.
"""

import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import sys
import numpy as np

DATA_DIR = Path("/projects/public/u5ah/data/genomes/atb/esmc-large")
NUM_FILES = 1

def explore_parquet_file(filepath):
    """Explore a single parquet file in detail."""
    print(f"\n{'='*80}")
    print(f"Exploring: {filepath.name}")
    print(f"{'='*80}")
    
    try:
        # Read parquet metadata without loading full file
        parquet_file = pq.ParquetFile(filepath)
        
        file_size_mb = filepath.stat().st_size / (1024**2)
        file_size_gb = filepath.stat().st_size / (1024**3)
        print(f"\nFile size: {file_size_mb:.2f} MB ({file_size_gb:.4f} GB)")
        print(f"Number of row groups: {parquet_file.num_row_groups}")
        print(f"Number of rows: {parquet_file.metadata.num_rows:,}")
        
        # Get schema
        schema = parquet_file.schema_arrow
        column_names = [field.name for field in schema]
        print(f"\nSchema ({len(schema)} columns):")
        for i, field in enumerate(schema):
            print(f"  {i+1}. {field.name}: {field.type}")
        
        # Read a small sample to understand structure
        print(f"\n{'='*80}")
        print("Reading sample data (first 5 rows)...")
        print(f"{'='*80}")
        df_full = pd.read_parquet(filepath, engine='pyarrow')
        df_sample = df_full.head(5)
        
        print(f"\nDataFrame shape: {df_sample.shape}")
        print(f"\nColumn names: {list(df_sample.columns)}")
        
        # Check for genome_id or genome_name column
        has_genome_id = 'genome_id' in column_names
        has_genome_name = 'genome_name' in column_names
        genome_key_col = 'genome_id' if has_genome_id else ('genome_name' if has_genome_name else None)
        
        print(f"\nHas 'genome_id' column: {has_genome_id}")
        print(f"Has 'genome_name' column: {has_genome_name}")
        
        if genome_key_col:
            print(f"\nUsing '{genome_key_col}' as genome key")
            print(f"\nSample {genome_key_col} values:")
            for idx, name in enumerate(df_sample[genome_key_col].head()):
                print(f"  Row {idx}: {name}")
            
            # Check total unique genome IDs/names
            print(f"\nCounting unique {genome_key_col} values...")
            num_unique_genomes = df_full[genome_key_col].nunique()
            print(f"  Total rows: {len(df_full):,}")
            print(f"  Unique {genome_key_col} values: {num_unique_genomes:,}")
            if num_unique_genomes == len(df_full):
                print(f"  ✓ Each row has a unique {genome_key_col} (one genome per row)")
            else:
                print(f"  ⚠ Multiple rows per {genome_key_col}")
        else:
            num_unique_genomes = None
        
        # Examine each column type and structure
        print(f"\n{'='*80}")
        print("Column Structure Analysis")
        print(f"{'='*80}")
        
        for col in df_sample.columns:
            print(f"\nColumn: {col}")
            dtype = df_sample[col].dtype
            print(f"  Pandas dtype: {dtype}")
            
            # Get first non-null value to examine structure
            first_val = df_sample[col].dropna().iloc[0] if not df_sample[col].dropna().empty else None
            
            if first_val is not None:
                print(f"  Type of first value: {type(first_val).__name__}")
                
                # Check if it's a list/array
                if isinstance(first_val, (list, np.ndarray)):
                    # Handle nested lists (like list of lists)
                    if isinstance(first_val, list) and len(first_val) > 0 and isinstance(first_val[0], list):
                        print(f"  Nested list structure: list of {len(first_val)} lists")
                        print(f"  First sub-list length: {len(first_val[0])}")
                        print(f"  First sub-list type: {type(first_val[0][0]).__name__ if first_val[0] else 'empty'}")
                        # Try to get total elements
                        total_elements = sum(len(sublist) for sublist in first_val if isinstance(sublist, list))
                        print(f"  Total elements across all sub-lists: {total_elements:,}")
                    else:
                        arr = np.array(first_val) if isinstance(first_val, list) else first_val
                        print(f"  Array/list shape: {arr.shape}")
                        print(f"  Array/list dtype: {arr.dtype}")
                        print(f"  Array/list length: {len(first_val) if isinstance(first_val, list) else arr.size}")
                        if arr.size > 0 and arr.size <= 20:
                            print(f"  Array/list values: {arr.flatten()[:20]}")
                        elif arr.size > 20:
                            print(f"  Array/list sample values (first 10): {arr.flatten()[:10]}")
                elif isinstance(first_val, str):
                    print(f"  String length: {len(first_val)}")
                    if len(first_val) > 100:
                        print(f"  String preview: {first_val[:100]}...")
                    else:
                        print(f"  String value: {first_val}")
                else:
                    print(f"  Value: {first_val}")
        
        # Look for list/array columns (potential embeddings)
        print(f"\n{'='*80}")
        print("List/Array Columns (potential embeddings)")
        print(f"{'='*80}")
        list_cols = []
        for col in df_sample.columns:
            first_val = df_sample[col].dropna().iloc[0] if not df_sample[col].dropna().empty else None
            if first_val is not None and isinstance(first_val, (list, np.ndarray)):
                list_cols.append(col)
        
        if list_cols:
            for col in list_cols:
                first_val = df_sample[col].dropna().iloc[0]
                print(f"\n  {col}:")
                # Handle nested lists
                if isinstance(first_val, list) and len(first_val) > 0 and isinstance(first_val[0], list):
                    print(f"    Type: Nested list (list of lists)")
                    print(f"    Number of sub-lists: {len(first_val)}")
                    print(f"    First sub-list length: {len(first_val[0]) if first_val[0] else 0}")
                    total_elements = sum(len(sublist) for sublist in first_val if isinstance(sublist, list))
                    print(f"    Total elements: {total_elements:,}")
                else:
                    arr = np.array(first_val) if isinstance(first_val, list) else first_val
                    print(f"    Shape: {arr.shape}")
                    print(f"    Dtype: {arr.dtype}")
                    print(f"    Total elements: {arr.size:,}")
        else:
            print("\n  No list/array columns found in sample")
        
        return {
            'filepath': str(filepath),
            'filename': filepath.name,
            'size_mb': file_size_mb,
            'num_rows': parquet_file.metadata.num_rows,
            'num_columns': len(column_names),
            'columns': column_names,
            'has_genome_id': has_genome_id,
            'has_genome_name': has_genome_name,
            'genome_key_col': genome_key_col,
            'num_unique_genomes': num_unique_genomes
        }
        
    except Exception as e:
        print(f"\nError reading {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Constants
    data_dir = DATA_DIR
    # Allow user to specify how many files to sample (default NUM_FILES)
    num_files = NUM_FILES

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        print("Please check the path and try again.")
        sys.exit(1)
    
    # Find all parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(parquet_files):,} parquet file(s) in directory")
    
    
    if len(sys.argv) > 1:
        try:
            num_files = int(sys.argv[1])
            num_files = min(num_files, 10)  # Cap at 10
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}, using default of 1")
    
    print(f"Sampling {num_files} file(s) for detailed exploration\n")
    
    # Sample files
    files_to_explore = sorted(parquet_files)[:num_files]
    
    # Explore each file
    results = []
    for parquet_file in files_to_explore:
        result = explore_parquet_file(parquet_file)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Files explored: {len(results)}")
        print(f"\nColumn names (common across explored files):")
        all_columns = set(results[0]['columns'])
        for r in results[1:]:
            all_columns = all_columns.intersection(set(r['columns']))
        
        if all_columns:
            for col in sorted(all_columns):
                print(f"  - {col}")
        
        print(f"\nFile statistics:")
        for r in results:
            print(f"\n  {r['filename']}:")
            print(f"    Size: {r['size_mb']:.2f} MB")
            print(f"    Rows: {r['num_rows']:,}")
            print(f"    Columns: {r['num_columns']}")
            if r['genome_key_col']:
                print(f"    Unique genomes ({r['genome_key_col']}): {r['num_unique_genomes']:,}")

if __name__ == "__main__":
    main()
