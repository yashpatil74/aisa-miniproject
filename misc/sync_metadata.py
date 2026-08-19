import os
import pandas as pd

def sync_metadata(csv_path: str, fna_dir: str, output_path: str):
    """
    Filters a metadata CSV to only retain rows corresponding to .fna files 
    that exist in the specified directory.
    """
    # 1. Load the metadata CSV
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure 'Genome ID' is treated as a string for exact matching
    df['Genome ID'] = df['Genome ID'].astype(str)
    original_count = len(df)

    # 2. Collect Genome IDs from remaining .fna files
    print(f"Scanning directory '{fna_dir}' for .fna files...")
    existing_files = os.listdir(fna_dir)
    
    existing_genome_ids = set()
    for filename in existing_files:
        if filename.endswith('.fna'):
            # Extracts '573.66266' from '573.66266.fna'
            genome_id = filename[:-4]
            existing_genome_ids.add(genome_id)

    print(f"Found {len(existing_genome_ids)} .fna files after QC.")

    # 3. Filter DataFrame to keep only existing Genome IDs
    synced_df = df[df['Genome ID'].isin(existing_genome_ids)].copy()
    synced_count = len(synced_df)

    # 4. Save the synchronized metadata CSV
    synced_df.to_csv(output_path, index=False)
    
    print("\n--- Synchronization Summary ---")
    print(f"Original CSV rows: {original_count}")
    print(f"Dropped rows:      {original_count - synced_count}")
    print(f"Synced CSV rows:   {synced_count}")
    print(f"Saved synchronized metadata to: {output_path}")

if __name__ == "__main__":
    # Update these paths to match your local setup
    CSV_INPUT_PATH = "metadata_combined.csv"        # Path to your original metadata CSV
    FNA_DIRECTORY = "../fna_files"                    # Path to the folder containing .fna files
    CSV_OUTPUT_PATH = "clean_labels_combined.csv"   # Path where synced CSV should be saved

    sync_metadata(CSV_INPUT_PATH, FNA_DIRECTORY, CSV_OUTPUT_PATH)