#!/bin/bash

# Create a folder to store the data
mkdir -p fna_files
cd fna_files

echo "Starting download process..."

# Read the IDs line by line
while read -r GENOME_ID; do
    # Strip hidden carriage returns (CRLF issues from Windows)
    GENOME_ID=$(echo "$GENOME_ID" | tr -d '\r')
    
    # Skip empty lines
    if [ -z "$GENOME_ID" ]; then continue; fi
    
    # The BV-BRC FTP URL for raw DNA FASTA files
    URL="ftp://ftp.bvbrc.org/genomes/${GENOME_ID}/${GENOME_ID}.fna"
    
    # Use wget: -q (quiet), -O (output filename)
    wget -q -O "${GENOME_ID}.fna" "$URL"
    
    # Check if download was successful and file is not empty
    if [ $? -eq 0 ] && [ -s "${GENOME_ID}.fna" ]; then
        echo "✅ Downloaded: ${GENOME_ID}.fna"
    else
        echo "❌ Failed or Empty: ${GENOME_ID}.fna"
        echo "$GENOME_ID" >> ../failed_downloads.log
        # Delete the empty/corrupt file
        rm -f "${GENOME_ID}.fna" 
    fi
    
    # Polite sleep to avoid IP rate-limiting
    sleep 1
    
done < ../genome_ids.txt

echo "Download complete! Check the fna_files directory."