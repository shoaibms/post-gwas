"""
Download required files for motif delta-PWM analysis.

Downloads AGPv4 genome FASTA and JASPAR Plants PWMs.
"""

import urllib.request
import gzip
import shutil
from pathlib import Path

BASE = Path(r"C:\Users\ms\Desktop\gwas")

# Create directories
FASTA_DIR = BASE / "data" / "maize" / "ref" / "agpv4"
MOTIF_DIR = BASE / "data" / "motifs" / "jaspar_plants"
FASTA_DIR.mkdir(parents=True, exist_ok=True)
MOTIF_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("DOWNLOADING MOTIF ANALYSIS RESOURCES")
print("="*60)

# ============================================================
# DOWNLOAD 1: AGPv4 GENOME FASTA
# ============================================================

print("\n" + "="*60)
print("DOWNLOAD 1: AGPv4 GENOME FASTA")
print("="*60)

fasta_url = "https://download.maizegdb.org/Zm-B73-REFERENCE-GRAMENE-4.0/Zm-B73-REFERENCE-GRAMENE-4.0.fa.gz"
fasta_gz = FASTA_DIR / "Zm-B73-REFERENCE-GRAMENE-4.0.fa.gz"
fasta_final = FASTA_DIR / "Zm-B73-REFERENCE-GRAMENE-4.0.fa"

if fasta_final.exists():
    print(f"\nFASTA already exists: {fasta_final}")
    print(f"Size: {fasta_final.stat().st_size / 1024 / 1024:.1f} MB")
    print("Skipping download.")
else:
    print(f"\nDownloading from: {fasta_url}")
    print(f"Destination: {fasta_gz}")
    print("This may take 5-10 minutes (700 MB)...")
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
        
        urllib.request.urlretrieve(fasta_url, fasta_gz, reporthook=report_progress)
        print("\n  Download complete!")
        
        # Decompress
        print(f"\nDecompressing to: {fasta_final}")
        print("This may take 2-3 minutes...")
        
        with gzip.open(fasta_gz, 'rb') as f_in:
            with open(fasta_final, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"  Decompressed successfully!")
        print(f"  Size: {fasta_final.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Clean up compressed file
        fasta_gz.unlink()
        print(f"  Removed compressed file")
        
    except Exception as e:
        print(f"\n  ERROR downloading FASTA: {e}")
        print(f"\n  Manual download:")
        print(f"    URL: {fasta_url}")
        print(f"    Save to: {fasta_gz}")
        print(f"    Then decompress with 7-Zip or similar")

# ============================================================
# DOWNLOAD 2: JASPAR PLANTS PWMs
# ============================================================

print("\n" + "="*60)
print("DOWNLOAD 2: JASPAR PLANTS PWMs")
print("="*60)

jaspar_url = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt"
jaspar_file = MOTIF_DIR / "JASPAR2024_CORE_plants_non-redundant.meme"

if jaspar_file.exists():
    print(f"\nJASPAR PWMs already exist: {jaspar_file}")
    print(f"Size: {jaspar_file.stat().st_size / 1024:.1f} KB")
    print("Skipping download.")
else:
    print(f"\nDownloading from: {jaspar_url}")
    print(f"Destination: {jaspar_file}")
    print("This should be quick (~3 MB)...")
    
    try:
        def report_progress_small(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            kb_downloaded = downloaded / 1024
            kb_total = total_size / 1024
            print(f"\r  Progress: {percent:.1f}% ({kb_downloaded:.1f}/{kb_total:.1f} KB)", end="")
        
        urllib.request.urlretrieve(jaspar_url, jaspar_file, reporthook=report_progress_small)
        print("\n  Download complete!")
        print(f"  Size: {jaspar_file.stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"\n  ERROR downloading JASPAR: {e}")
        print(f"\n  Manual download:")
        print(f"    URL: {jaspar_url}")
        print(f"    Save to: {jaspar_file}")

# ============================================================
# VERIFICATION
# ============================================================

print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

all_good = True

print("\nChecking downloaded files:")

if fasta_final.exists():
    size_mb = fasta_final.stat().st_size / 1024 / 1024
    print(f"  OK FASTA: {fasta_final.name} ({size_mb:.1f} MB)")
    if size_mb < 500:
        print(f"    WARNING: File seems small, may be incomplete")
        all_good = False
else:
    print(f"  MISSING FASTA: Not found")
    all_good = False

if jaspar_file.exists():
    size_kb = jaspar_file.stat().st_size / 1024
    print(f"  OK JASPAR: {jaspar_file.name} ({size_kb:.1f} KB)")
    if size_kb < 100:
        print(f"    WARNING: File seems small, may be incomplete")
        all_good = False
else:
    print(f"  MISSING JASPAR: Not found")
    all_good = False

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("DOWNLOAD SUMMARY")
print("="*60)

if all_good:
    print("\nAll files downloaded successfully.")
    print("\nReady for motif delta-PWM analysis.")
    print("\nFiles saved to:")
    print(f"  FASTA: {fasta_final}")
    print(f"  JASPAR: {jaspar_file}")
else:
    print("\nSome files missing or incomplete")
    print("\nPlease check error messages above.")
    print("You may need to download manually if automatic download failed.")

print("\n" + "="*60)