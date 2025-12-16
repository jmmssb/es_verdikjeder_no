import re
from pathlib import Path
import pandas as pd
import numpy as np  

SRC = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\14. EXIOBASE NACE ISIC crosswalk.csv")
OUT_MAP = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\isic_division_code_mapping.csv")

def slugify(name: str) -> str:
    """Convert name to uppercase slug for code generation."""
    s = re.sub(r'[^0-9A-Za-z]+', '_', name or '').strip('_')
    return s.upper()[:40] if s else ''

def infer_code_for_division(df: pd.DataFrame, division: str) -> str:
    """Infer ISIC Division code from ISIC Unique Group code or division name.
    If the numeric part is single-digit, pad with a leading zero to produce two digits.
    Keep the letter prefix taken from the group/class code when available.
    """
    rows = df[df["ISIC Division"].astype(str).str.strip() == division]
    
    # Try to extract the letter prefix and 1-2 digit code from ISIC Unique Group code
    if "ISIC Unique Group code" in df.columns:
        for ug in rows["ISIC Unique Group code"].dropna().astype(str).unique():
            m = re.search(r'([A-Z])_(\d{1,2})_', ug) or re.search(r'([A-Z])_(\d{1,2})$', ug)
            if m:
                num = int(m.group(2))
                num_s = f"{num:02d}"            # pad single digits
                return f"{m.group(1)}_{num_s}"  # Use the letter from the group code
    
    # Try to extract 1-2 digit from the division name and use letter from class/group if available
    m = re.search(r'(\d{1,2})', division)
    if m:
        num_s = f"{int(m.group(1)):02d}"
        # Prefer letter from ISIC Unique Group code first
        if "ISIC Unique Group code" in df.columns:
            ug_letters = rows["ISIC Unique Group code"].str.extract(r'([A-Z])')[0].dropna().unique()
            if ug_letters.size > 0:
                return f"{ug_letters[0]}_{num_s}"
        # Then try ISIC Class letter
        class_letter = rows["ISIC Class"].str.extract(r'([A-Z])')[0].dropna().unique()
        if class_letter.size > 0:
            return f"{class_letter[0]}_{num_s}"
        return f"D_{num_s}"  # Default to 'D' if no letter found
    
    # Fallback: create slug from division name
    slug = slugify(division)
    return f"D_{slug}" if slug else None  # Default to 'D' if slug is empty

def build_division_mapping(src: Path) -> pd.DataFrame:
    """Build mapping of ISIC Division -> ISIC Division code."""
    df = pd.read_csv(src, dtype=str).fillna('')
    df["ISIC Division"] = df["ISIC Division"].astype(str).str.strip()
    
    unique_divs = df["ISIC Division"].unique()
    mapping = []
    seen_codes = {}
    
    for div in sorted(unique_divs):
        code = infer_code_for_division(df, div)
        if not code:
            code = "D_UNKNOWN"
        
        # Ensure unique codes (avoid collisions)
        base = code
        i = 1
        while code in seen_codes and seen_codes[code] != div:
            i += 1
            code = f"{base}_{i}"
        seen_codes[code] = div
        
        mapping.append({"ISIC Division": div, "ISIC Division code": code})
    
    map_df = pd.DataFrame(mapping)
    map_df.to_csv(OUT_MAP, index=False)
    print(f"Saved division code mapping to: {OUT_MAP}")
    
    return map_df

def merge_division_codes_to_crosswalk(src: Path, map_df: pd.DataFrame) -> Path:
    """Merge division codes into original NACE/ISIC crosswalk."""
    df = pd.read_csv(src, dtype=str).fillna('')
    df["ISIC Division"] = df["ISIC Division"].astype(str).str.strip()
    
    # Merge on ISIC Division
    merged = df.merge(map_df, on="ISIC Division", how="left")
    
    # Save merged crosswalk
    out_path = src.with_name(src.stem + "_with_division_codes" + src.suffix)
    merged.to_csv(out_path, index=False)
    print(f"Saved merged crosswalk to: {out_path}")
    
    return out_path

def extract_division_nace_mapping(src: Path, map_df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique ISIC Division codes with their corresponding NACE codes."""
    df = pd.read_csv(src, dtype=str).fillna('')
    df["ISIC Division"] = df["ISIC Division"].astype(str).str.strip()
    
    # Merge division codes
    merged = df.merge(map_df, on="ISIC Division", how="left")
    
    # Extract unique combinations: Division code, Division name, NACE Code
    division_nace = merged[["ISIC Division code", "ISIC Division", "NACE Code", "NACE class"]].drop_duplicates()
    division_nace = division_nace.sort_values(["ISIC Division code", "NACE Code"])
    
    # Save to CSV
    out_path = src.parent / "division_nace_mapping.csv"
    division_nace.to_csv(out_path, index=False)
    print(f"Saved division-NACE mapping to: {out_path}")
    
    return division_nace

def main():
    """Main workflow: build mapping and merge into crosswalk."""
    print(f"Reading crosswalk from: {SRC}\n")
    
    # Step 1: Build division code mapping
    mapping = build_division_mapping(SRC)
    print(f"\nGenerated {len(mapping)} division codes:")
    print(mapping.to_string(index=False))
    
    # Step 2: Merge codes back into crosswalk
    out_crosswalk = merge_division_codes_to_crosswalk(SRC, mapping)
    
    # Step 3: Extract division-NACE mapping
    division_nace = extract_division_nace_mapping(SRC, mapping)
    print(f"\nDivision-NACE mapping:")
    print(division_nace.to_string(index=False))
    
    print(f"\n✓ Done!")

if __name__ == "__main__":
    main()