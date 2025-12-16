import re
from pathlib import Path 
import pandas as pd

A64_SRC = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\A64 - NACE codes.csv")
NACE_CROSSWALK = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\14. EXIOBASE NACE ISIC crosswalk_with_division_codes.csv")
OUT_SUMMARY = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\A64_ISIC_summary.csv")
OUT_FULL = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\A64_NACE_ISIC_concordance_full.csv")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

def expand_patterns(cell: str):
    if pd.isna(cell) or str(cell).strip() == '':
        return []
    return [p.strip() for p in str(cell).split(';') if p.strip()]

def join_unique(series):
    vals = [v for v in sorted(set(series.dropna().astype(str).str.strip())) if v and v.lower() != 'nan']
    return ';'.join(vals) if vals else ''

def pattern_is_whole_number(p: str) -> bool:
    return bool(re.fullmatch(r'\d+', str(p).strip()))

def load_csv_replace_errors(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return pd.read_csv(fh, dtype=str)

def build_concordance(a64_df: pd.DataFrame, nace_df: pd.DataFrame) -> pd.DataFrame:
    all_nace_codes = nace_df['NACE Code'].dropna().astype(str).str.strip().unique().tolist()
    rows = []
    for _, r in a64_df.iterrows():
        a64_cat = r.get('A64', '')
        patterns = expand_patterns(r.get('NACE_codes', ''))
        if not patterns:
            continue
        for pat in patterns:
            pat_norm = pat.strip()
            for code in all_nace_codes:
                # Check if code matches pattern
                match = False
                if pat_norm.isdigit() and len(pat_norm) == 1:
                    # Single digit pattern like "1" -> match only "1" or "1.x"
                    if code == pat_norm or re.match(rf'^{re.escape(pat_norm)}\.', code):
                        match = True
                elif '.' in pat_norm:
                    # Decimal pattern like "3.1" or "05.2" -> match "3.1", "3.11", "3.12", etc.
                    if code == pat_norm or code.startswith(pat_norm):
                        match = True
                else:
                    # Multi-digit pattern like "05" -> exact match or prefix with dot
                    if code == pat_norm or code.startswith(pat_norm + '.'):
                        match = True
                
                if match:
                    rows.append({
                        'A64': a64_cat,
                        'NACE_pattern': pat_norm,
                        'NACE_code': code
                    })
    return pd.DataFrame(rows).drop_duplicates().sort_values(['A64', 'NACE_code']).reset_index(drop=True)

def main():
    a64_df = normalize_cols(load_csv_replace_errors(A64_SRC))
    nace_df = normalize_cols(load_csv_replace_errors(NACE_CROSSWALK))

    # Build concordance A64 -> NACE codes
    concordance = build_concordance(a64_df, nace_df)
    if concordance.empty:
        print("No concordance rows generated.")
        return

    # Prepare NACE->ISIC columns (include unique/code columns if present)
    keep_cols = ['NACE level', 'NACE Code','ISIC Group', 'ISIC Division', 'ISIC Section']
    # include possible code columns
    for extra in ['ISIC Unique Group code', 'ISIC Division code']:
        if extra in nace_df.columns and extra not in keep_cols:
            keep_cols.append(extra)
    nace_cols = nace_df[keep_cols].drop_duplicates().copy()

    # Standardize column names
    rename_map = {'NACE Code': 'NACE_code'}
    if 'ISIC Unique Group code' in nace_cols.columns:
        rename_map['ISIC Unique Group code'] = 'ISIC_Group_code'
    if 'ISIC Division code' in nace_cols.columns:
        rename_map['ISIC Division code'] = 'ISIC_Division_code'
    if 'NACE level' in nace_cols.columns:
        rename_map['NACE level'] = 'NACE_level'
    nace_cols = nace_cols.rename(columns=rename_map)

    # Merge to attach ISIC metadata
    concordance_with_isic = concordance.merge(nace_cols, on='NACE_code', how='left')

    # One-row-per-A64 summary: collect semicolon-separated NACE and ISIC codes/names
    agg_dict = {
        'NACE level': ('NACE_level', lambda s: join_unique(s)),
        'NACE_codes': ('NACE_code', lambda s: join_unique(s)),
        'ISIC_Group_codes': ('ISIC_Group_code', lambda s: join_unique(s)),
        'ISIC_Groups': ('ISIC Group', lambda s: join_unique(s)),
        'ISIC_Division_codes': ('ISIC_Division_code', lambda s: join_unique(s)),
        'ISIC_Divisions': ('ISIC Division', lambda s: join_unique(s)),
        'ISIC_Sections': ('ISIC Section', lambda s: join_unique(s))
    }

    summary = concordance_with_isic.groupby('A64').agg(**agg_dict).reset_index()

    # Detect A64 categories defined by whole-number aggregate patterns
    patterns_by_a64 = concordance.groupby('A64')['NACE_pattern'].apply(lambda s: sorted(set([p for p in s if p and str(p).strip() != '']))).to_dict()

    # Ensure all code columns are strings (empty if missing)
    for col in ['NACE_codes', 'ISIC_Group_codes', 'ISIC_Groups',
                'ISIC_Division_codes', 'ISIC_Divisions', 'ISIC_Sections']:
        if col in summary.columns:
            summary[col] = summary[col].fillna('')

    # Reorder columns: A64, then NACE, then ISIC (codes first, then names)
    col_order = ['A64', 'NACE level', 'NACE_codes', 
                 'ISIC_Group_codes', 'ISIC_Groups', 'ISIC_Division_codes', 
                 'ISIC_Divisions', 'ISIC_Sections']
    summary = summary[[c for c in col_order if c in summary.columns]]

    # Save outputs
    summary.to_csv(OUT_SUMMARY, index=False)
    concordance_with_isic.to_csv(OUT_FULL, index=False)

    print(f"✓ Saved one-row-per-A64 summary: {OUT_SUMMARY}")
    print(f"✓ Saved full concordance (A64 x NACE x ISIC): {OUT_FULL}")
    print(f"\nSummary columns: {', '.join(summary.columns)}")

if __name__ == '__main__':
    main()