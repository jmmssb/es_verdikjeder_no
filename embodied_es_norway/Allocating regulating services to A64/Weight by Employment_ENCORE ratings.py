from pathlib import Path
import pandas as pd
import numpy as np

# INPUTS - adjust paths
A64_NACE_MAPPING = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\A64 - NACE codes.csv")  # A64 with NACE patterns
CONCORD = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\A64_NACE_ISIC_concordance_full.csv")
EMPLOYMENT_BY_DETAILED_CODE = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\SN07_employed.csv")  # columns: ID_code, employed_persons (detailed with decimals)
ISIC_RATINGS = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\ecosystem_services_weighted_by_isic_division.csv")
ISIC_UNITS = None  # optional
OUT = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\A64_service_weighted_by_employment.csv")

SERVICE_TOTALS = {
    'Air Filtration_value': 63979.28,
    'Flood mitigation services_value': 165855.72,
    'Local (micro and meso) climate regulation_value': 1.6565,
}

def find_isic_col(df):
    for c in df.columns:
        if c.strip().lower() in ('isic division','isic_division','isic_division_code','isic division code'):
            return c
    return df.columns[0]

def load_concordance(path):
    df = pd.read_csv(path, dtype=str).fillna('')
    cols = {c.lower(): c for c in df.columns}
    a64 = cols.get('a64') or next((c for c in df.columns if 'a64' in c.lower()), None)
    nace = cols.get('nace_code') or next((c for c in df.columns if 'nace' in c.lower()), None)
    isic = next((c for c in df.columns if 'isic' in c.lower() and 'division' in c.lower()), None)
    if not (a64 and nace and isic):
        raise SystemExit(f"Concordance missing columns. Found: {df.columns.tolist()}")
    df = df[[a64, nace, isic]].rename(columns={a64:'A64', nace:'NACE_code', isic:'ISIC_division_code'})
    df['A64'] = df['A64'].astype(str).str.strip()
    df['NACE_code'] = df['NACE_code'].astype(str).str.strip()
    df['ISIC_division_code'] = df['ISIC_division_code'].astype(str).str.strip()
    return df

def load_employment_data(path, a64_nace_file):
    """
    Load employment data with detailed SN07/NACE codes.
    Smart aggregation: keep codes that exist in NACE mapping (like 3.1, 3.2),
    aggregate detailed sub-codes (like 10.00, 10.01 → 10).
    """
    # Load NACE codes to know which to preserve
    nace_df = pd.read_csv(a64_nace_file, dtype=str).fillna('')
    nace_codes_str = nace_df[nace_df.columns[0]].str.cat(sep=';')  # First column has NACE codes
    # Split all grouped codes and extract unique NACE codes
    all_nace = set()
    for code_group in nace_codes_str.split(';'):
        code = code_group.strip()
        if code and code != 'nan':
            all_nace.add(code)
    
    print(f"\nFound {len(all_nace)} unique NACE codes in A64 mapping")
    
    df = pd.read_csv(path, dtype=str).fillna('')
    
    # Look for id/code column and employment column
    id_col = next((c for c in df.columns if any(k in c.lower() for k in ('id_sn07', 'id', 'sn07_code', 'code'))), None)
    emp_col = next((c for c in df.columns if any(k in c.lower() for k in ('employ','person','worker'))), None)
    
    if not (id_col and emp_col):
        raise SystemExit(f"Employment file missing columns. Found: {df.columns.tolist()}")
    
    df = df[[id_col, emp_col]].rename(columns={id_col:'ID_code', emp_col:'Employed'})
    df['ID_code'] = df['ID_code'].astype(str).str.strip()
    df['Employed'] = pd.to_numeric(df['Employed'], errors='coerce').fillna(0.0)
    
    # Filter out row if ID_code is '0' (likely totals row)
    df = df[df['ID_code'] != '0']
    
    print(f"Loaded {len(df)} detailed employment records")
    
    # Smart aggregation: preserve codes in NACE, aggregate or distribute others
    def get_aggregation_key(code):
        # If code exists in NACE mapping, keep it as-is
        if code in all_nace:
            return code
        
        # Check if this is a "parent" code where child codes exist in NACE
        # E.g., code "3" when "3.1" and "3.2" exist in NACE
        if '.' not in code:
            # Check if any x.y codes exist for this parent
            child_codes = [nc for nc in all_nace if nc.startswith(code + '.')]
            if child_codes:
                # Parent code should be distributed to children
                # Mark for special handling
                return f"DISTRIBUTE_{code}"
        
        # Otherwise, aggregate to appropriate level
        if '.' in code:
            parts = code.split('.')
            main = parts[0]
            
            # Check if main code exists in NACE
            if main in all_nace:
                return main
            
            # Check if there's a detailed version (e.g., 3.11 → 3.1)
            if len(parts) >= 2 and len(parts[1]) >= 2:
                first_decimal = f"{parts[0]}.{parts[1][0]}"
                if first_decimal in all_nace:
                    return first_decimal
            
            # Default: aggregate to main code (10.0 → 10, 10.11 → 10)
            return main
        return code
    
    df['Agg_code'] = df['ID_code'].apply(get_aggregation_key)
    
    # Handle distribution cases: split parent employment among children
    distribute_rows = df[df['Agg_code'].str.startswith('DISTRIBUTE_', na=False)]
    if len(distribute_rows) > 0:
        print(f"Distributing {len(distribute_rows)} parent codes to child codes...")
        
        for _, row in distribute_rows.iterrows():
            parent_code = row['Agg_code'].replace('DISTRIBUTE_', '')
            children = [nc for nc in all_nace if nc.startswith(parent_code + '.')]
            
            if children:
                # Distribute employment equally among children
                per_child = row['Employed'] / len(children)
                for child in children:
                    # Add distributed employment to existing child or create new row
                    df = pd.concat([df, pd.DataFrame([{
                        'ID_code': row['ID_code'],
                        'Employed': per_child,
                        'Agg_code': child
                    }])], ignore_index=True)
        
        # Remove original DISTRIBUTE_ rows
        df = df[~df['Agg_code'].str.startswith('DISTRIBUTE_', na=False)]
    
    # Sum employment by aggregation key
    aggregated = df.groupby('Agg_code', as_index=False)['Employed'].sum()
    aggregated = aggregated.rename(columns={'Agg_code': 'ID_code'})
    
    print(f"Aggregated to {len(aggregated)} categories (preserved codes in NACE mapping)")
    
    return aggregated

def load_a64_nace_patterns(path):
    """Load A64-NACE mapping with NACE patterns."""
    df = pd.read_csv(path, dtype=str).fillna('')
    cols = {c.lower(): c for c in df.columns}
    a64_col = cols.get('a64') or next((c for c in df.columns if 'a64' in c.lower()), None)
    nace_col = next((c for c in df.columns if 'nace' in c.lower()), None)
    if not (a64_col and nace_col):
        raise SystemExit(f"A64-NACE file missing columns. Found: {df.columns.tolist()}")
    df = df[[a64_col, nace_col]].rename(columns={a64_col:'A64', nace_col:'NACE_code'})
    return df

def map_employment_to_a64(employment_df, a64_nace_patterns):
    """
    Map employment data to A64 categories via NACE codes.
    Prioritizes exact matches over main code matches.
    Takes first match to avoid double-counting.
    NACE codes may be grouped with semicolons like "10;11;12"
    Returns DataFrame with: ID_code, A64, Employed
    """
    print("\nMapping employment data to A64 via NACE patterns...")
    
    results = []
    unmatched_codes = set()
    
    for _, emp_row in employment_df.iterrows():
        id_code = emp_row['ID_code']
        employed = emp_row['Employed']
        
        # Try exact match first
        exact_match_a64 = None
        for _, pattern_row in a64_nace_patterns.iterrows():
            a64 = pattern_row['A64']
            nace_codes_str = str(pattern_row['NACE_code']).strip()
            
            if not nace_codes_str or nace_codes_str == 'nan':
                continue
            
            # Split by semicolon to handle grouped codes like "10;11;12"
            nace_codes = [nc.strip() for nc in nace_codes_str.split(';') if nc.strip()]
            
            for nace_code in nace_codes:
                if id_code == nace_code:
                    exact_match_a64 = a64
                    break
            
            if exact_match_a64:
                break
        
        if exact_match_a64:
            results.append({
                'ID_code': id_code,
                'A64': exact_match_a64,
                'Employed': employed
            })
            continue
        
        # If no exact match, try main code match
        main_match_a64 = None
        id_main = id_code.split('.')[0] if '.' in id_code else id_code
        
        for _, pattern_row in a64_nace_patterns.iterrows():
            a64 = pattern_row['A64']
            nace_codes_str = str(pattern_row['NACE_code']).strip()
            
            if not nace_codes_str or nace_codes_str == 'nan':
                continue
            
            nace_codes = [nc.strip() for nc in nace_codes_str.split(';') if nc.strip()]
            
            for nace_code in nace_codes:
                nace_main = nace_code.split('.')[0] if '.' in nace_code else nace_code
                
                if id_main == nace_main:
                    main_match_a64 = a64
                    break
            
            if main_match_a64:
                break
        
        if main_match_a64:
            results.append({
                'ID_code': id_code,
                'A64': main_match_a64,
                'Employed': employed
            })
        else:
            unmatched_codes.add(id_code)
    
    # Report unmatched codes
    if unmatched_codes:
        print(f"\nWarning: {len(unmatched_codes)} codes not matched to A64:")
        for code in sorted(unmatched_codes)[:10]:
            print(f"  - {code}")
        if len(unmatched_codes) > 10:
            print(f"  ... and {len(unmatched_codes) - 10} more")
    
    result_df = pd.DataFrame(results)
    print(f"Mapped {len(result_df)} employment records to A64 categories")
    
    return result_df

def load_ratings(path):
    df = pd.read_csv(path, dtype=str).fillna('')
    isic_col = find_isic_col(df)
    rating_cols = [c for c in df.columns if c!=isic_col and (c.lower().endswith('_value') or 'value' in c.lower() or c.lower().endswith('_rating') or
                    pd.to_numeric(df[c], errors='coerce').notna().any())]
    df = df[[isic_col] + rating_cols].rename(columns={isic_col:'ISIC_division_code'})
    for c in rating_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df, rating_cols

def load_units(path_or_none, ratings_df, rating_cols):
    if path_or_none:
        df = pd.read_csv(path_or_none, dtype=str).fillna('')
        isic_col = find_isic_col(df)
        unit_cols = [c for c in df.columns if c!=isic_col and any(rc.split('_value')[0].strip().lower() in c.lower() for rc in rating_cols)]
        df = df[[isic_col] + unit_cols].rename(columns={isic_col:'ISIC_division_code'})
        for c in unit_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        return df, unit_cols
    unit_cols = [rc.replace('_value','_unit') for rc in rating_cols if (rc.replace('_value','_unit') in ratings_df.columns)]
    if unit_cols:
        df = ratings_df[['ISIC_division_code'] + unit_cols].copy()
        for c in unit_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        return df, unit_cols
    return None, []

def aggregate_employment_to_isic(employment_a64_df, concordance):
    """
    Map A64 employment to ISIC divisions via NACE-ISIC concordance.
    Returns employment at (A64, ISIC_division_code) level.
    """
    print("\nAggregating employment to ISIC division level...")
    
    # Use NACE code (ID_code) to map to ISIC divisions
    # Get NACE-ISIC mapping from concordance
    nace_isic = concordance[['A64', 'NACE_code', 'ISIC_division_code']].drop_duplicates()
    
    # Create matching keys for concordance
    # For codes like 3.11, 49.10, etc., create prefix "3.1", "49.1"
    nace_isic['NACE_str'] = nace_isic['NACE_code'].astype(str)
    nace_isic['NACE_prefix'] = nace_isic['NACE_str'].str[:3].str.rstrip('.')  # First 3 chars or less
    nace_isic['NACE_main'] = nace_isic['NACE_str'].str.split('.').str[0]      # Main code (e.g., 3, 49)
    
    # Create matching keys for employment
    employment_a64_df = employment_a64_df.copy()
    employment_a64_df['ID_str'] = employment_a64_df['ID_code'].astype(str)
    # Remove trailing .0 (e.g., 55.0 -> 55)
    employment_a64_df['ID_clean'] = employment_a64_df['ID_str'].str.replace(r'\.0+$', '', regex=True)
    
    # Try three levels of matching:
    # 1. Exact match on full code (e.g., 3.1 matches concordance prefix 3.1X)
    # 2. Match on main code (e.g., 55 matches 55.X)
    
    # First, try to match on prefix (handles 3.1 -> 3.11, 49.1 -> 49.10)
    nace_prefix_isic = nace_isic[['A64', 'NACE_prefix', 'ISIC_division_code']].drop_duplicates()
    merged = employment_a64_df.merge(
        nace_prefix_isic,
        left_on=['A64', 'ID_clean'],
        right_on=['A64', 'NACE_prefix'],
        how='left'
    )
    
    # For unmatched, try main code match (handles 55.0 -> 55.X)
    unmatched_mask = merged['ISIC_division_code'].isna()
    if unmatched_mask.any():
        employment_a64_df['ID_main'] = employment_a64_df['ID_clean'].str.split('.').str[0]
        nace_main_isic = nace_isic[['A64', 'NACE_main', 'ISIC_division_code']].drop_duplicates()
        
        for idx in merged[unmatched_mask].index:
            a64 = merged.loc[idx, 'A64']
            id_main = employment_a64_df.loc[employment_a64_df.index[idx], 'ID_main']
            matches = nace_main_isic[(nace_main_isic['A64'] == a64) & (nace_main_isic['NACE_main'] == id_main)]
            if not matches.empty:
                # Use the first match
                merged.loc[idx, 'ISIC_division_code'] = matches.iloc[0]['ISIC_division_code']
    
    # Check for unmatched records
    unmatched = merged[merged['ISIC_division_code'].isna()]
    if not unmatched.empty:
        print(f"\nWarning: {len(unmatched)} records could not be mapped to ISIC divisions")
        print(unmatched[['A64', 'ID_code', 'ID_clean', 'Employed']].head(10))
    
    # Sum employment per (A64, ISIC_division_code)
    emp_by_isic = merged[merged['ISIC_division_code'].notna()].groupby(['A64', 'ISIC_division_code'], as_index=False)['Employed'].sum()
    
    # Calculate total employment per A64
    total_emp_a64 = employment_a64_df.groupby('A64', as_index=False)['Employed'].sum()
    total_emp_a64 = total_emp_a64.rename(columns={'Employed':'Employed_total_A64'})
    
    # Save outputs
    employment_a64_df.to_csv(
        Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\Employment_by_A64.csv"),
        index=False
    )
    
    emp_by_isic.to_csv(
        Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\Employment_distribution_by_ISIC_division.csv"),
        index=False
    )
    
    print(f"Mapped {len(emp_by_isic)} (A64, ISIC) combinations")
    return emp_by_isic, total_emp_a64

def compute(out_path, concordance, employment_by_isic, total_emp_a64, ratings_df, rating_cols, units_df, unit_cols):
    df = employment_by_isic.merge(ratings_df, on='ISIC_division_code', how='left')
    if units_df is not None:
        df = df.merge(units_df, on='ISIC_division_code', how='left')
    
    results = []
    for a64, g in df.groupby('A64'):
        row = {'A64': a64}
        
        # Get actual total employment for this A64 (from original data, not duplicated by ISIC mapping)
        tot = total_emp_a64.loc[total_emp_a64['A64'] == a64, 'Employed_total_A64']
        emp_total = float(tot.iloc[0]) if not tot.empty else g['Employed'].sum()
        row['Employed_total_A64'] = emp_total
        
        for rc in rating_cols:
            # Exclude NaN and zero ratings (0 = "no value")
            sub = g[g[rc].notna()].copy()
            if not sub.empty:
                sub[rc] = pd.to_numeric(sub[rc], errors='coerce')
                sub = sub[sub[rc] != 0]  # Exclude zero ratings
            
            # Weighted average using employment as weights
            numer = (sub[rc].astype(float) * sub['Employed']).sum() if not sub.empty else 0.0
            denom = sub['Employed'].sum() if not sub.empty else 0.0
            weighted = (numer / denom) if denom > 0 else np.nan
            
            row[f'weighted_{rc}'] = weighted
            row[f'coverage_{rc}'] = (denom / emp_total) if emp_total > 0 else 0.0
            
            # Units aggregation
            unit_col_candidates = [c for c in unit_cols if c and c.split('_unit')[0].strip().lower() in rc.lower()]
            total_units = 0.0
            if units_df is not None and unit_col_candidates and not sub.empty:
                uc = unit_col_candidates[0]
                total_units = sub[uc].fillna(0).astype(float).sum()
            
            row[f'total_units_{rc}'] = total_units
            row[f'units_per_weighted_rating_{rc}'] = (total_units / weighted) if (pd.notna(weighted) and weighted != 0) else np.nan
        
        results.append(row)
    
    out = pd.DataFrame(results)
    
    # === Allocate service totals only to A64 with non-zero weighted rating ===
    if SERVICE_TOTALS:
        emp_series = out.set_index('A64')['Employed_total_A64']
        
        for rc, total_units in SERVICE_TOTALS.items():
            col_alloc = f'allocated_total_units_{rc}'
            col_weighted = f'weighted_{rc}'
            
            # Mark eligible A64 (non-zero weighted rating)
            out['_eligible'] = out[col_weighted].notna() & (out[col_weighted] != 0)
            
            # Get eligible employment total
            eligible_emp_total = out.loc[out['_eligible'], 'A64'].map(emp_series).sum()
            
            if eligible_emp_total > 0:
                # Allocate proportionally to eligible A64 only
                out[col_alloc] = out.apply(
                    lambda row: (emp_series.get(row['A64'], 0.0) / eligible_emp_total) * float(total_units)
                    if row['_eligible']
                    else 0.0,
                    axis=1
                )
            else:
                out[col_alloc] = 0.0
            
            # Clean up temp column
            out.drop('_eligible', axis=1, inplace=True)
    # === END allocation ===
    
    # Ensure allocated columns are numeric
    for c in out.columns:
        if c.startswith('allocated_total_units_'):
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
            out[c] = out[c].round(6)
    
    out.to_csv(out_path, index=False)
    return out

def main():
    # Load A64-NACE-ISIC concordance (for ISIC division mapping)
    concordance = load_concordance(CONCORD)
    
    # Load A64-NACE pattern mapping
    a64_nace_patterns = load_a64_nace_patterns(A64_NACE_MAPPING)
    
    # Load and aggregate employment data (smart aggregation based on NACE codes)
    employment_data = load_employment_data(EMPLOYMENT_BY_DETAILED_CODE, A64_NACE_MAPPING)
    
    # Map aggregated employment to A64 via NACE patterns
    employment_a64 = map_employment_to_a64(employment_data, a64_nace_patterns)
    
    # Map A64 employment to ISIC divisions
    emp_by_isic, total_emp_a64 = aggregate_employment_to_isic(employment_a64, concordance)
    
    ratings_df, rating_cols = load_ratings(ISIC_RATINGS)
    units_df, unit_cols = load_units(ISIC_UNITS, pd.read_csv(ISIC_RATINGS, dtype=str).fillna(''), rating_cols)
    
    res = compute(OUT, concordance, emp_by_isic, total_emp_a64, ratings_df, rating_cols, units_df, unit_cols)
    print("Saved:", OUT)
    print("Columns written:", res.columns.tolist())

if __name__ == '__main__':
    main()
