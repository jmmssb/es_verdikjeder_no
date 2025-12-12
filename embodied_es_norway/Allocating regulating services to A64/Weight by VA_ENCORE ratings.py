from pathlib import Path
import pandas as pd
import numpy as np

# INPUTS - adjust paths
CONCORD = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\A64_NACE_ISIC_concordance_full.csv")
VA_BY_A64 = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\a64_value_added.csv")  # columns: A64, VA
ISIC_RATINGS = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\ecosystem_services_weighted_by_isic_division.csv")
ISIC_UNITS = None  # optional
OUT = Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\A64_service_units_per_weighted_rating.csv")

SERVICE_TOTALS = globals().get('SERVICE_TOTALS', {
        'Air Filtration_value': 63979.28,
        'Flood mitigation services_value': 165855.72,
        'Local (micro and meso) climate regulation_value': 1.6565,
    })

def find_isic_col(df):
    for c in df.columns:
        if c.strip().lower() in ('isic division','isic_division','isic_division_code','isic division code'):
            return c
    return c

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

def load_va_a64(path):
    df = pd.read_csv(path, dtype=str).fillna('')
    cols = {c.lower(): c for c in df.columns}
    a64_col = cols.get('a64') or next((c for c in df.columns if 'a64' in c.lower()), None)
    va_col = next((c for c in df.columns if any(k in c.lower() for k in ('value','va','gva'))), None)
    if not (a64_col and va_col):
        raise SystemExit(f"A64 VA file missing columns. Found: {df.columns.tolist()}")
    df = df[[a64_col, va_col]].rename(columns={a64_col:'A64', va_col:'VA'})
    df['A64'] = df['A64'].astype(str).str.strip()
    df['VA'] = pd.to_numeric(df['VA'], errors='coerce').fillna(0.0)
    return df

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

def distribute_a64_va_to_nace(concord, va_a64):
    """
    Distribute A64 VA equally to each NACE code within the A64.
    Result: DataFrame with columns A64, NACE_code, ISIC_division_code, VA (per NACE)
    """
    merged = concord.merge(va_a64, on='A64', how='left')  # A64, NACE_code, ISIC_division_code, VA
    # count distinct NACE per A64
    counts = merged.groupby('A64')['NACE_code'].nunique().rename('n_nace').reset_index()
    merged = merged.merge(counts, on='A64', how='left')
    # avoid division by zero
    merged['n_nace'] = merged['n_nace'].replace(0, np.nan)
    merged['VA_per_NACE'] = merged['VA'] / merged['n_nace']
    merged['VA_per_NACE'] = merged['VA_per_NACE'].fillna(0.0)
    
    # Save detailed VA per NACE mapping
    nace_detail = merged[['A64', 'NACE_code', 'ISIC_division_code', 'VA_per_NACE', 'n_nace']].copy()
    nace_detail.to_csv(
        Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\VA_distribution_by_NACE.csv"),
        index=False
    )
    
    # aggregate to ISIC division: sum VA_per_NACE for all NACE codes mapping to that ISIC division
    va_by_isic = merged.groupby(['A64','ISIC_division_code'], as_index=False)['VA_per_NACE'].sum().rename(columns={'VA_per_NACE':'VA'})
    
    # Save VA aggregated to ISIC division
    va_by_isic.to_csv(
        Path(r"C:\Users\dth\OneDrive - Statistisk sentralbyrå\VSCode\ENCORE\VA_distribution_by_ISIC_division.csv"),
        index=False
    )
    
    total_va_a64 = va_a64.rename(columns={'VA':'VA_total_A64'})
    return va_by_isic, total_va_a64

def compute(out_path, concord, va_a64, ratings_df, rating_cols, units_df, unit_cols):
    va_by_isic_df, total_va_a64 = distribute_a64_va_to_nace(concord, va_a64)
    df = va_by_isic_df.merge(ratings_df, on='ISIC_division_code', how='left')
    if units_df is not None:
        df = df.merge(units_df, on='ISIC_division_code', how='left')
    results = []
    for a64, g in df.groupby('A64'):
        row = {'A64': a64}
        va_total = g['VA'].sum()
        row['VA_total_mapped_to_A64'] = va_total
        for rc in rating_cols:       
            sub = g[g[rc].notna()].copy()  # Keep zeros, exclude only NaN
            if not sub.empty:
                sub[rc] = pd.to_numeric(sub[rc], errors='coerce')
            # Weighted average using VA as weights (only divisions with non-zero rating)
            numer = (sub[rc].astype(float) * sub['VA']).sum() if not sub.empty else 0.0
            denom = sub['VA'].sum() if not sub.empty else 0.0
            weighted = (numer / denom) if denom > 0 else np.nan
            row[f'weighted_{rc}'] = weighted
            row[f'coverage_{rc}'] = (denom / va_total) if va_total > 0 else 0.0
            # Units aggregation (only for divisions that contributed to weighted calculation)
            unit_col_candidates = [c for c in unit_cols if c and c.split('_unit')[0].strip().lower() in rc.lower()]
            total_units = 0.0
            if units_df is not None and unit_col_candidates and not sub.empty:
                uc = unit_col_candidates[0]
                total_units = sub[uc].fillna(0).astype(float).sum()
            row[f'total_units_{rc}'] = total_units
            row[f'units_per_weighted_rating_{rc}'] = (total_units / weighted) if (pd.notna(weighted) and weighted != 0) else np.nan
        tot = total_va_a64.loc[total_va_a64['A64'] == a64, 'VA_total_A64']
        row['VA_total_A64_full'] = float(tot.iloc[0]) if not tot.empty else va_total
        results.append(row)
    out = pd.DataFrame(results)

    # === Allocate service totals only to A64 with non-zero weighted rating ===
    if SERVICE_TOTALS:
        if 'VA_total_A64_full' in out.columns:
            va_series = out.set_index('A64')['VA_total_A64_full']
        else:
            va_series = out.set_index('A64')['VA_total_mapped_to_A64']

        for rc, total_units in SERVICE_TOTALS.items():
            col_alloc = f'allocated_total_units_{rc}'
            col_weighted = f'weighted_{rc}'
            
            # Mark eligible A64 (non-zero weighted rating)
            out['_eligible'] = out[col_weighted].notna() & (out[col_weighted] != 0)
            
            # Get eligible VA total
            eligible_va_total = out.loc[out['_eligible'], 'A64'].map(va_series).sum()
            
            if eligible_va_total > 0:
                # Allocate proportionally to eligible A64 only
                out[col_alloc] = out.apply(
                    lambda row: (va_series.get(row['A64'], 0.0) / eligible_va_total) * float(total_units)
                    if row['_eligible']
                    else 0.0,
                    axis=1
                )
            else:
                out[col_alloc] = 0.0
            
            # Clean up temp column
            out.drop('_eligible', axis=1, inplace=True)
    # === END allocation ===

    out.to_csv(out_path, index=False)
    return out

def main():
    concord = load_concordance(CONCORD)
    va_a64 = load_va_a64(VA_BY_A64)
    ratings_df, rating_cols = load_ratings(ISIC_RATINGS)
    units_df, unit_cols = load_units(ISIC_UNITS, pd.read_csv(ISIC_RATINGS, dtype=str).fillna(''), rating_cols)
    res = compute(OUT, concord, va_a64, ratings_df, rating_cols, units_df, unit_cols)
    print("Saved:", OUT)
    print("Columns written:", res.columns.tolist())

if __name__ == '__main__':
    main()

