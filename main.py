"""
Energy Transition and Air Pollution Mortality - Data Preparation Script

This script cleans and prepares two datasets for statistical analysis:
1. OWID Energy Data: energy source shares by country/year
2. WHO Mortality Data: air pollution attributable deaths by country/year

Output: Two cleaned CSV files ready for R analysis

Usage:
    python main.py --energy-data-path data/owid-energy-data.csv \
                   --deaths-data-path data/air_attributable_deaths.csv \
                   --output-dir data \
                   --verbose

Data Sources:
    - Energy: https://github.com/owid/energy-data (owid-energy-data.csv)
    - Mortality: WHO Global Health Observatory
      https://www.who.int/data/gho/data/indicators/indicator-details/GHO/ambient-air-pollution-attributable-deaths

WHO Mortality Data Format:
    - Location: country name
    - SpatialDimValueCode: 3-letter ISO code  
    - Period: year
    - Dim3ValueCode: deaths as "avg [min-max]" string
    - Dim2ValueCode: cause of death code
    - Dim1ValueCode: sex code (SEX_MLE, SEX_FMLE, SEX_BTSX)
    
    The script extracts the average from "avg [min-max]" and uses only SEX_BTSX
    (both sexes) to avoid double counting.
"""

import pandas as pd
import argparse
import os
import sys



def clean_energy_data(data, year_period=[2010, 2020]):
    """
    Clean the OWID energy dataset.
    
    Steps:
    1. Remove rows without ISO codes (aggregates like "World", "Europe", etc.)
    2. Filter to specified year range
    3. Select relevant columns (both shares and absolute consumption values)
    4. Compute missing shares from absolute consumption values
    5. Handle missing values
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw OWID energy data
    year_period : list
        [start_year, end_year] inclusive range
        
    Returns:
    --------
    pd.DataFrame
        Cleaned energy data with computed share columns
    """
    print(f"  Raw energy data: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Step 1: Remove rows without ISO codes (these are aggregates)
    # Known countries have 3-letter ISO codes; aggregates have longer codes or NaN
    data_clean = data[data["iso_code"].notna()].copy()
    data_clean = data_clean[data_clean["iso_code"].str.len() == 3]
    print(f"  After removing aggregates: {data_clean.shape[0]} rows")
    
    # Step 2: Filter to year range
    data_clean = data_clean[
        (data_clean["year"] >= year_period[0]) & 
        (data_clean["year"] <= year_period[1])
    ]
    print(f"  After year filter ({year_period[0]}-{year_period[1]}): {data_clean.shape[0]} rows")
    
    # Step 3: Select useful columns (both shares and consumption values)
    # We need consumption values to compute missing shares
    useful_columns = [
        # Identifiers
        "country",
        "year", 
        "iso_code",
        # Share columns (pre-calculated)
        "fossil_share_energy",
        "coal_share_energy",
        "gas_share_energy",
        "oil_share_energy",
        "low_carbon_share_energy",
        "nuclear_share_energy",
        "hydro_share_energy",
        "solar_share_energy",
        "wind_share_energy",
        "biofuel_share_energy",
        "other_renewables_share_energy",
        # Consumption columns (for computing missing shares)
        "primary_energy_consumption",
        "fossil_fuel_consumption",
        "coal_consumption",
        "gas_consumption",
        "oil_consumption",
        "low_carbon_consumption",
        "nuclear_consumption",
        "hydro_consumption",
        "solar_consumption",
        "wind_consumption",
        "biofuel_consumption",
        "other_renewable_consumption",
        # Optional: GDP for confounding analysis
        "gdp",
        "population",
    ]
    
    # Keep only columns that exist in the data
    available_columns = [col for col in useful_columns if col in data_clean.columns]
    data_clean = data_clean[available_columns].copy()
    
    # Step 4: Compute missing shares from absolute consumption values
    # This is the key improvement for nations lacking pre-calculated shares
    print("  Computing missing energy shares from consumption values...")
    
    # Check if we have primary energy consumption (needed for all calculations)
    if "primary_energy_consumption" in data_clean.columns:
        # Define mapping: share_column -> consumption_column
        share_mappings = {
            "coal_share_energy": "coal_consumption",
            "gas_share_energy": "gas_consumption", 
            "oil_share_energy": "oil_consumption",
            "nuclear_share_energy": "nuclear_consumption",
            "hydro_share_energy": "hydro_consumption",
            "solar_share_energy": "solar_consumption",
            "wind_share_energy": "wind_consumption",
            "biofuel_share_energy": "biofuel_consumption",
            "other_renewables_share_energy": "other_renewable_consumption",
        }
        
        computed_count = 0
        for share_col, consumption_col in share_mappings.items():
            if consumption_col in data_clean.columns:
                # Create share column if it doesn't exist
                if share_col not in data_clean.columns:
                    data_clean[share_col] = None
                
                # Compute share where it's missing but consumption data exists
                # Formula: (source_consumption / total_consumption) * 100
                mask = (
                    data_clean[share_col].isna() & 
                    data_clean[consumption_col].notna() & 
                    data_clean["primary_energy_consumption"].notna() &
                    (data_clean["primary_energy_consumption"] > 0)
                )
                
                if mask.sum() > 0:
                    data_clean.loc[mask, share_col] = (
                        data_clean.loc[mask, consumption_col] / 
                        data_clean.loc[mask, "primary_energy_consumption"]
                    ) * 100
                    computed_count += mask.sum()
        
        # Compute composite shares (fossil and low-carbon)
        # Fossil share = coal + gas + oil
        if all(col in data_clean.columns for col in ["coal_share_energy", "gas_share_energy", "oil_share_energy"]):
            if "fossil_share_energy" not in data_clean.columns:
                data_clean["fossil_share_energy"] = None
            
            mask = data_clean["fossil_share_energy"].isna()
            if mask.sum() > 0:
                data_clean.loc[mask, "fossil_share_energy"] = (
                    data_clean.loc[mask, "coal_share_energy"].fillna(0) +
                    data_clean.loc[mask, "gas_share_energy"].fillna(0) +
                    data_clean.loc[mask, "oil_share_energy"].fillna(0)
                )
        
        # Low-carbon share = nuclear + renewables
        renewable_cols = ["hydro_share_energy", "solar_share_energy", "wind_share_energy", 
                         "biofuel_share_energy", "other_renewables_share_energy"]
        available_renewable_cols = [col for col in renewable_cols if col in data_clean.columns]
        
        if "nuclear_share_energy" in data_clean.columns and available_renewable_cols:
            if "low_carbon_share_energy" not in data_clean.columns:
                data_clean["low_carbon_share_energy"] = None
            
            mask = data_clean["low_carbon_share_energy"].isna()
            if mask.sum() > 0:
                renewable_sum = sum(data_clean.loc[mask, col].fillna(0) for col in available_renewable_cols)
                data_clean.loc[mask, "low_carbon_share_energy"] = (
                    data_clean.loc[mask, "nuclear_share_energy"].fillna(0) + renewable_sum
                )
        
        print(f"  Computed {computed_count} missing share values from consumption data")
    else:
        print("  Warning: primary_energy_consumption not available - cannot compute missing shares")
    
    # Step 5: Create GDP per capita if GDP and population are available
    if "gdp" in data_clean.columns and "population" in data_clean.columns:
        data_clean["gdp_per_capita"] = data_clean["gdp"] / data_clean["population"]
        print("  Created gdp_per_capita variable")
    
    # Step 6: Select final columns (keep only share columns and metadata)
    final_columns = [
        "country", "year", "iso_code",
        "fossil_share_energy", "coal_share_energy", "gas_share_energy", "oil_share_energy",
        "low_carbon_share_energy", "nuclear_share_energy", 
        "hydro_share_energy", "solar_share_energy", "wind_share_energy",
        "biofuel_share_energy", "other_renewables_share_energy",
        "gdp", "population", "gdp_per_capita"
    ]
    final_columns = [col for col in final_columns if col in data_clean.columns]
    data_clean = data_clean[final_columns].copy()
    
    # Step 7: Drop rows where ALL energy share variables are still missing
    energy_cols = [col for col in data_clean.columns if "share_energy" in col]
    initial_rows = len(data_clean)
    data_clean = data_clean.dropna(subset=energy_cols, how="all")
    dropped = initial_rows - len(data_clean)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with all-missing energy data")
    
    print(f"  Final energy data: {data_clean.shape[0]} rows, {data_clean.shape[1]} columns")
    
    return data_clean

def clean_deaths_data(data, year_period=[2010, 2020]):
    """
    Clean the WHO air pollution mortality dataset.
    
    WHO GHO format specifics:
    - Location: country name
    - SpatialDimValueCode: 3-letter ISO code
    - Period: year
    - Dim3ValueCode: deaths as "avg [min-max]" string (we extract avg)
    - Dim2ValueCode: cause of death code
    - Dim1ValueCode: sex code (SEX_MLE, SEX_FMLE, SEX_BTSX)
    
    Important: Avoid double counting when all 3 sex codes are present.
    We prioritize SEX_BTSX (both sexes) when available.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw WHO mortality data
    year_period : list
        [start_year, end_year] inclusive range
        
    Returns:
    --------
    pd.DataFrame
        Cleaned mortality data with columns: iso_code, country, year, deaths
    """
    import re
    
    print(f"  Raw mortality data: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"  Available columns: {list(data.columns)}")
    
    data_clean = data.copy()
    
    rename_map = {
        "Location": "country",
        "SpatialDimValueCode": "iso_code", 
        "Period": "year",
        "Value": "deaths_raw",      # "avg [min-max]" format
        "Dim2ValueCode": "cause_code",       # Cause of death
        "Dim1ValueCode": "sex_code"          # SEX_MLE, SEX_FMLE, SEX_BTSX
    }
    
    # Check which columns exist
    missing_cols = [col for col in rename_map.keys() if col not in data_clean.columns]
    if missing_cols:
        print(f"  Warning: Missing expected columns: {missing_cols}")
    
    data_clean = data_clean.rename(columns=rename_map)
    print(f"  Renamed columns: {rename_map}")
    
    # =========================================================================
    # STEP 2: Parse deaths from "avg [min-max]" format
    # =========================================================================
    def extract_avg_deaths(value):
        """
        Extract average deaths from WHO format: "1234 [1000-1500]"
        Returns just the first number (the average/point estimate).
        """
        if pd.isna(value):
            return None
        
        value_str = str(value).strip()
        
        # Try to extract the first number (before any bracket or space)
        # Pattern: captures digits (possibly with spaces as thousand separators)
        match = re.match(r'^[\s]*([\d\s]+)', value_str)
        if match:
            # Remove spaces from number (thousand separators)
            num_str = match.group(1).replace(' ', '').replace('\xa0', '')
            try:
                return float(num_str)
            except ValueError:
                pass
        
        # Fallback: try direct conversion
        try:
            return float(value_str)
        except ValueError:
            return None
    
    if "deaths_raw" in data_clean.columns:
        print("  Parsing deaths from 'avg [min-max]' format...")
        data_clean["deaths"] = data_clean["deaths_raw"].apply(extract_avg_deaths)
        
        # Check parsing success
        n_parsed = data_clean["deaths"].notna().sum()
        n_total = len(data_clean)
        print(f"  Successfully parsed {n_parsed}/{n_total} death values ({100*n_parsed/n_total:.1f}%)")
        
        # Show example of parsing
        sample = data_clean[["deaths_raw", "deaths"]].dropna().head(3)
        print(f"  Sample parsing: {sample.to_dict('records')}")
    else:
        print("  ERROR: No deaths column found")
        return None
    
    # =========================================================================
    # STEP 3: Handle sex codes - avoid double counting
    # =========================================================================
    # Priority: SEX_BTSX (both sexes) > individual sex codes
    # If SEX_BTSX exists for a country/year/cause, use only that
    # Otherwise, we might need to sum SEX_MLE + SEX_FMLE (but this is risky)
    
    if "sex_code" in data_clean.columns:
        print(f"  Sex codes found: {data_clean['sex_code'].unique()}")
        
        # Strategy: Keep only SEX_BTSX where available
        # This avoids double counting entirely
        
        # Identify group keys
        group_cols = ["iso_code", "year"]
        if "cause_code" in data_clean.columns:
            group_cols.append("cause_code")
        
        # Filter to SEX_BTSX (both sexes) only
        btsx_data = data_clean[data_clean["sex_code"] == "SEX_BTSX"].copy()
        print(f"  Rows with SEX_BTSX: {len(btsx_data)}")
        
        # For groups without SEX_BTSX, we could sum male+female
        # But safer to just use SEX_BTSX to avoid issues
        data_clean = btsx_data
        print(f"  Using only SEX_BTSX to avoid double counting: {len(data_clean)} rows")
    
    # =========================================================================
    # STEP 4: Filter years
    # =========================================================================
    data_clean["year"] = pd.to_numeric(data_clean["year"], errors="coerce")
    data_clean = data_clean[
        (data_clean["year"] >= year_period[0]) & 
        (data_clean["year"] <= year_period[1])
    ]
    print(f"  After year filter ({year_period[0]}-{year_period[1]}): {len(data_clean)} rows")
    
    # =========================================================================
    # STEP 5: Aggregate deaths by country and year (sum across causes)
    # =========================================================================
    # If there are multiple cause codes, we might want to:
    # a) Keep them separate for detailed analysis, OR
    # b) Sum them for total air pollution mortality
    
    if "cause_code" in data_clean.columns:
        print(f"  Cause codes found: {data_clean['cause_code'].nunique()} unique codes")
        print(f"  Cause codes: {data_clean['cause_code'].unique()}")
        
        # Aggregate: sum deaths across all causes for each country-year
        # This gives total air pollution attributable deaths
        agg_data = data_clean.groupby(["iso_code", "country", "year"]).agg({
            "deaths": "sum"
        }).reset_index()
        
        print(f"  After aggregating across causes: {len(agg_data)} rows")
        data_clean = agg_data
    
    # =========================================================================
    # STEP 6: Remove invalid rows and finalize
    # =========================================================================
    data_clean = data_clean.dropna(subset=["deaths", "iso_code", "year"])
    
    
    # Keep only 3-letter ISO codes (filter out aggregates)
    data_clean = data_clean[data_clean["iso_code"].str.len() == 3]
    
    # Final columns
    final_cols = ["iso_code", "country", "year", "deaths"]
    final_cols = [c for c in final_cols if c in data_clean.columns]
    data_clean = data_clean[final_cols]
    
    # Remove duplicates
    data_clean = data_clean.drop_duplicates(subset=["iso_code", "year"], keep="first")
    
    print(f"  Final mortality data: {len(data_clean)} rows, {len(data_clean.columns)} columns")
    print(f"  Years: {sorted(data_clean['year'].unique())}")
    print(f"  Countries: {data_clean['iso_code'].nunique()}")
    
    return data_clean


def merge_datasets(energy_data, mortality_data):
    """
    Merge energy and mortality datasets.
    
    Merges on iso_code and year. If iso_code is not available in mortality data,
    attempts to merge on country name.
    
    Parameters:
    -----------
    energy_data : pd.DataFrame
        Cleaned energy data
    mortality_data : pd.DataFrame
        Cleaned mortality data
        
    Returns:
    --------
    pd.DataFrame
        Merged dataset
    """
    print("\nMerging datasets...")
    
    # Determine merge columns
    if "iso_code" in mortality_data.columns and "iso_code" in energy_data.columns:
        merge_cols = ["iso_code", "year"]
    elif "country" in mortality_data.columns and "country" in energy_data.columns:
        merge_cols = ["country", "year"]
        print("  Warning: Merging on country name (may have matching issues)")
    else:
        print("  ERROR: No common identifier columns found")
        return None
    
    # Perform merge
    merged = pd.merge(
        energy_data,
        mortality_data,
        on=merge_cols,
        how="inner",
        suffixes=("", "_mort")
    )
    
    print(f"  Merged data: {merged.shape[0]} rows, {merged.shape[1]} columns")
    print(f"  Years in merged data: {sorted(merged['year'].unique())}")
    print(f"  Countries in merged data: {merged['iso_code'].nunique() if 'iso_code' in merged.columns else merged['country'].nunique()}")
    
    return merged


def main():
    """Main function to orchestrate data cleaning."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Energy transition and public health data preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --verbose
    python main.py --year-start 2015 --year-end 2019
    python main.py --energy-data-path my_energy.csv --deaths-data-path my_deaths.csv
        """
    )
    
    parser.add_argument(
        "--energy-data-path", 
        type=str, 
        default="data/owid-energy-data.csv",
        help="Path to OWID energy data CSV"
    )
    parser.add_argument(
        "--deaths-data-path", 
        type=str, 
        default="data/air_attributable_deaths.csv",
        help="Path to WHO mortality data CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory for output files"
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2010,
        help="Start year for filtering (default: 2010)"
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2020,
        help="End year for filtering (default: 2020)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help="Also create a merged dataset"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        default=False,
        help="Print detailed progress"
    )

    args = parser.parse_args()
    year_period = [args.year_start, args.year_end]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ENERGY-MORTALITY DATA PREPARATION")
    print("=" * 60)
    
    # =========================================================================
    # LOAD AND CLEAN ENERGY DATA
    # =========================================================================
    print(f"\n[1/3] Loading energy data from: {args.energy_data_path}")
    
    if not os.path.exists(args.energy_data_path):
        print(f"  ERROR: File not found: {args.energy_data_path}")
        print("  Download from: https://github.com/owid/energy-data")
        sys.exit(1)
    
    try:
        energy_raw = pd.read_csv(args.energy_data_path)
        print("  Cleaning energy data...")
        energy_clean = clean_energy_data(energy_raw, year_period)
        
        # Save cleaned data
        energy_output = os.path.join(args.output_dir, "energy_clean.csv")
        energy_clean.to_csv(energy_output, index=False)
        print(f"  Saved to: {energy_output}")
        
    except Exception as e:
        print(f"  ERROR processing energy data: {e}")
        sys.exit(1)
    
    # =========================================================================
    # LOAD AND CLEAN MORTALITY DATA
    # =========================================================================
    print(f"\n[2/3] Loading mortality data from: {args.deaths_data_path}")
    
    if not os.path.exists(args.deaths_data_path):
        print(f"  ERROR: File not found: {args.deaths_data_path}")
        print("  Download from: https://www.who.int/data/gho/")
        sys.exit(1)
    
    try:
        deaths_raw = pd.read_csv(args.deaths_data_path)
        print("  Cleaning mortality data...")
        deaths_clean = clean_deaths_data(deaths_raw, year_period)
        
        if deaths_clean is None:
            print("  ERROR: Could not clean mortality data")
            sys.exit(1)
        
        # Save cleaned data
        deaths_output = os.path.join(args.output_dir, "mortality_clean.csv")
        deaths_clean.to_csv(deaths_output, index=False)
        print(f"  Saved to: {deaths_output}")
        
    except Exception as e:
        print(f"  ERROR processing mortality data: {e}")
        sys.exit(1)
    
    # =========================================================================
    # OPTIONAL: MERGE DATASETS
    # =========================================================================
    if args.merge:
        print("\n[3/3] Merging datasets...")
        merged = merge_datasets(energy_clean, deaths_clean)
        
        if merged is not None:
            merged_output = os.path.join(args.output_dir, "merged_data.csv")
            merged.to_csv(merged_output, index=False)
            print(f"  Saved to: {merged_output}")
    else:
        print("\n[3/3] Skipping merge (use --merge flag to create merged dataset)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Energy data:   {energy_clean.shape[0]} rows, {energy_clean.shape[1]} columns")
    print(f"Mortality data: {deaths_clean.shape[0]} rows, {deaths_clean.shape[1]} columns")
    print(f"Output directory: {args.output_dir}/")
    print("\nFiles created:")
    print(f"  - energy_clean.csv")
    print(f"  - mortality_clean.csv")
    if args.merge:
        print(f"  - merged_data.csv")
    print("\nReady for R analysis!")


if __name__ == "__main__":
    main()