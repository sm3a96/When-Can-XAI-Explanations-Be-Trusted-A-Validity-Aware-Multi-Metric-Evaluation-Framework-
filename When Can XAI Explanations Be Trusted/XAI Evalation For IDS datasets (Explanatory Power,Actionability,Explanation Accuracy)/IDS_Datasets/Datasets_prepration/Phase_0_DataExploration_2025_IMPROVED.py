"""
Phase 0.1 IMPROVED: Dataset Exploration & Characterization for 2025 Threat Datasets

Updated to handle:
- Excel files (.xlsx) for IDS2025
- Directory filtering for CIC IIoT
- Label discovery for Windows-APT

Author: AI Agent Phase 0.1 (IMPROVED)
Date: 2026-05-12
Random Seed: 42
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_PATH = Path(__file__).parent.parent
DATASETS_PATH = BASE_PATH / "2025 IDS Datasets"
RESULTS_PATH = BASE_PATH / "Dataset_Analysis_Results"
RESULTS_PATH.mkdir(exist_ok=True)

# ============================================================================
# IMPROVED LOADING FUNCTIONS
# ============================================================================

def load_ciciot_dataset() -> pd.DataFrame:
    """Load CIC IIoT 2025 dataset - IMPROVED version."""
    logger.info("Loading CIC IIoT 2025 dataset...")

    try:
        ciciot_path = DATASETS_PATH / "CIC IIoT dataset 2025"

        # Find ACTUAL CSV files (not directories)
        csv_files = [f for f in ciciot_path.rglob("*.csv") if f.is_file()]
        logger.info(f"Found {len(csv_files)} actual CSV files in CIC IIoT dataset")

        if not csv_files:
            logger.warning("No CSV files found in CIC IIoT dataset")
            return pd.DataFrame()

        # Load first file to inspect
        sample_df = pd.read_csv(csv_files[0], nrows=10)
        logger.info(f"CIC IIoT sample shape: {sample_df.shape}")
        logger.info(f"CIC IIoT columns (first 20): {sample_df.columns.tolist()[:20]}")

        # Load all files and combine
        dfs = []
        for csv_file in csv_files[:3]:  # Load first 3 files for analysis
            try:
                data = pd.read_csv(csv_file)
                dfs.append(data)
                logger.info(f"Loaded {csv_file.name}: {data.shape}")
            except Exception as e:
                logger.warning(f"Error loading {csv_file.name}: {e}")

        if dfs:
            df_ciciot = pd.concat(dfs, ignore_index=True)
            logger.info(f"CIC IIoT combined shape: {df_ciciot.shape}")

            # Try to identify label column
            label_col = None
            for col in df_ciciot.columns:
                if 'label' in col.lower() or 'attack' in col.lower() or 'class' in col.lower():
                    logger.info(f"Found potential label column: {col}")
                    label_col = col
                    break

            if label_col:
                logger.info(f"Unique classes in '{label_col}': {df_ciciot[label_col].unique()[:10]}")

            return df_ciciot
        else:
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error loading CIC IIoT dataset: {e}")
        return pd.DataFrame()


def load_windows_apt_dataset() -> pd.DataFrame:
    """Load Windows-APT 2025 dataset - IMPROVED version."""
    logger.info("Loading Windows-APT 2025 dataset...")

    try:
        apt_path = DATASETS_PATH / "Windows-APT 2025 A Dataset for APT-Inspired Attack"

        # Try combined file first
        combined_files = list(apt_path.rglob("combined.csv"))

        if combined_files:
            df = pd.read_csv(combined_files[0])
            logger.info(f"Windows-APT combined file shape: {df.shape}")

            # Investigate columns for label
            logger.info(f"Columns (first 30): {df.columns.tolist()[:30]}")
            logger.info(f"Last column: '{df.columns[-1]}'")

            # Check if last column might be label
            if df.columns[-1].lower() not in ['index', 'unnamed']:
                last_col_unique = df.iloc[:, -1].nunique()
                logger.info(f"Last column unique values: {last_col_unique}")
                if last_col_unique < 100:  # Likely a label column
                    logger.info(f"Last column '{df.columns[-1]}' has {last_col_unique} unique values (potential label)")

            # Check scenario_manifest for label mapping
            scenario_files = list(apt_path.rglob("scenario_manifest.csv"))
            if scenario_files:
                scenario_df = pd.read_csv(scenario_files[0])
                logger.info(f"Scenario manifest shape: {scenario_df.shape}")
                logger.info(f"Scenario columns: {scenario_df.columns.tolist()}")

            return df
        else:
            logger.warning("No combined.csv found in Windows-APT")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error loading Windows-APT dataset: {e}")
        return pd.DataFrame()


def load_ids2025_dataset() -> pd.DataFrame:
    """Load IDS2025 Balanced dataset - IMPROVED version with Excel support."""
    logger.info("Loading IDS2025 Balanced dataset...")

    try:
        ids2025_path = DATASETS_PATH / "IDS2025 (Balanced Intrusion Detection Evaluation D"

        # Look for Excel files
        xlsx_files = list(ids2025_path.rglob("*.xlsx"))
        logger.info(f"Found {len(xlsx_files)} Excel files in IDS2025 dataset")

        if xlsx_files:
            # Load Excel file
            df = pd.read_excel(xlsx_files[0])
            logger.info(f"IDS2025 shape: {df.shape}")
            logger.info(f"IDS2025 columns: {df.columns.tolist()}")

            # Look for label column
            label_col = None
            for col in df.columns:
                if 'label' in col.lower() or 'attack' in col.lower() or 'class' in col.lower():
                    logger.info(f"Found label column: {col}")
                    label_col = col
                    break

            if label_col:
                logger.info(f"Unique classes: {df[label_col].unique()}")
                logger.info(f"Class distribution:\n{df[label_col].value_counts()}")

            return df
        else:
            # Try CSV as fallback
            csv_files = list(ids2025_path.rglob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                logger.info(f"IDS2025 (CSV) shape: {df.shape}")
                return df
            else:
                logger.warning("No files found in IDS2025 dataset")
                return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error loading IDS2025 dataset: {e}")
        return pd.DataFrame()


# ============================================================================
# ANALYSIS FUNCTION
# ============================================================================

def analyze_dataset(df: pd.DataFrame, dataset_name: str) -> Dict:
    """Comprehensive dataset analysis."""
    logger.info(f"\n{'='*80}")
    logger.info(f"ANALYZING: {dataset_name}")
    logger.info(f"{'='*80}")

    if df.empty:
        logger.warning(f"{dataset_name} is empty")
        return {'Dataset Name': dataset_name, 'Status': 'EMPTY'}

    analysis = {
        'Dataset Name': dataset_name,
        'Total Samples': len(df),
        'Total Features': len(df.columns),
        'Missing Values Count': df.isnull().sum().sum(),
        'Missing Values %': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'Duplicates': len(df) - len(df.drop_duplicates()),
    }

    # Feature analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    analysis['Numeric Features'] = len(numeric_cols)
    analysis['Categorical Features'] = len(categorical_cols)

    logger.info(f"Samples: {analysis['Total Samples']}")
    logger.info(f"Features: {analysis['Total Features']}")
    logger.info(f"Numeric: {analysis['Numeric Features']}, Categorical: {analysis['Categorical Features']}")
    logger.info(f"Missing: {analysis['Missing Values Count']} ({analysis['Missing Values %']:.2f}%)")

    # Find label column
    label_candidates = ['label', 'Label', 'class', 'Class', 'attack', 'Attack', 'attack_type', 'target']
    label_col = None

    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break

    if label_col is None and len(categorical_cols) > 0:
        # Check last column
        last_col = df.columns[-1]
        if df[last_col].nunique() < len(df) * 0.5:
            label_col = last_col

    if label_col:
        analysis['Label Column'] = label_col
        analysis['Unique Classes'] = df[label_col].nunique()
        analysis['Class Distribution'] = df[label_col].value_counts().to_dict()
        logger.info(f"Label column: '{label_col}'")
        logger.info(f"Unique classes: {analysis['Unique Classes']}")
        logger.info(f"Class distribution (top 5):")
        for cls, count in df[label_col].value_counts().head(5).items():
            logger.info(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")
    else:
        logger.warning(f"No label column identified in {dataset_name}")
        analysis['Label Column'] = 'NOT FOUND'

    return analysis


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    logger.info("\n" + "="*80)
    logger.info("PHASE 0.1 IMPROVED: DATASET EXPLORATION FOR 2025 THREAT DATASETS")
    logger.info("="*80)
    logger.info(f"Random Seed: {RANDOM_SEED}\n")

    # Load datasets
    logger.info("STEP 1: LOADING ALL DATASETS")
    logger.info("-" * 80)

    ciciot_df = load_ciciot_dataset()
    windows_apt_df = load_windows_apt_dataset()
    ids2025_df = load_ids2025_dataset()

    # Analyze datasets
    logger.info("\n\nSTEP 2: COMPREHENSIVE ANALYSIS")
    logger.info("-" * 80)

    analyses = {}

    if not ciciot_df.empty:
        analyses['CIC IIoT 2025'] = analyze_dataset(ciciot_df, 'CIC IIoT 2025')

    if not windows_apt_df.empty:
        analyses['Windows-APT 2025'] = analyze_dataset(windows_apt_df, 'Windows-APT 2025')

    if not ids2025_df.empty:
        analyses['IDS2025 Balanced'] = analyze_dataset(ids2025_df, 'IDS2025 Balanced')

    # Save results
    logger.info("\n\nSTEP 3: SAVING RESULTS")
    logger.info("-" * 80)

    summary_data = []
    for dataset_name, analysis in analyses.items():
        summary_data.append({
            'Dataset': dataset_name,
            'Samples': analysis.get('Total Samples', 0),
            'Features': analysis.get('Total Features', 0),
            'Numeric': analysis.get('Numeric Features', 0),
            'Categorical': analysis.get('Categorical Features', 0),
            'Classes': analysis.get('Unique Classes', 'N/A'),
            'Label Column': analysis.get('Label Column', 'N/A'),
            'Missing %': f"{analysis.get('Missing Values %', 0):.2f}%",
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = RESULTS_PATH / "summary_statistics_2025.csv"
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"\n✓ Summary Statistics saved to: {summary_path}")
    logger.info(f"\n{summary_df.to_string()}")

    # Save detailed analysis
    import json
    detailed_path = RESULTS_PATH / "detailed_analysis_2025.json"
    with open(detailed_path, 'w') as f:
        json.dump(analyses, f, indent=2, default=str)
    logger.info(f"✓ Detailed analysis saved to: {detailed_path}")

    logger.info("\n" + "="*80)
    logger.info("PHASE 0.1 EXPLORATION COMPLETE")
    logger.info("="*80)
    logger.info("\nNext Steps:")
    logger.info(f"1. Review summary statistics")
    logger.info(f"2. Proceed to Phase 0.2: Methodology Design")

    return analyses


if __name__ == "__main__":
    analyses = main()
