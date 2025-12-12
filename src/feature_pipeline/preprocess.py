"""
⚡ Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.

"""

"""
Preprocessing: city normalization + (optional) lat/lng merge, duplicate drop, outlier removal.

- Production defaults read from data/raw/ and write to data/processed/
- Tests can override `raw_dir`, `processed_dir`, and pass `metros_path=None`
  to skip merge safely without touching disk assets.
"""

import re
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

canonical_coords = {
    "Atlanta-Sandy Springs-Roswell": (33.7490, -84.3880),
    "Pittsburgh": (40.4406, -79.9959),
    "Boston-Cambridge-Newton": (42.3601, -71.0589),
    "Tampa-St. Petersburg-Clearwater": (27.9506, -82.4572),
    "Baltimore-Columbia-Towson": (39.2904, -76.6122),
    "Portland-Vancouver-Hillsboro": (45.5152, -122.6784),
    "Philadelphia-Camden-Wilmington": (39.9526, -75.1652),
    "New York-Newark-Jersey City": (40.7128, -74.0060),
    "Chicago-Naperville-Elgin": (41.8781, -87.6298),
    "Orlando-Kissimmee-Sanford": (28.5383, -81.3792),
    "Seattle-Tacoma-Bellevue": (47.6062, -122.3321),
    "San Francisco-Oakland-Fremont": (37.7749, -122.4194),
    "San Diego-Chula Vista-Carlsbad": (32.7157, -117.1611),
    "Austin-Round Rock-San Marcos": (30.2672, -97.7431),
    "St. Louis": (38.6270, -90.1994),
    "Sacramento-Roseville-Folsom": (38.5816, -121.4944),
    "Phoenix-Mesa-Chandler": (33.4484, -112.0740),
    "Riverside-San Bernardino-Ontario": (34.0522, -117.2898),
    "San Antonio-New Braunfels": (29.4241, -98.4936),
    "Detroit-Warren-Dearborn": (42.3314, -83.0458),
    "Cincinnati": (39.1031, -84.5120),
    "Houston-Pasadena-The Woodlands": (29.7604, -95.3698),
    "Charlotte-Concord-Gastonia": (35.2271, -80.8431),
    "Denver-Aurora-Centennial": (39.7392, -104.9903),
    "Los Angeles-Long Beach-Anaheim": (34.0522, -118.2437),
    "Washington-Arlington-Alexandria": (38.9072, -77.0369),
    "Dallas-Fort Worth-Arlington": (32.7767, -96.7970),
    "Minneapolis-St. Paul-Bloomington": (44.9778, -93.2650),
    "Las Vegas-Henderson-North Las Vegas": (36.1699, -115.1398),
    "Miami-Fort Lauderdale-West Palm Beach": (25.7617, -80.1918)
}

metros = pd.DataFrame([
    {"metro_full": name, "lat": coords[0], "lng": coords[1]}
    for name, coords in canonical_coords.items()
])


# Manual fixes for known mismatches (normalized form)
CITY_MAPPING = {
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
}


def normalize_city(s: str) -> str:
    """Lowercase, strip, unify dashes. Safe for NA."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = re.sub(r"[–—-]", "-", s)          # unify dashes
    s = re.sub(r"\s+", " ", s)            # collapse spaces
    return s


def clean_and_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize city names, optionally merge lat/lng from metros dataset.
    If `city_full` column or `metros_path` is missing, skip gracefully.
    """

    if "city_full" not in df.columns:
        print("⚠️ Skipping city merge: no 'city_full' column present.")
        return df

    # Normalize city_full
    df["city_full"] = df["city_full"].apply(normalize_city)
    # Apply mapping
    norm_mapping = {normalize_city(k): normalize_city(v) for k, v in CITY_MAPPING.items()}
    df["city_full"] = df["city_full"].replace(norm_mapping)

    # 🚨 If lat/lng already present, skip merge
    if {"lat", "lng"}.issubset(df.columns):
        print("⚠️ Skipping lat/lng merge: already present in DataFrame.")
        return df

    # Merge lat/lng
    global metros
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(metros.columns):
        print("⚠️ Skipping lat/lng merge: metros file missing required columns.")
        return df

    metros["metro_full"] = metros["metro_full"].apply(normalize_city)
    df = df.merge(metros[["metro_full", "lat", "lng"]],
                  how="left", left_on="city_full", right_on="metro_full")
    df.drop(columns=["metro_full"], inplace=True, errors="ignore")

    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print("⚠️ Still missing lat/lng for:", missing)
    else:
        print("✅ All cities matched with metros dataset.")
    return df



def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years."""
    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    after = df.shape[0]
    print(f"✅ Dropped {before - after} duplicate rows (excluding date/year).")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price (> 19M)."""
    if "median_list_price" not in df.columns:
        return df
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print(f"✅ Removed {before - after} rows with median_list_price > 19M.")
    return df


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_and_merge(df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    out_path = processed_dir / f"clean_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Preprocessed {split} saved to {out_path} ({df.shape})")
    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR
):
    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir)


if __name__ == "__main__":
    run_preprocess()
