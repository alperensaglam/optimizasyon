#!/usr/bin/env python3
"""
Data preparation script.

Inputs  (data/raw/):
  - Muhtarlık Adres Bilgileri  : IBB muhtarlık GeoJSON (neighborhood coordinates)
  - pivot.csv                  : TUIK neighborhood populations (2025)
  - istanbul_ilce_kira_fiyatlari.csv : Endeksa district-level rent per m²

Outputs (data/processed/):
  - neighborhoods.csv          : neighborhood id, name, district, lat, lon, population (w_i)
  - rents.csv                  : district, avg rent per m² (r̄_d(j))
  - travel_times_peak.npy      : travel time matrix — peak traffic (hours)
  - travel_times_offpeak.npy   : travel time matrix — off-peak traffic (hours)
  - travel_times.npy           : blended (average of peak and off-peak)
"""

import json
import re
import numpy as np
import pandas as pd
import requests

RAW = "data/raw"
OUT = "data/processed"
AVG_SPEED_KMH = 30  # average urban driving speed in Istanbul

# ── helpers ───────────────────────────────────────────────────────────────────

def haversine_matrix(lats, lons):
    """Pairwise haversine distance matrix for all neighborhood pairs (km)."""
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat = lats_r[:, None] - lats_r[None, :]
    dlon = lons_r[:, None] - lons_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats_r[:, None]) * np.cos(lats_r[None, :]) * np.sin(dlon / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def normalize(s):
    # Python's .upper() maps 'i' → 'I', but Turkish requires 'i' → 'İ'
    return s.strip().replace('i', 'İ').upper()

# ── 1. Muhtarlık JSON → coordinates ──────────────────────────────────────────

print("1. Loading Muhtarlık coordinates...")
with open(f"{RAW}/Muhtarlık Adres Bilgileri", encoding="utf-8") as f:
    geojson = json.load(f)

coords = [
    {
        "name":     normalize(feat["properties"]["Mahalle Adı"]),
        "district": normalize(feat["properties"]["İlçe Adı"]),
        "lat":      feat["properties"]["Latitude"],
        "lon":      feat["properties"]["Longtitude"],
    }
    for feat in geojson["features"]
]
df_coords = pd.DataFrame(coords)
# Aynı mahallede birden fazla muhtarlık olabilir — koordinatları ortala
df_coords = df_coords.groupby(["name", "district"], as_index=False).agg({"lat": "mean", "lon": "mean"})
print(f"   {len(df_coords)} unique neighborhoods (after merging multiple muhtarlık per neighborhood)")

# ── 2. pivot.csv → population ────────────────────────────────────────────────

print("2. Loading TUIK population data...")
raw_lines = open(f"{RAW}/pivot.csv", encoding="utf-8-sig").readlines()

pop_rows = []
for line in raw_lines:
    if "İstanbul(" not in line:
        continue
    parts = line.strip().split("|")
    loc_part = next((p for p in parts if "İstanbul(" in p), None)
    pop_str  = next((p for p in parts if re.match(r"^\s*[\d.]+\s*$", p)), None)
    if not loc_part or not pop_str:
        continue
    m = re.search(r"İstanbul\(([^/]+)/[^/]+/(.+?) Mah\.\)", loc_part)
    if not m:
        continue
    pop_rows.append({
        "name":       normalize(m.group(2)),
        "district":   normalize(m.group(1)),
        "population": int(float(pop_str.strip())),
    })

df_pop = pd.DataFrame(pop_rows).drop_duplicates(subset=["name", "district"])
print(f"   {len(df_pop)} neighborhoods with population")

# ── 3. Merge coordinates + population ────────────────────────────────────────

print("3. Merging coordinates and population...")
df_coords["_key"] = df_coords["name"] + "|" + df_coords["district"]
df_pop["_key"]    = df_pop["name"]    + "|" + df_pop["district"]

# Manual aliases: Muhtarlık name → TUIK name (name differs between the two sources)
ALIASES = {
    "BAHÇEKÖY YENİMAHALLE|SARIYER": "BAHÇEKÖY YENİ|SARIYER",
    "KÜÇÜKÇAMLICA|ÜSKÜDAR":         "KÜÇÜK ÇAMLICA|ÜSKÜDAR",
    "RUMELİ KAVAĞI|SARIYER":        "RUMELİKAVAĞI|SARIYER",
    "CUMHURİYET|BEYKOZ":            "CUMHURİYETKÖY|BEYKOZ",
    "SARIYER YENİMAHALLE|SARIYER":  "SARIYER MERKEZ|SARIYER",
}
df_coords["_key"] = df_coords["_key"].replace(ALIASES)

df = df_coords.merge(df_pop[["_key", "population"]], on="_key", how="left").drop(columns="_key")

matched = df["population"].notna().sum()
unmatched = len(df) - matched
print(f"   Matched: {matched}, unmatched: {unmatched}")

if unmatched > 0:
    print("   Unmatched neighborhoods (no population assigned):")
    for _, r in df[df["population"].isna()].iterrows():
        print(f"     {r['name']} / {r['district']}")

# Manual population entries — sourced from TUIK 2025, looked up individually
MANUAL_POP = {
    ("BAĞLARÇEŞME", "ESENYURT"):     32537,
    ("HAVAALANI",   "ESENLER"):      32456,
    ("MİMAR SİNAN", "BÜYÜKÇEKMECE"): 9382,
    ("SARAY",       "ÜMRANİYE"):      3460,
    ("YENİŞEHİR",   "ÜMRANİYE"):      6584,
    ("ÇİFTLİK",     "BEYKOZ"):        5750,
}
for (name, district), pop in MANUAL_POP.items():
    mask = (df["name"] == name) & (df["district"] == district)
    df.loc[mask, "population"] = pop
    print(f"   Manual population set: {name} / {district} → {pop:,}")

# Drop neighborhoods with no population (cannot be used in the model)
df = df.dropna(subset=["population"]).reset_index(drop=True)
df["population"] = df["population"].astype(int)
df.index.name = "neighborhood_id"
df = df.reset_index()

print(f"   Final: {len(df)} neighborhoods")

# ── 4. Rent data ──────────────────────────────────────────────────────────────

print("4. Processing Endeksa rent data (neighborhood + district)...")

# Parse district level data first
df_dist_rent = pd.read_csv(f"{RAW}/istanbul_tum_ilceler.csv", sep=";")
def parse_rent(val):
    if pd.isna(val) or val == "-" or "₺" not in str(val): return np.nan
    v = str(val).split("₺")[0].replace(".", "").strip()
    try: return float(v)
    except: return np.nan

df_dist_rent["avg_rent_per_m2"] = df_dist_rent["Birim Fiyatı (₺/m2)"].apply(parse_rent)
df_dist_rent["district"] = df_dist_rent["Mahalle"].apply(normalize)
df_dist_rent = df_dist_rent.dropna(subset=["avg_rent_per_m2"])
district_rents = dict(zip(df_dist_rent["district"], df_dist_rent["avg_rent_per_m2"]))
median_district_rent = df_dist_rent["avg_rent_per_m2"].median()

# Parse neighborhood level data
df_mah_rent = pd.read_csv(f"{RAW}/istanbul_tum_mahalleler.csv", sep=";")
df_mah_rent["rent"] = df_mah_rent["Birim Fiyatı (₺/m2)"].apply(parse_rent)
df_mah_rent["name"] = df_mah_rent["Mahalle"].apply(normalize)
df_mah_rent["district"] = df_mah_rent["İlçe"].apply(normalize)
df_mah_rent["_rent_key"] = df_mah_rent["name"] + "|" + df_mah_rent["district"]
df_mah_rent = df_mah_rent.dropna(subset=["rent"])
mah_rents = dict(zip(df_mah_rent["_rent_key"], df_mah_rent["rent"]))

ALIASES_RENT = {
    "AKŞEMSETTİN|SULTANBEYLİ": "AKŞEMSEDDİN|SULTANBEYLİ",
    "ATAKÖY 2-5-6. KISIM|BAKIRKÖY": "ATAKÖY 2. 5. 6. KISIM|BAKIRKÖY",
    "AŞIKVEYSEL|ATAŞEHİR": "AŞIK VEYSEL|ATAŞEHİR",
    "KAMER HATUN|BEYOĞLU": "KAMERHATUN|BEYOĞLU",
    "KEÇECİ PİRİ|BEYOĞLU": "KEÇECİPİRİ|BEYOĞLU",
    "KUMKÖY (KİLYOS)|SARIYER": "KUMKÖY|SARIYER",
    "MERKEZ|BEYKOZ": "BEYKOZ MERKEZ|BEYKOZ",
    "MERKEZ|EYÜPSULTAN": "EYÜP MERKEZ|EYÜPSULTAN",
    "MURAT ÇESME|BÜYÜKÇEKMECE": "MURAT ÇEŞME|BÜYÜKÇEKMECE",
    "MUSTAFA KEMALPAŞA|AVCILAR": "MUSTAFA KEMAL PAŞA|AVCILAR",
    "MİMAR SİNAN|BÜYÜKÇEKMECE": "MİMARSİNAN|BÜYÜKÇEKMECE",
    "NENEHATUN|ARNAVUTKÖY": "NENE HATUN|ARNAVUTKÖY",
    "NİŞANCI|EYÜPSULTAN": "NİŞANCA|EYÜPSULTAN",
    "ORHANGAZİ|SULTANBEYLİ": "ORHAN GAZİ|SULTANBEYLİ",
    "SURURİ MEHMET EFENDİ|BEYOĞLU": "SURURİ|BEYOĞLU",
    "ÇİFTLİK|BEYKOZ": "ÇAVUŞBAŞI ÇİFTLİK|BEYKOZ",
    "İSMET PAŞA|BAYRAMPAŞA": "İSMETPAŞA|BAYRAMPAŞA",
    "İSMETPAŞA|SULTANGAZİ": "İSMET PAŞA|SULTANGAZİ",
    "PİRİPAŞA|BEYOĞLU": "PİRİ MEHMET PAŞA|BEYOĞLU",
}

# Add rent column to df
df["_rent_key"] = (df["name"] + "|" + df["district"]).replace(ALIASES_RENT)
df["rent_per_m2"] = df["_rent_key"].map(mah_rents)

missing_mah = df["rent_per_m2"].isna().sum()
print(f"   {len(df) - missing_mah} neighborhoods matched with mahalle rent")

# Fill missing mahalle rent with district rent
df.loc[df["rent_per_m2"].isna(), "rent_per_m2"] = df.loc[df["rent_per_m2"].isna(), "district"].map(district_rents)
missing_dist = df["rent_per_m2"].isna().sum()
print(f"   {missing_mah - missing_dist} filled with district rent")

# Fill remaining with median district rent
if missing_dist > 0:
    df.loc[df["rent_per_m2"].isna(), "rent_per_m2"] = median_district_rent
    print(f"   {missing_dist} filled with median district rent ({median_district_rent:.0f})")

df = df.drop(columns=["_rent_key"])
df.to_csv(f"{OUT}/neighborhoods.csv", index=False)
print(f"   Saved → {OUT}/neighborhoods.csv (with rent_per_m2)")

df_dist_rent[["district", "avg_rent_per_m2"]].to_csv(f"{OUT}/rents.csv", index=False)
print(f"   Saved → {OUT}/rents.csv (district level, for compatibility)")

# ── 5. IBB Traffic API → τ multipliers ───────────────────────────────────────

print("5. Fetching traffic data from IBB Traffic Index API...")
RESOURCE_ID = "ba47eacb-a4e1-441c-ae51-0e622d4a18e2"
API_URL = "https://data.ibb.gov.tr/api/3/action/datastore_search"

tau_peak, tau_offpeak = 1.35, 0.85  # fallback defaults

try:
    resp = requests.get(
        API_URL,
        params={"resource_id": RESOURCE_ID, "limit": 1000},
        timeout=15,
    )
    resp.raise_for_status()
    records = resp.json()["result"]["records"]
    df_traffic = pd.DataFrame(records)
    print(f"   {len(records)} records fetched, columns: {list(df_traffic.columns)}")

    avg_col = next(c for c in df_traffic.columns if "ortalama" in c.lower() or "average" in c.lower())
    max_col = next(c for c in df_traffic.columns if "maksimum" in c.lower() or "maximum" in c.lower())
    min_col = next(c for c in df_traffic.columns if "minimum" in c.lower())

    for col in [avg_col, max_col, min_col]:
        df_traffic[col] = pd.to_numeric(df_traffic[col], errors="coerce")

    avg_of_avgs = pd.to_numeric(df_traffic[avg_col], errors="coerce").mean()
    avg_of_maxes = pd.to_numeric(df_traffic[max_col], errors="coerce").mean()
    avg_of_mins = pd.to_numeric(df_traffic[min_col], errors="coerce").mean()

    # Index is on a 0-100 congestion scale.
    # Speed factor: speed = free_flow × (1 - 0.5 × index/100)
    # τ = avg_speed_factor / target_speed_factor
    spd_avg  = 1 - 0.5 * avg_of_avgs  / 100  # average traffic conditions
    spd_peak = 1 - 0.5 * avg_of_maxes / 100  # peak hour (daily max average)
    spd_off  = 1 - 0.5 * avg_of_mins  / 100  # off-peak (daily min average)

    tau_peak    = spd_avg / spd_peak
    tau_offpeak = spd_avg / spd_off
    print(f"   avg_index={avg_of_avgs:.1f}, max_index={avg_of_maxes:.1f}, min_index={avg_of_mins:.1f}")
    print(f"   τ_peak={tau_peak:.3f}, τ_offpeak={tau_offpeak:.3f} (from IBB data)")

except Exception as e:
    print(f"   API error ({e}), using fallback defaults: τ_peak={tau_peak}, τ_offpeak={tau_offpeak}")

# ── 6. Travel time matrices ───────────────────────────────────────────────────

print("6. Computing travel time matrices...")
lats = df["lat"].values
lons = df["lon"].values
n = len(df)

dist_km   = haversine_matrix(lats, lons)       # (n, n) km
base_time = dist_km / AVG_SPEED_KMH            # (n, n) hours

tt_peak    = base_time * tau_peak
tt_offpeak = base_time * tau_offpeak
tt_blended = (tt_peak + tt_offpeak) / 2

np.save(f"{OUT}/travel_times_peak.npy",    tt_peak)
np.save(f"{OUT}/travel_times_offpeak.npy", tt_offpeak)
np.save(f"{OUT}/travel_times.npy",         tt_blended)

print(f"   Matrix size: {n}×{n}")
print(f"   Saved → travel_times_peak.npy, travel_times_offpeak.npy, travel_times.npy")

print("\nDone. data/processed/ contents:")
import os
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(f"{OUT}/{f}")
    print(f"   {f}  ({size/1024:.1f} KB)")
