# Data Documentation

## Overview

This document describes all data sources, processing steps, and output files used in the
Traffic-Aware Distribution Center Location project.

---

## Raw Data Sources (`data/raw/`)

### 1. Muhtarlık Adres Bilgileri (GeoJSON)
- **Source:** Istanbul Metropolitan Municipality (IBB) Open Data Portal
- **URL:** https://data.ibb.gov.tr/dataset/muhtarlik-adres-bilgileri/resource/71f75529-7fae-4a85-b05f-664c62eda422
- **Content:** Location records of all muhtarlık (neighborhood office) buildings in Istanbul
- **Fields used:** `Mahalle Adı` (neighborhood name), `İlçe Adı` (district name), `Latitude`, `Longtitude`
- **Coverage:** 963 muhtarlık records → 890 unique neighborhoods after aggregation
- **Role in model:** Provides neighborhood centroid coordinates for travel time calculation

### 2. pivot.csv
- **Source:** Turkish Statistical Institute (TUIK) — MEDAS Address-Based Population Registration System
- **URL:** https://biruni.tuik.gov.tr/medas/
- **Content:** Neighborhood-level population data for Istanbul, year 2025
- **Fields used:** Neighborhood name, district name, population count
- **Coverage:** 961 Istanbul neighborhoods
- **Role in model:** Demand weights `w_i` for each neighborhood

### 3. istanbul_ilce_kira_fiyatlari.csv
- **Source:** Endeksa — Commercial Real Estate Indices
- **URL:** https://www.endeksa.com
- **Content:** District-level commercial rent per m² (TL) for Istanbul districts
- **Fields used:** `İlçe` (district), `Metrekare_Kira_Bedeli_TL` (rent per m²)
- **Coverage:** 40 districts (15 missing districts filled with median)
- **Role in model:** Base rent values `r̄_d(j)` used in opening cost formula `f_j = α · r̄_d(j) · √(Q/Q₀)`

### 4. IBB Traffic Index API (fetched at runtime)
- **Source:** Istanbul Metropolitan Municipality Open Data Portal — CKAN API
- **URL:** https://data.ibb.gov.tr/dataset/istanbul-trafik-indeksi/resource/ba47eacb-a4e1-441c-ae51-0e622d4a18e2
- **Content:** Daily minimum, maximum, and average traffic index values for Istanbul (2015–present)
- **Authentication:** None required
- **Role in model:** Derives traffic multipliers `τ_peak` and `τ_offpeak` for travel time matrices

---

## Processing Steps (`data/prepare_data.py`)

### Step 1 — Neighborhood Coordinates
- Loaded the Muhtarlık GeoJSON (963 records)
- Multiple muhtarlık offices can share the same neighborhood; coordinates were averaged per unique (neighborhood, district) pair
- Result: **890 unique neighborhoods** with centroid coordinates

### Step 2 — Population Data
- Parsed `pivot.csv` using regex to extract neighborhood name, district, and population
- Applied Turkish-aware normalization: Python's `.upper()` maps `'i' → 'I'` instead of the correct `'İ'`, so `replace('i', 'İ')` is applied before `.upper()`
- Result: **961 neighborhoods** with population

### Step 3 — Merging Coordinates and Population
- Left-joined on normalized `(name, district)` key
- **5 neighborhoods** had mismatched names between IBB and TUIK — resolved with manual aliases:

| Muhtarlık name (IBB) | TUIK name | Reason |
|----------------------|-----------|--------|
| BAHÇEKÖY YENİMAHALLE | BAHÇEKÖY YENİ | TUIK regex stops before "Mah." suffix |
| KÜÇÜKÇAMLICA | KÜÇÜK ÇAMLICA | One word vs. two words |
| RUMELİ KAVAĞI | RUMELİKAVAĞI | Two words vs. one word |
| CUMHURİYET / BEYKOZ | CUMHURİYETKÖY | Different suffix |
| SARIYER YENİMAHALLE | SARIYER MERKEZ | Different name entirely |

- **6 neighborhoods** were not present in TUIK data at all — populations were sourced manually from TUIK 2025:

| Neighborhood | District | Population (2025) |
|--------------|----------|-------------------|
| BAĞLARÇEŞME | ESENYURT | 32,537 |
| HAVAALANI | ESENLER | 32,456 |
| MİMAR SİNAN | BÜYÜKÇEKMECE | 9,382 |
| SARAY | ÜMRANİYE | 3,460 |
| YENİŞEHİR | ÜMRANİYE | 6,584 |
| ÇİFTLİK | BEYKOZ | 5,750 |

- Final result: **890 neighborhoods** — all with coordinates and population

> **Coverage note:** Istanbul has approximately 963 official neighborhoods. The IBB Muhtarlık dataset
> contains 890 unique neighborhood locations; the remaining ~73 neighborhoods exist in TUIK population
> data but have no corresponding entry in the Muhtarlık JSON and were therefore excluded. This gives
> ~93% coverage of Istanbul's neighborhoods. Adding the missing coordinates (e.g., from a community
> GeoJSON boundary file) was deemed unnecessary for project purposes.

### Step 4 — Rent Data
- Loaded `istanbul_tum_mahalleler.csv` (neighborhood-level) and `istanbul_tum_ilceler.csv` (district-level) for rent data
- Mismatched neighborhoods were fixed using a dictionary of aliases
- Filled missing mahalle rents with district-level averages, and remaining with the median rent (357 TL/m²)

### Step 5 — Traffic Multipliers (IBB API)
- Fetched 1,000 most recent records from the IBB Traffic Index API
- Traffic index is on a 0–100 congestion scale
- Speed factor model: `speed = free_flow × (1 − 0.5 × index / 100)`
- Multipliers are normalized to average traffic conditions (30 km/h baseline):

```
τ = avg_speed_factor / target_speed_factor
```

- Results from IBB data:
  - `avg_index = 27.8`, `max_index = 60.4`, `min_index = 2.1`
  - `τ_peak = 1.233` — peak hours are ~23% slower than average
  - `τ_offpeak = 0.870` — off-peak hours are ~13% faster than average
- Fallback defaults if API is unavailable: `τ_peak = 1.35`, `τ_offpeak = 0.85`

### Step 6 — Travel Time Matrices
- Computed pairwise Haversine distances between all 890 neighborhood centroids
- Base travel time: `distance_km / 30 km/h`
- Three matrices produced:
  - `travel_times_peak.npy`: base time × `τ_peak`
  - `travel_times_offpeak.npy`: base time × `τ_offpeak`
  - `travel_times.npy`: average of peak and off-peak (used as default `t_ij`)

---

## Output Files (`data/processed/`)

| File | Shape / Size | Description |
|------|-------------|-------------|
| `neighborhoods.csv` | 890 rows × 7 cols | `neighborhood_id`, `name`, `district`, `lat`, `lon`, `population`, `rent_per_m2` |
| `rents.csv` | 39 rows × 2 cols | `district`, `avg_rent_per_m2` |
| `travel_times.npy` | (890, 890) float64 | Blended travel time matrix in hours |
| `travel_times_peak.npy` | (890, 890) float64 | Peak-hour travel time matrix in hours |
| `travel_times_offpeak.npy` | (890, 890) float64 | Off-peak travel time matrix in hours |
