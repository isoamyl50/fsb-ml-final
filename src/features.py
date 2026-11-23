"""Pluggable feature engineering recipes for continuous house price regression."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ADDRESS_COL, RANDOM_SEED, RAW_NUM_COLS
from .preprocessing import normalize_vietnamese_text, parse_address

RowList = Sequence[Mapping[str, str]]
FeatureBuilder = Callable[..., "FeatureRecipe"]

FEATURE_REGISTRY: dict[str, FeatureBuilder] = {}


def register_features(name: str) -> Callable[[FeatureBuilder], FeatureBuilder]:
    """Decorator registering a named feature recipe."""

    def _decorator(fn: FeatureBuilder) -> FeatureBuilder:
        FEATURE_REGISTRY[name] = fn
        return fn

    return _decorator


def list_feature_recipes() -> list[str]:
    """Return sorted registered feature recipe names."""
    return sorted(FEATURE_REGISTRY.keys())


def build_feature_recipe(
    name: str,
    scale_numeric: bool = True,
    preserve_nans: bool = False,
) -> "FeatureRecipe":
    """Instantiate a registered feature recipe."""
    if name not in FEATURE_REGISTRY:
        raise KeyError(f"Unknown feature recipe '{name}'. Available: {list_feature_recipes()}")
    return FEATURE_REGISTRY[name](scale_numeric=scale_numeric, preserve_nans=preserve_nans)


def build_numeric_only_preprocessor(
    numeric_cols: Sequence[str] = RAW_NUM_COLS,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Baseline numeric-only preprocessor for preprocessing ablation comparisons."""
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[("num", Pipeline(steps), list(numeric_cols))],
        remainder="drop",
    )


def _to_float(val: Any) -> float:
    if val is None:
        return float("nan")
    s = str(val).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _extract_raw_numeric_matrix(rows: RowList) -> np.ndarray:
    return np.array(
        [[_to_float(r.get(col)) for col in RAW_NUM_COLS] for r in rows],
        dtype=np.float32,
    )


def _extract_street(address: Any, district: str) -> str:
    norm = normalize_vietnamese_text(address)
    if not norm:
        return f"Unknown | {district}"
    parts = [normalize_vietnamese_text(p) for p in norm.split(",") if p.strip()]
    if len(parts) >= 4:
        return f"{parts[0]} | {district}"
    return f"Unknown | {district}"


def _extract_geo_and_cat(
    rows: RowList,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str], list[str], list[str], list[str], list[str]]:
    cities: list[str] = []
    districts: list[str] = []
    wards: list[str] = []
    streets: list[str] = []
    addr_texts: list[str] = []
    parsed_list: list[dict[str, Any]] = []
    cat_rows: list[list[str]] = []

    for r in rows:
        raw_addr = r.get(ADDRESS_COL)
        p = parse_address(raw_addr)
        c = p["city"]
        d = p["district"]
        w_full = f"{p['ward']} | {d}"
        s = _extract_street(raw_addr, d)

        parsed_list.append(p)
        cities.append(c)
        districts.append(d)
        wards.append(w_full)
        streets.append(s)
        addr_texts.append(normalize_vietnamese_text(raw_addr))
        cat_rows.append(
            [
                c,
                d,
                p["ward"],
                normalize_vietnamese_text(r.get("Legal status")) or "Missing",
                normalize_vietnamese_text(r.get("Furniture state")) or "Missing",
                normalize_vietnamese_text(r.get("House direction")) or "Missing",
            ]
        )

    return (
        np.array(cat_rows, dtype=object),
        parsed_list,
        cities,
        districts,
        wards,
        streets,
        addr_texts,
    )


def _fit_grouped_medians(
    values: np.ndarray,
    group_keys: Sequence[str],
) -> tuple[dict[str, float], float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    valid_all: list[float] = []
    for val, key in zip(values, group_keys):
        if not np.isnan(val):
            buckets[key].append(float(val))
            valid_all.append(float(val))
    global_med = float(np.median(valid_all)) if valid_all else 0.0
    group_med = {k: float(np.median(vs)) for k, vs in buckets.items()}
    return group_med, global_med


def _apply_grouped_medians(
    values: np.ndarray,
    group_keys: Sequence[str],
    group_med: dict[str, float],
    global_med: float,
) -> np.ndarray:
    out = values.copy()
    for idx, (val, key) in enumerate(zip(out, group_keys)):
        if np.isnan(val):
            out[idx] = group_med.get(key, global_med)
    return out


def _smoothed_mean_map(
    keys: Sequence[str],
    values: np.ndarray,
    prior_values: np.ndarray | float,
    smoothing: float,
) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    prior_sums: dict[str, float] = defaultdict(float)

    for idx, (key, val) in enumerate(zip(keys, values)):
        sums[key] += float(val)
        counts[key] += 1
        p_val = float(prior_values) if np.ndim(prior_values) == 0 else float(prior_values[idx])
        prior_sums[key] += p_val

    out: dict[str, float] = {}
    for key, cnt in counts.items():
        mean_prior = prior_sums[key] / cnt
        out[key] = (sums[key] + smoothing * mean_prior) / (cnt + smoothing)
    return out


def _fit_hierarchical_target_means(
    cities: Sequence[str],
    districts: Sequence[str],
    wards: Sequence[str],
    streets: Sequence[str],
    values: np.ndarray,
) -> dict[str, Any]:
    global_mean = float(np.mean(values))
    city_map = _smoothed_mean_map(cities, values, global_mean, smoothing=10.0)
    city_priors = np.array([city_map.get(c, global_mean) for c in cities], dtype=float)

    dist_map = _smoothed_mean_map(districts, values, city_priors, smoothing=10.0)
    dist_priors = np.array(
        [dist_map.get(d, city_map.get(c, global_mean)) for d, c in zip(districts, cities)],
        dtype=float,
    )

    ward_map = _smoothed_mean_map(wards, values, dist_priors, smoothing=10.0)
    ward_priors = np.array(
        [
            ward_map.get(w, dist_map.get(d, city_map.get(c, global_mean)))
            for w, d, c in zip(wards, districts, cities)
        ],
        dtype=float,
    )

    street_map = _smoothed_mean_map(streets, values, ward_priors, smoothing=8.0)
    return {
        "global": global_mean,
        "city": city_map,
        "district": dist_map,
        "ward": ward_map,
        "street": street_map,
    }


def _apply_hierarchical_target_means(
    cities: Sequence[str],
    districts: Sequence[str],
    wards: Sequence[str],
    streets: Sequence[str],
    state: dict[str, Any],
) -> np.ndarray:
    g = float(state["global"])
    c_map = state["city"]
    d_map = state["district"]
    w_map = state["ward"]
    s_map = state["street"]

    c_vals = np.array([c_map.get(c, g) for c in cities], dtype=np.float32)
    d_vals = np.array([d_map.get(d, cv) for d, cv in zip(districts, c_vals)], dtype=np.float32)
    w_vals = np.array([w_map.get(w, dv) for w, dv in zip(wards, d_vals)], dtype=np.float32)
    s_vals = np.array([s_map.get(s, wv) for s, wv in zip(streets, w_vals)], dtype=np.float32)
    return np.column_stack([c_vals, d_vals, w_vals, s_vals]).astype(np.float32)


def _price_decile_strata(y_price: np.ndarray) -> np.ndarray:
    edges = np.quantile(y_price, np.linspace(0.1, 0.9, 9))
    return np.digitize(y_price, bins=edges, right=False)


class FeatureRecipe:
    """Base interface for strictly train-fit tabular regression feature recipes."""

    name: str = "base"

    def __init__(self, scale_numeric: bool = True, preserve_nans: bool = False) -> None:
        self.scale_numeric = scale_numeric
        self.preserve_nans = preserve_nans
        self.feature_names: list[str] = []

    def fit_transform(
        self,
        train_rows: RowList,
        y_train_price: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def transform(self, rows: RowList) -> np.ndarray:
        raise NotImplementedError


@register_features("P0_numeric")
class P0NumericRecipe(FeatureRecipe):
    """6 raw numeric features (baseline for preprocessing ablation)."""

    name = "P0_numeric"

    def __init__(self, scale_numeric: bool = True, preserve_nans: bool = False) -> None:
        super().__init__(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        self.feature_names = list(RAW_NUM_COLS)

    def fit_transform(
        self,
        train_rows: RowList,
        y_train_price: np.ndarray | None = None,
    ) -> np.ndarray:
        x_num = _extract_raw_numeric_matrix(train_rows)
        if not self.preserve_nans:
            x_num = self._imputer.fit_transform(x_num)
        if self.scale_numeric:
            x_num = self._scaler.fit_transform(x_num)
        return np.asarray(x_num, dtype=np.float32)

    def transform(self, rows: RowList) -> np.ndarray:
        x_num = _extract_raw_numeric_matrix(rows)
        if not self.preserve_nans:
            x_num = self._imputer.transform(x_num)
        if self.scale_numeric:
            x_num = self._scaler.transform(x_num)
        return np.asarray(x_num, dtype=np.float32)


@register_features("P1_location")
class P1LocationRecipe(FeatureRecipe):
    """Raw numerics + NFC-normalized City/District/Ward & raw categoricals (OHE)."""

    name = "P1_location"
    _CAT_NAMES = ["city", "district", "ward", "Legal status", "Furniture state", "House direction"]
    _TOP_LIMITS = {"city": 25, "district": 120, "ward": 120}

    def __init__(self, scale_numeric: bool = True, preserve_nans: bool = False) -> None:
        super().__init__(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._num_recipe = P0NumericRecipe(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._top_vocabs: dict[int, set[str]] = {}
        self._ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)

    def _clip_categories(self, cat_mat: np.ndarray) -> np.ndarray:
        clipped = cat_mat.copy()
        for col_idx, vocab in self._top_vocabs.items():
            for row_idx in range(len(clipped)):
                if clipped[row_idx, col_idx] not in vocab:
                    clipped[row_idx, col_idx] = "Other"
        return clipped

    def fit_transform(
        self,
        train_rows: RowList,
        y_train_price: np.ndarray | None = None,
    ) -> np.ndarray:
        x_num = self._num_recipe.fit_transform(train_rows, y_train_price)
        cat_mat, *_ = _extract_geo_and_cat(train_rows)

        self._top_vocabs = {}
        for col_idx, col_name in enumerate(self._CAT_NAMES):
            limit = self._TOP_LIMITS.get(col_name)
            if limit is not None:
                counts = Counter(cat_mat[:, col_idx].tolist())
                self._top_vocabs[col_idx] = {k for k, _ in counts.most_common(limit)}

        x_cat = self._ohe.fit_transform(self._clip_categories(cat_mat))
        self.feature_names = list(self._num_recipe.feature_names) + list(
            self._ohe.get_feature_names_out(self._CAT_NAMES)
        )
        return np.hstack([x_num, x_cat]).astype(np.float32)

    def transform(self, rows: RowList) -> np.ndarray:
        x_num = self._num_recipe.transform(rows)
        cat_mat, *_ = _extract_geo_and_cat(rows)
        x_cat = self._ohe.transform(self._clip_categories(cat_mat))
        return np.hstack([x_num, x_cat]).astype(np.float32)


@register_features("P2_domain")
class P2DomainRecipe(FeatureRecipe):
    """P1 + grouped median imputation + missing flags + log transforms + structural ratios."""

    name = "P2_domain"
    _OPTIONAL_FIELDS = [
        "Frontage",
        "Access Road",
        "House direction",
        "Balcony direction",
        "Floors",
        "Bedrooms",
        "Bathrooms",
        "Legal status",
        "Furniture state",
    ]

    def __init__(self, scale_numeric: bool = True, preserve_nans: bool = False) -> None:
        super().__init__(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self._top_vocabs: dict[int, set[str]] = {}
        self._group_medians: dict[int, tuple[dict[str, float], float]] = {}
        self._fallback_imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()

    def _compute_numeric_and_flags(
        self,
        rows: RowList,
        parsed_list: list[dict[str, Any]],
        fit: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_num = _extract_raw_numeric_matrix(rows)
        districts = [p["district"] for p in parsed_list]

        n_missing = np.array(
            [
                float(sum(1 for col in self._OPTIONAL_FIELDS if not str(r.get(col, "")).strip()))
                for r in rows
            ],
            dtype=np.float32,
        )

        flags = np.column_stack(
            [
                np.isnan(raw_num[:, 1]).astype(np.float32),
                np.isnan(raw_num[:, 2]).astype(np.float32),
                np.array(
                    [0.0 if str(r.get("Furniture state", "")).strip() else 1.0 for r in rows],
                    dtype=np.float32,
                ),
                np.array(
                    [0.0 if str(r.get("House direction", "")).strip() else 1.0 for r in rows],
                    dtype=np.float32,
                ),
                np.array(
                    [0.0 if str(r.get("Balcony direction", "")).strip() else 1.0 for r in rows],
                    dtype=np.float32,
                ),
                np.array([float(p["is_project"]) for p in parsed_list], dtype=np.float32),
                np.array([float(p["is_alley"]) for p in parsed_list], dtype=np.float32),
            ]
        )

        if not self.preserve_nans:
            for col_idx in (1, 2):
                if fit:
                    self._group_medians[col_idx] = _fit_grouped_medians(raw_num[:, col_idx], districts)
                g_med, glob_med = self._group_medians[col_idx]
                raw_num[:, col_idx] = _apply_grouped_medians(
                    raw_num[:, col_idx], districts, g_med, glob_med
                )
            raw_num = (
                self._fallback_imputer.fit_transform(raw_num)
                if fit
                else self._fallback_imputer.transform(raw_num)
            )

        area = raw_num[:, 0]
        frontage = raw_num[:, 1]
        access_road = raw_num[:, 2]
        floors = raw_num[:, 3]
        bedrooms = raw_num[:, 4]
        bathrooms = raw_num[:, 5]

        log_area = np.log1p(np.maximum(area, 0.0))
        log_front = np.where(np.isnan(frontage), np.nan, np.log1p(np.maximum(frontage, 0.0)))
        log_road = np.where(np.isnan(access_road), np.nan, np.log1p(np.maximum(access_road, 0.0)))
        usable_area = np.where(np.isnan(floors), np.nan, area * np.maximum(floors, 1.0))
        log_usable = np.where(np.isnan(usable_area), np.nan, np.log1p(np.maximum(usable_area, 0.0)))

        area_per_floor = np.where(floors > 0, area / np.maximum(floors, 1.0), np.nan)
        total_rooms = bedrooms + bathrooms
        area_per_bedroom = np.where(bedrooms > 0, area / np.maximum(bedrooms, 1.0), np.nan)
        frontage_road_ratio = np.where(
            access_road > 0, frontage / np.maximum(access_road, 0.5), np.nan
        )

        cont = np.column_stack(
            [
                raw_num,
                log_area,
                log_front,
                log_road,
                usable_area,
                log_usable,
                area_per_floor,
                total_rooms,
                area_per_bedroom,
                frontage_road_ratio,
                n_missing,
            ]
        ).astype(np.float32)

        if not self.preserve_nans:
            cont = np.nan_to_num(cont, nan=0.0)
        if self.scale_numeric:
            cont = self._scaler.fit_transform(cont) if fit else self._scaler.transform(cont)

        return cont.astype(np.float32), flags

    def _prep_cats(self, cat_mat: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._top_vocabs = {}
            for col_idx, col_name in enumerate(P1LocationRecipe._CAT_NAMES):
                limit = P1LocationRecipe._TOP_LIMITS.get(col_name)
                if limit is not None:
                    counts = Counter(cat_mat[:, col_idx].tolist())
                    self._top_vocabs[col_idx] = {k for k, _ in counts.most_common(limit)}
        clipped = cat_mat.copy()
        for col_idx, vocab in self._top_vocabs.items():
            for row_idx in range(len(clipped)):
                if clipped[row_idx, col_idx] not in vocab:
                    clipped[row_idx, col_idx] = "Other"
        return self._ohe.fit_transform(clipped) if fit else self._ohe.transform(clipped)

    def fit_transform(
        self,
        train_rows: RowList,
        y_train_price: np.ndarray | None = None,
    ) -> np.ndarray:
        cat_mat, parsed_list, *_ = _extract_geo_and_cat(train_rows)
        cont, flags = self._compute_numeric_and_flags(train_rows, parsed_list, fit=True)
        x_cat = self._prep_cats(cat_mat, fit=True)

        cont_names = list(RAW_NUM_COLS) + [
            "log1p_Area",
            "log1p_Frontage",
            "log1p_AccessRoad",
            "usable_area",
            "log1p_usable_area",
            "Area_per_Floor",
            "Total_Rooms",
            "Area_per_Bedroom",
            "Frontage_Road_Ratio",
            "n_missing_fields",
        ]
        flag_names = [
            "Frontage_missing",
            "AccessRoad_missing",
            "Furniture_missing",
            "HouseDir_missing",
            "BalconyDir_missing",
            "is_project",
            "is_alley",
        ]
        self.feature_names = (
            cont_names
            + flag_names
            + list(self._ohe.get_feature_names_out(P1LocationRecipe._CAT_NAMES))
        )
        return np.hstack([cont, flags, x_cat]).astype(np.float32)

    def transform(self, rows: RowList) -> np.ndarray:
        cat_mat, parsed_list, *_ = _extract_geo_and_cat(rows)
        cont, flags = self._compute_numeric_and_flags(rows, parsed_list, fit=False)
        x_cat = self._prep_cats(cat_mat, fit=False)
        return np.hstack([cont, flags, x_cat]).astype(np.float32)


@register_features("P3_tfidf_te")
class P3TfidfTargetEncRecipe(FeatureRecipe):
    """P2 + 5-fold OOF continuous Target Encoding (city/district/ward/street) + TF-IDF SVD(20)."""

    name = "P3_tfidf_te"

    def __init__(
        self,
        scale_numeric: bool = True,
        preserve_nans: bool = False,
        svd_components: int = 20,
    ) -> None:
        super().__init__(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._p2 = P2DomainRecipe(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self.svd_components = svd_components
        self._te_state: dict[str, Any] = {}
        self._tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
        self._svd = TruncatedSVD(n_components=svd_components, random_state=RANDOM_SEED)
        self._extra_scaler = StandardScaler()

    def fit_transform(
        self,
        train_rows: RowList,
        y_train_price: np.ndarray | None = None,
    ) -> np.ndarray:
        if y_train_price is None:
            raise ValueError("P3_tfidf_te requires y_train_price during fit_transform().")

        p_arr = np.asarray(y_train_price, dtype=float)
        x_p2 = self._p2.fit_transform(train_rows, p_arr)
        _, _, cities, districts, wards, streets, addr_texts = _extract_geo_and_cat(train_rows)

        strata = _price_decile_strata(p_arr)
        oof_te = np.zeros((len(train_rows), 4), dtype=np.float32)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for tr_i, oof_i in skf.split(np.zeros(len(train_rows)), strata):
            fold_state = _fit_hierarchical_target_means(
                [cities[i] for i in tr_i],
                [districts[i] for i in tr_i],
                [wards[i] for i in tr_i],
                [streets[i] for i in tr_i],
                p_arr[tr_i],
            )
            oof_te[oof_i] = _apply_hierarchical_target_means(
                [cities[i] for i in oof_i],
                [districts[i] for i in oof_i],
                [wards[i] for i in oof_i],
                [streets[i] for i in oof_i],
                fold_state,
            )

        self._te_state = _fit_hierarchical_target_means(cities, districts, wards, streets, p_arr)
        svd_mat = self._svd.fit_transform(self._tfidf.fit_transform(addr_texts)).astype(np.float32)

        extra = np.hstack([oof_te, svd_mat]).astype(np.float32)
        if self.scale_numeric:
            extra = self._extra_scaler.fit_transform(extra).astype(np.float32)

        te_names = ["te_city_price", "te_district_price", "te_ward_price", "te_street_price"]
        svd_names = [f"addr_svd_{i}" for i in range(self.svd_components)]
        self.feature_names = list(self._p2.feature_names) + te_names + svd_names
        return np.hstack([x_p2, extra]).astype(np.float32)

    def transform(self, rows: RowList) -> np.ndarray:
        x_p2 = self._p2.transform(rows)
        _, _, cities, districts, wards, streets, addr_texts = _extract_geo_and_cat(rows)
        te_mat = _apply_hierarchical_target_means(cities, districts, wards, streets, self._te_state)
        svd_mat = self._svd.transform(self._tfidf.transform(addr_texts)).astype(np.float32)
        extra = np.hstack([te_mat, svd_mat]).astype(np.float32)
        if self.scale_numeric:
            extra = self._extra_scaler.transform(extra).astype(np.float32)
        return np.hstack([x_p2, extra]).astype(np.float32)


@register_features("P4_geo_priors")
class P4GeoPriorsRecipe(FeatureRecipe):
    """P3 + 5-fold OOF hierarchical Price/m² priors + OOF Ridge TF-IDF text predictions."""

    name = "P4_geo_priors"

    def __init__(self, scale_numeric: bool = True, preserve_nans: bool = False) -> None:
        super().__init__(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._p3 = P3TfidfTargetEncRecipe(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._ppm2_state: dict[str, Any] = {}
        self._freq_maps: dict[str, Counter[str]] = {}
        self._addr_tfidf = TfidfVectorizer(
            ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True
        )
        self._addr_ridge_price = Ridge(alpha=2.0, random_state=RANDOM_SEED)
        self._addr_ridge_ppm2 = Ridge(alpha=2.0, random_state=RANDOM_SEED)
        self._extra_scaler = StandardScaler()

    @staticmethod
    def _compute_price_priors(
        cities: Sequence[str],
        districts: Sequence[str],
        wards: Sequence[str],
        streets: Sequence[str],
        areas: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        ppm2_mat = _apply_hierarchical_target_means(cities, districts, wards, streets, state)
        safe_areas = np.maximum(areas.astype(np.float32), 1.0)
        est_d = (safe_areas * ppm2_mat[:, 1]) / 1000.0
        est_w = (safe_areas * ppm2_mat[:, 2]) / 1000.0
        est_s = (safe_areas * ppm2_mat[:, 3]) / 1000.0
        return np.column_stack(
            [
                ppm2_mat,
                est_d,
                est_w,
                est_s,
                np.log1p(np.maximum(est_d, 0.0)),
                np.log1p(np.maximum(est_w, 0.0)),
            ]
        ).astype(np.float32)

    def _compute_freq_feats(
        self,
        cities: Sequence[str],
        districts: Sequence[str],
        wards: Sequence[str],
        streets: Sequence[str],
    ) -> np.ndarray:
        return np.column_stack(
            [
                np.log1p([float(self._freq_maps["city"].get(k, 0)) for k in cities]),
                np.log1p([float(self._freq_maps["district"].get(k, 0)) for k in districts]),
                np.log1p([float(self._freq_maps["ward"].get(k, 0)) for k in wards]),
                np.log1p([float(self._freq_maps["street"].get(k, 0)) for k in streets]),
            ]
        ).astype(np.float32)

    def fit_transform(
        self,
        train_rows: RowList,
        y_train_price: np.ndarray | None = None,
    ) -> np.ndarray:
        if y_train_price is None:
            raise ValueError("P4_geo_priors requires y_train_price during fit_transform().")

        p_arr = np.asarray(y_train_price, dtype=float)
        x_p3 = self._p3.fit_transform(train_rows, p_arr)
        _, _, cities, districts, wards, streets, addr_texts = _extract_geo_and_cat(train_rows)

        self._freq_maps = {
            "city": Counter(cities),
            "district": Counter(districts),
            "ward": Counter(wards),
            "street": Counter(streets),
        }

        areas = np.array([_to_float(r.get("Area")) for r in train_rows], dtype=float)
        ppm2 = np.clip((p_arr * 1000.0) / np.maximum(areas, 1.0), 10.0, 400.0)
        strata = _price_decile_strata(p_arr)

        oof_priors = np.zeros((len(train_rows), 9), dtype=np.float32)
        oof_txt_price = np.zeros(len(train_rows), dtype=np.float32)
        oof_txt_ppm2 = np.zeros(len(train_rows), dtype=np.float32)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for tr_i, oof_i in skf.split(np.zeros(len(train_rows)), strata):
            fold_state = _fit_hierarchical_target_means(
                [cities[i] for i in tr_i],
                [districts[i] for i in tr_i],
                [wards[i] for i in tr_i],
                [streets[i] for i in tr_i],
                ppm2[tr_i],
            )
            oof_priors[oof_i] = self._compute_price_priors(
                [cities[i] for i in oof_i],
                [districts[i] for i in oof_i],
                [wards[i] for i in oof_i],
                [streets[i] for i in oof_i],
                areas[oof_i],
                fold_state,
            )

            fold_tfidf = TfidfVectorizer(
                ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True
            )
            x_tr_txt = fold_tfidf.fit_transform([addr_texts[i] for i in tr_i])
            x_oof_txt = fold_tfidf.transform([addr_texts[i] for i in oof_i])

            rp = Ridge(alpha=2.0, random_state=RANDOM_SEED)
            rp.fit(x_tr_txt, p_arr[tr_i])
            oof_txt_price[oof_i] = rp.predict(x_oof_txt).astype(np.float32)

            ru = Ridge(alpha=2.0, random_state=RANDOM_SEED)
            ru.fit(x_tr_txt, ppm2[tr_i])
            oof_txt_ppm2[oof_i] = ru.predict(x_oof_txt).astype(np.float32)

        self._ppm2_state = _fit_hierarchical_target_means(cities, districts, wards, streets, ppm2)
        x_all_txt = self._addr_tfidf.fit_transform(addr_texts)
        self._addr_ridge_price.fit(x_all_txt, p_arr)
        self._addr_ridge_ppm2.fit(x_all_txt, ppm2)

        freq_mat = self._compute_freq_feats(cities, districts, wards, streets)
        extra = np.column_stack([oof_priors, oof_txt_price, oof_txt_ppm2, freq_mat]).astype(
            np.float32
        )
        if self.scale_numeric:
            extra = self._extra_scaler.fit_transform(extra).astype(np.float32)

        extra_names = [
            "oof_city_ppm2",
            "oof_dist_ppm2",
            "oof_ward_ppm2",
            "oof_street_ppm2",
            "est_price_district",
            "est_price_ward",
            "est_price_street",
            "log1p_est_price_district",
            "log1p_est_price_ward",
            "addr_ridge_price",
            "addr_ridge_ppm2",
            "log1p_freq_city",
            "log1p_freq_district",
            "log1p_freq_ward",
            "log1p_freq_street",
        ]
        self.feature_names = list(self._p3.feature_names) + extra_names
        return np.hstack([x_p3, extra]).astype(np.float32)

    def transform(self, rows: RowList) -> np.ndarray:
        x_p3 = self._p3.transform(rows)
        _, _, cities, districts, wards, streets, addr_texts = _extract_geo_and_cat(rows)
        areas = np.array([_to_float(r.get("Area")) for r in rows], dtype=float)

        priors_mat = self._compute_price_priors(
            cities, districts, wards, streets, areas, self._ppm2_state
        )
        x_txt = self._addr_tfidf.transform(addr_texts)
        txt_price = self._addr_ridge_price.predict(x_txt).astype(np.float32)
        txt_ppm2 = self._addr_ridge_ppm2.predict(x_txt).astype(np.float32)
        freq_mat = self._compute_freq_feats(cities, districts, wards, streets)

        extra = np.column_stack([priors_mat, txt_price, txt_ppm2, freq_mat]).astype(np.float32)
        if self.scale_numeric:
            extra = self._extra_scaler.transform(extra).astype(np.float32)
        return np.hstack([x_p3, extra]).astype(np.float32)


@register_features("P4_compact")
class P4CompactRecipe(FeatureRecipe):
    """Compact dense subset of P4_geo_priors dropping sparse district_*/ward_* OHE columns."""

    name = "P4_compact"

    def __init__(self, scale_numeric: bool = True, preserve_nans: bool = False) -> None:
        super().__init__(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._p4 = P4GeoPriorsRecipe(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._keep_mask: np.ndarray = np.array([], dtype=bool)

    def fit_transform(
        self,
        train_rows: RowList,
        y_train_price: np.ndarray | None = None,
    ) -> np.ndarray:
        x_full = self._p4.fit_transform(train_rows, y_train_price=y_train_price)
        self._keep_mask = np.array(
            [
                not (name.startswith("district_") or name.startswith("ward_"))
                for name in self._p4.feature_names
            ],
            dtype=bool,
        )
        self.feature_names = [
            name for name, keep in zip(self._p4.feature_names, self._keep_mask) if keep
        ]
        return x_full[:, self._keep_mask].astype(np.float32)

    def transform(self, rows: RowList) -> np.ndarray:
        x_full = self._p4.transform(rows)
        return x_full[:, self._keep_mask].astype(np.float32)
