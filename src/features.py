"""Pluggable feature engineering recipes and scikit-learn preprocessor helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
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


# Backward-compatible ColumnTransformer helpers kept for external notebooks
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


def build_full_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    binary_cols: Sequence[str] = (),
    scale_numeric: bool = True,
    impute_numeric: bool = True,
    max_categories: int = 120,
) -> ColumnTransformer:
    """Build ColumnTransformer combining numeric, binary indicator, and categorical features."""
    num_steps: list[tuple[str, object]] = []
    if impute_numeric:
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_transformer: object = Pipeline(num_steps) if num_steps else "passthrough"

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=max_categories,
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, object, list[str]]] = [
        ("num", num_transformer, list(numeric_cols)),
        ("cat", cat_transformer, list(categorical_cols)),
    ]
    if binary_cols:
        transformers.append(("bin", "passthrough", list(binary_cols)))

    return ColumnTransformer(transformers=transformers, remainder="drop")


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


def _extract_location_and_cat_matrix(rows: RowList) -> tuple[np.ndarray, list[dict[str, Any]]]:
    parsed_list = [parse_address(r.get(ADDRESS_COL)) for r in rows]
    cat_rows = [
        [
            p["city"],
            p["district"],
            p["ward"],
            normalize_vietnamese_text(r.get("Legal status")) or "Missing",
            normalize_vietnamese_text(r.get("Furniture state")) or "Missing",
            normalize_vietnamese_text(r.get("House direction")) or "Missing",
        ]
        for r, p in zip(rows, parsed_list)
    ]
    return np.array(cat_rows, dtype=object), parsed_list


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


class FeatureRecipe:
    """Base interface for strictly train-fit tabular feature recipes."""

    name: str = "base"

    def __init__(self, scale_numeric: bool = True, preserve_nans: bool = False) -> None:
        self.scale_numeric = scale_numeric
        self.preserve_nans = preserve_nans
        self.feature_names: list[str] = []

    def fit_transform(self, train_rows: RowList, y_train: np.ndarray | None = None) -> np.ndarray:
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

    def fit_transform(self, train_rows: RowList, y_train: np.ndarray | None = None) -> np.ndarray:
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

    def fit_transform(self, train_rows: RowList, y_train: np.ndarray | None = None) -> np.ndarray:
        x_num = self._num_recipe.fit_transform(train_rows, y_train)
        cat_mat, _ = _extract_location_and_cat_matrix(train_rows)

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
        cat_mat, _ = _extract_location_and_cat_matrix(rows)
        x_cat = self._ohe.transform(self._clip_categories(cat_mat))
        return np.hstack([x_num, x_cat]).astype(np.float32)


@register_features("P2_domain")
class P2DomainRecipe(FeatureRecipe):
    """P1 + district-grouped median imputation + missing flags + address keywords + structural ratios."""

    name = "P2_domain"

    def __init__(self, scale_numeric: bool = True, preserve_nans: bool = False) -> None:
        super().__init__(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._p1_ohe_only = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
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

        # Binary missingness & address type flags (never scaled)
        flags = np.column_stack(
            [
                np.isnan(raw_num[:, 1]).astype(np.float32),  # Frontage_missing
                np.isnan(raw_num[:, 2]).astype(np.float32),  # AccessRoad_missing
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

        # Grouped median imputation for Frontage (col 1) & Access Road (col 2)
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
                area_per_floor,
                total_rooms,
                area_per_bedroom,
                frontage_road_ratio,
            ]
        ).astype(np.float32)

        if not self.preserve_nans:
            # Ensure derived ratios carry zero NaNs if inputs were valid
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
        return (
            self._p1_ohe_only.fit_transform(clipped)
            if fit
            else self._p1_ohe_only.transform(clipped)
        )

    def fit_transform(self, train_rows: RowList, y_train: np.ndarray | None = None) -> np.ndarray:
        cat_mat, parsed_list = _extract_location_and_cat_matrix(train_rows)
        cont, flags = self._compute_numeric_and_flags(train_rows, parsed_list, fit=True)
        x_cat = self._prep_cats(cat_mat, fit=True)

        cont_names = list(RAW_NUM_COLS) + [
            "log1p_Area",
            "Area_per_Floor",
            "Total_Rooms",
            "Area_per_Bedroom",
            "Frontage_Road_Ratio",
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
            + list(self._p1_ohe_only.get_feature_names_out(P1LocationRecipe._CAT_NAMES))
        )
        return np.hstack([cont, flags, x_cat]).astype(np.float32)

    def transform(self, rows: RowList) -> np.ndarray:
        cat_mat, parsed_list = _extract_location_and_cat_matrix(rows)
        cont, flags = self._compute_numeric_and_flags(rows, parsed_list, fit=False)
        x_cat = self._prep_cats(cat_mat, fit=False)
        return np.hstack([cont, flags, x_cat]).astype(np.float32)


@register_features("P3_tfidf_te")
class P3TfidfTargetEncRecipe(FeatureRecipe):
    """P2 + 5-fold out-of-fold smoothed Target Encoding (district/ward) + TF-IDF SVD(20) on Address."""

    name = "P3_tfidf_te"

    def __init__(
        self,
        scale_numeric: bool = True,
        preserve_nans: bool = False,
        svd_components: int = 20,
        te_smoothing: float = 10.0,
    ) -> None:
        super().__init__(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self._p2 = P2DomainRecipe(scale_numeric=scale_numeric, preserve_nans=preserve_nans)
        self.svd_components = svd_components
        self.te_smoothing = te_smoothing
        self._n_classes = 0
        self._priors: np.ndarray | None = None
        self._te_maps: dict[str, dict[str, np.ndarray]] = {}
        self._tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
        self._svd = TruncatedSVD(n_components=svd_components, random_state=RANDOM_SEED)
        self._extra_scaler = StandardScaler()

    @staticmethod
    def _compute_te_map(
        keys: Sequence[str],
        y: np.ndarray,
        n_classes: int,
        priors: np.ndarray,
        smoothing: float,
    ) -> dict[str, np.ndarray]:
        counts: dict[str, np.ndarray] = {}
        for key, cls_id in zip(keys, y):
            vec = counts.get(key)
            if vec is None:
                vec = np.zeros(n_classes, dtype=np.float32)
                counts[key] = vec
            vec[int(cls_id)] += 1.0

        te_map: dict[str, np.ndarray] = {}
        for key, vec in counts.items():
            total = float(vec.sum())
            te_map[key] = ((vec + smoothing * priors) / (total + smoothing)).astype(np.float32)
        return te_map

    @staticmethod
    def _apply_te_map(
        keys: Sequence[str],
        te_map: dict[str, np.ndarray],
        priors: np.ndarray,
    ) -> np.ndarray:
        return np.vstack([te_map.get(key, priors) for key in keys]).astype(np.float32)

    def fit_transform(self, train_rows: RowList, y_train: np.ndarray | None = None) -> np.ndarray:
        if y_train is None:
            raise ValueError("P3_tfidf_te requires y_train during fit_transform() for OOF Target Encoding.")

        y_arr = np.asarray(y_train, dtype=int)
        self._n_classes = int(y_arr.max()) + 1
        class_counts = np.bincount(y_arr, minlength=self._n_classes).astype(np.float32)
        self._priors = class_counts / max(float(class_counts.sum()), 1.0)

        x_p2 = self._p2.fit_transform(train_rows, y_arr)
        parsed = [parse_address(r.get(ADDRESS_COL)) for r in train_rows]
        districts = [p["district"] for p in parsed]
        wards = [f"{p['ward']} | {p['district']}" for p in parsed]

        # Strict 5-fold out-of-fold Target Encoding on train rows (prevents target leakage)
        oof_dist = np.zeros((len(train_rows), self._n_classes), dtype=np.float32)
        oof_ward = np.zeros((len(train_rows), self._n_classes), dtype=np.float32)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for tr_i, oof_i in skf.split(np.zeros(len(train_rows)), y_arr):
            tr_priors = np.bincount(y_arr[tr_i], minlength=self._n_classes).astype(np.float32)
            tr_priors /= max(float(tr_priors.sum()), 1.0)

            d_map = self._compute_te_map(
                [districts[i] for i in tr_i], y_arr[tr_i], self._n_classes, tr_priors, self.te_smoothing
            )
            w_map = self._compute_te_map(
                [wards[i] for i in tr_i], y_arr[tr_i], self._n_classes, tr_priors, self.te_smoothing
            )
            oof_dist[oof_i] = self._apply_te_map([districts[i] for i in oof_i], d_map, tr_priors)
            oof_ward[oof_i] = self._apply_te_map([wards[i] for i in oof_i], w_map, tr_priors)

        # Fit full-train TE maps for val/test inference
        self._te_maps = {
            "district": self._compute_te_map(
                districts, y_arr, self._n_classes, self._priors, self.te_smoothing
            ),
            "ward": self._compute_te_map(
                wards, y_arr, self._n_classes, self._priors, self.te_smoothing
            ),
        }

        addr_texts = [normalize_vietnamese_text(r.get(ADDRESS_COL)) for r in train_rows]
        tfidf_mat = self._tfidf.fit_transform(addr_texts)
        svd_mat = self._svd.fit_transform(tfidf_mat).astype(np.float32)

        extra = np.hstack([oof_dist, oof_ward, svd_mat]).astype(np.float32)
        if self.scale_numeric:
            extra = self._extra_scaler.fit_transform(extra).astype(np.float32)

        te_names = [f"te_district_c{c}" for c in range(self._n_classes)] + [
            f"te_ward_c{c}" for c in range(self._n_classes)
        ]
        svd_names = [f"addr_svd_{i}" for i in range(self.svd_components)]
        self.feature_names = list(self._p2.feature_names) + te_names + svd_names
        return np.hstack([x_p2, extra]).astype(np.float32)

    def transform(self, rows: RowList) -> np.ndarray:
        assert self._priors is not None
        x_p2 = self._p2.transform(rows)
        parsed = [parse_address(r.get(ADDRESS_COL)) for r in rows]
        districts = [p["district"] for p in parsed]
        wards = [f"{p['ward']} | {p['district']}" for p in parsed]

        te_dist = self._apply_te_map(districts, self._te_maps["district"], self._priors)
        te_ward = self._apply_te_map(wards, self._te_maps["ward"], self._priors)

        addr_texts = [normalize_vietnamese_text(r.get(ADDRESS_COL)) for r in rows]
        svd_mat = self._svd.transform(self._tfidf.transform(addr_texts)).astype(np.float32)

        extra = np.hstack([te_dist, te_ward, svd_mat]).astype(np.float32)
        if self.scale_numeric:
            extra = self._extra_scaler.transform(extra).astype(np.float32)
        return np.hstack([x_p2, extra]).astype(np.float32)
