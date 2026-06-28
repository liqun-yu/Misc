# DbD Online-vs-Offline — Root-Cause Investigation

**Date:** 2026-06-24
**Model:** `dbd_ranker_v0_1_metaflow_1781132405` (predictor `doubledash_store_recommendation_ranking`)

**The model is fine. None of the four bugs found are model-side.** Online AUC ~0.64 vs offline ~0.89 on the same rows + labels is explained by four parallel failures in the *serving* of features to the model. All numbers below are from production prediction-log SQL + AIMS catalog + feed-service source code. Today's date 2026-06-24.

---

## Entity shorthand

| short | meaning |
|---|---|
| `cs` | `consumer_id` — the requesting user (one per request) |
| `b` | `business_id` — a DoorDash business (parent of one or more stores). Feed-service emits *two* of these per request: `seqIndex=0=O1` (cart-anchor business) and `seqIndex=1=O2` (the candidate being scored) |
| `st` | `store_id` — a single store. Feed-service emits *two* per request, same O1/O2 pattern as `b` |
| `sm` | `submarket_id` — geographic submarket |
| `bizv` | `business_vertical_id` — the candidate's vertical (restaurant, grocery, etc.) |
| `o1b` / `o2b` | explicit `o1_business_id` / `o2_business_id` — single-emission, no ambiguity |
| `o2bizv` | `o2_business_vertical_id` — single-emission |

The O1/O2 emission is the engine of Bug 4 below.

---

## TL;DR — feature accounting (180 model inputs)

The model's `feature_config.py` declares **180 features**: 8 in `QUERY_FEATURES` and 172 in `CANDIDATE_FEATURES`. (Plus a small inference-time `list_features` registry that the caller populates; the 12 Bug-1 features below partially overlap with the QUERY block.)

| bug | what's broken | count | confidence |
|---|---|---:|---|
| 1 | **Feed-service request-payload features all defaulted** — both OLD and NEW paths emit the same entity ids + categorical data, but the OLD path emits it as `CategoricalFeature`/`NumericalFeature` with `*_cai`/`*_cas` names; the metaflow DNN's `list_features` slots only read from `ListFeature`s with specific names that ONLY the NEW path supplies. The DV gate `useNewFeatureSet` doesn't fire in prod → OLD path runs → 12 `ListFeature` slots fall through to defaults. Even if the gate fires, 3 have an `_int` name mismatch and 5 are never emitted in either path. | 12 | high (prediction-log SQL) |
| 2 | **No online materialization for the 5 `cos_sim_*` features** — only computed inline by a `pandas_udf` in v6's training pipeline. Zero Cassandra/Sibyl hits. Defaulted to 0.0 since model first served on 2026-06-11. | 5 | high (exhaustive cross-repo search) |
| 3 | **18 Fabricator features always defaulted** — splits three ways after per-feature source-coverage SQL (§3): **6-7 features with healthy source (≥80% coverage) → Sibyl/Cassandra-side bug**; **10 features with low source coverage (<55%) → source-pipeline/Fabricator under-population**; **1 feature with 0% (`offers_store_promo_quality`) — known stopped pipeline**. | 18 | high (per-feature source-coverage SQL) |
| 4 | **Sibyl resolves `business_id`- and `store_id`-keyed features to O1 (cart-anchor) instead of O2 (candidate).** Feed-service emits both ids per request (`seqIndex=0=O1`, `seqIndex=1=O2`). AIMS declares the entity with no `seqIndex` hint. **Empirically Sibyl returns the O1-keyed value** (98.8% / 98.6% measured). New DNN expects these as CANDIDATE features (O2). | **up to 110** (33 `*_cs_b_*` + 39 `baf_b_*` + 38 store-keyed `*_cs_st_*` / `*_st_*`) | high for `b`-keyed: per-feature value test 9.2-113× O1:O2; universal entity test 98.8% O1. High for `st`-keyed mechanism (98.6% O1 universal); per-feature value test for st-keyed features not yet done. The mechanism of *why* Sibyl picks O1 (default-to-seqIndex=0, vs. some other code path) is inferred from data, not from reading Sibyl source. |
| — | **Healthy / no known bug** — features keyed only on consumer, or with explicit `o1_*`/`o2_*` entity names (`baf_o1b_o2b_*`, `*_cs_o2b_*`, `baf_o2b_sm_*`) | ~18 | inferred from entity declarations + feed-service code |
| — | **Unverified** — `*_cs_bizv_*` (18 features) and a handful of singletons (`saf_cs_*`, `saf_dprg_st_*`). Need entity-binding spot-check (does feed-service emit `business_vertical_id` twice?). | ~21 | open |

Numbers above are mutually exclusive after one overlap: `caf_cs_b_p28d_consumer_business_view_recency` is in both Bug 3 (always-defaulted in Sibyl — confirmed at 100% by §4 default-audit) AND in the 33 `*_cs_b_*` family. Net unique broken inputs ≈ 12 + 5 + 18 + 110 − 1 = **144 of 180**.

---

## §0 — Production model context

The Sibyl predictor descriptor at [`doordash/columbus:config/prod/predictors/doubledash_store_recommendation_ranking.json`](https://github.com/doordash/columbus/blob/master/config/prod/predictors/doubledash_store_recommendation_ranking.json):

```json
{
  "default_model_id": "dbd_sr_v5_1_1",
  "shadow_models": [
    {"id": "dbd_sr_v5_1_1",                  "shadow_fraction": 7.4e-9},
    {"id": "dbd_sr_v6_3_1752102214",         "shadow_fraction": 0.2},
    {"id": "dbd_ranker_v0_1_metaflow_1781132405",  "shadow_fraction": 0.1, "shadow_expire_date": "2026-06-15"}
  ]
}
```

- Production traffic still goes to legacy LGBM `dbd_sr_v5_1_1`; the metaflow DNN runs as a 10% shadow.
- `shadow_expire_date: 2026-06-15` is 9 days in the past — worth chasing why it's still running.
- The descriptor has **no `featureSets` field**. Per-feature entity binding is determined by the caller (feed-service) at runtime, not the predictor descriptor. This is the structural origin of Bug 4.

Prediction-log volume confirms the DNN has been serving every day since 2026-06-11 at 250-330M predictions/day.

---

## §1 — Bug 1: feed-service request-payload features all defaulted in production

The 12 inputs the caller must populate, broken out by role:

| sub-group | features | role |
|---|---|---|
| O1 business ids (cart-anchor) | `o1_business_id_int`, `o1_business_vertical_id`, `o1_primary_category_id`, `o1_is_caviar` | identifies cart-anchor |
| O2 business ids (candidate) | `o2_business_id_int`, `o2_business_vertical_id`, `o2_primary_category_id`, `o2_is_caviar` | identifies candidate |
| Request context | `submarket_id`, `hour_of_day`, `day_of_week`, `is_promoted` | request scalars |

In `BundleMLFeatureGenerator.kt:38-46`:
```kotlin
val ddModelOverrideId = doubleDashFeatureGenConfig?.modelOverrideId   // DV doubledash_store_ranking_config
val useNewFeatureSet = ddModelOverrideId != null &&
    postCheckoutModelId != null &&                                    // DV bundle_ranking_config_postcheckout
    ddModelOverrideId == postCheckoutModelId
return if (useNewFeatureSet) _generateStoreRankingFeatureSetNew(...)
       else                  _generateStoreRankingFeatureSetOld(...)
```

**Important nuance** — both OLD and NEW paths emit *the same entity IDs and the same underlying data* (primary/candidate `business_id`, `business_vertical_id`, `store_id` each at `seqIndex=0/1`, plus explicit `o1_business_id` / `o2_business_id` tags, `consumer_id`, `submarket_id`, `day_part_regroup`, `o1_order_uuid`, etc.). The two paths diverge in *how* the categorical/context data is shaped for downstream lookups:

| feature | OLD path emits as | NEW path emits as |
|---|---|---|
| business_id | `CategoricalFeature` (entity-id only — no list version) | `ListFeature` named **`o1_business_id_int`** / **`o2_business_id_int`** with Int64 values |
| business_vertical_id | `CategoricalFeature` named `business_vertical_id_cai` (one per seqIndex) | `ListFeature` named **`o1_business_vertical_id_int`** / **`o2_business_vertical_id_int`** |
| submarket_id | `CategoricalFeature` (entity-id only) | `ListFeature` named **`submarket_id_int`** |
| hour_of_day / day_of_week | `CategoricalFeature` named `hour_of_day_cai` / `day_of_week_cai` | `ListFeature` named **`hour_of_day`** / **`day_of_week`** (post PR #65387) |
| subtotal / ETA | `NumericalFeature` | `ListFeature` (Int64) |

**The metaflow DNN's `list_features` registry reads from `ListFeature` slots with specific names.** The OLD path emits the same data as `CategoricalFeature`/`NumericalFeature` with different names (`*_cai`, `*_cas`) — so the model's `ListFeature` slots stay empty and fall through to defaults. **The gate `useNewFeatureSet` is not firing in production today, so the OLD path runs for the metaflow DNN → all 12 ListFeature slots are 100% defaulted.**

### Per-feature breakdown of the 12 (under the NEW path, if/when the gate fires)

The 12 categorical / context inputs the model's `list_features` registry expects, cross-referenced against the NEW path's `addListFeatures` calls in `_generateStoreRankingFeatureSetNew(...)`:

#### ✅ 4 features emitted under the correct name

| model expects | code that emits it | resolved name |
|---|---|---|
| `o1_business_id_int` | `buildLongListFeature("${SIBYL_PREDICTOR_PRIMARY_BUSINESS_ID}_int", primaryBusinessId.toLong())` (BundleMLFG.kt:309) | `"o1_business_id_int"` ✓ |
| `o2_business_id_int` | `buildLongListFeature("${SIBYL_PREDICTOR_O2_BUSINESS_ID}_int", candidateBusinessId.toLong())` (BundleMLFG.kt:312) | `"o2_business_id_int"` ✓ |
| `hour_of_day` | `buildLongListFeature(RealTimeFeatures.HOUR_OF_DAY.label, ...)` (BundleMLFG.kt:382). `.label == "hour_of_day"` post [PR #65387](https://github.com/doordash/feed-service/pull/65387) | `"hour_of_day"` ✓ |
| `day_of_week` | `buildLongListFeature(RealTimeFeatures.DAY_OF_WEEK.label, ...)` (BundleMLFG.kt:377). `.label == "day_of_week"` post PR #65387 | `"day_of_week"` ✓ |

#### ❌ 3 features emitted with an extra `_int` suffix the model doesn't want

The constant string already encodes the right canonical name (e.g. `SIBYL_PREDICTOR_PRIMARY_BUSINESS_VERTICAL_ID = "o1_business_vertical_id"` in Constants.kt), but the NEW-path code appends `"_int"` when emitting as a ListFeature — so the resolved name diverges from what the model reads.

| model expects | code that emits it | resolved name (mismatch) |
|---|---|---|
| `o1_business_vertical_id` | `buildLongListFeature("${SIBYL_PREDICTOR_PRIMARY_BUSINESS_VERTICAL_ID}_int", ...)` (BundleMLFG.kt:315-319) | `"o1_business_vertical_id_int"` ❌ |
| `o2_business_vertical_id` | `buildLongListFeature("${SIBYL_PREDICTOR_O2_BUSINESS_VERTICAL_ID}_int", ...)` (BundleMLFG.kt:320-325) | `"o2_business_vertical_id_int"` ❌ |
| `submarket_id` | `buildLongListFeature("${SIBYL_PREDICTOR_FEATURE_SUBMARKET_ID}_int", submarketId.toLong())` (BundleMLFG.kt:329) | `"submarket_id_int"` ❌ |

Fix: drop the `"_int"` suffix for these three (one-line change in the NEW path).

#### ❌ 5 features never emitted as ListFeature in either path

Verified by `grep -c '"<name>"\|${...<name>}'` returning 0 across both `_generateStoreRankingFeatureSetOld(...)` and `_generateStoreRankingFeatureSetNew(...)`:

| model expects | status | what code does today |
|---|---|---|
| `o1_primary_category_id` | not emitted | exists as a field on `BundleStoresSearchContext` / `BundleSearchStore`, but never wired into the Sibyl `ListFeature` payload |
| `o2_primary_category_id` | not emitted | same — field exists, never plumbed through |
| `o1_is_caviar` | not emitted | `Boolean` field on context, not used by feed-service feature generator |
| `o2_is_caviar` | not emitted | same — `Boolean` on `BundleSearchStore`, never plumbed |
| `is_promoted` | not emitted | exists as `Int` field in `base_instance_v2` schema; never read by `BundleMLFeatureGenerator` |

Fix: add 5 new `addListFeatures(buildLongListFeature("<name>", <field>.toLong()))` calls in the NEW path. All 5 source values are already available in `BundleSearchStore` / `BundleStoresSearchContext`.

**Total**: 4 OK + 3 name-mismatch + 5 never-emitted = 12 ListFeature slots the model needs. Today (OLD path) all 12 default; even after the gate fires (NEW path), 8 of 12 are still broken.

### Verified by SQL — 100% defaulting on the metaflow DNN only

```sql
SELECT model_id, COUNT(*) AS n,
  SUM(IF(default_values_used['hour_of_day']        IS NOT NULL, 1, 0)) AS hod_default,
  SUM(IF(default_values_used['o1_business_id_int'] IS NOT NULL, 1, 0)) AS o1biz_default
FROM datalake.sps_prediction_logging.search
WHERE predictor_name='doubledash_store_recommendation_ranking'
  AND prediction_ts_hour='2026-06-22-12'
GROUP BY model_id;
```

| model_id | rows | hour_of_day default | o1_business_id_int default |
|---|---:|---:|---:|
| `dbd_sr_v6_1_1` (LGBM) | 15,831,249 | 0 | 0 |
| `dbd_sr_v6_3_1752102214` (LGBM) | 3,298,498 | 0 | 0 |
| **`dbd_ranker_v0_1_metaflow_1781132405`** (DNN) | **1,647,975** | **1,647,975 (100%)** | **1,647,975 (100%)** |
| `dbd_sr_v5_1_1` (LGBM, default) | 655,165 | 0 | 0 |

LGBM peers at 0% defaulting in the same hour. Only the metaflow DNN sees 100% — gate is rejecting traffic only for this model.

### Likely root cause + actions

DV [`doubledash_store_ranking_config`](https://ops.doordash.team/decision-systems/dynamic-values-v2/experiments/147f2680-3fd6-454c-bef7-12f91125d3de) (id `147f2680-...`) holds `modelOverrideId`. Gate compares it against DV [`bundle_ranking_config_postcheckout`](https://ops.doordash.team/decision-systems/dynamic-values-v2/experiments/4ba4fe5e-5bdd-4302-a1f9-2d983b29d394). Both must be non-null and equal. The first DV's `modelOverrideId` doesn't surface in the ops UI → likely null.

Actions: (1) Read both DVs via `dv` CLI / `dynamic-values-service.service.prod.ddsd` gRPC and confirm what `modelOverrideId` resolves to. (2) Follow-up PR in feed-service to drop the `_int` suffix on 3 features and emit the missing 5. (3) Add a metric on `useNewFeatureSet` — there's none today.

---

## §2 — Bug 2: 5 cos_sim features have no online materialization

`cos_sim_copurchase_cx_o2`, `cos_sim_cuisine_tag_cx_o2`, `cos_sim_food_labels_cx_o2`, `cos_sim_food_labels_o1_o2`, `cos_sim_saf_st_cuisine_tag_o1_o2` are computed inline by a `pandas_udf` in [`v6.py:1265-1289`](https://github.com/doordash/fabricator/blob/b373252f4377b57c740526087021ca792d5b3db2/fabricator/repository/features/doubledash/store_ranking/training_instance/doubledash_store_ranking_full_training_instance_v6.py#L1265):

```python
@F.pandas_udf(T.DoubleType())
def cos_sim(a, b):
    return pd.Series([np.dot(a[i], b[i]) / (np.linalg.norm(a[i]) * np.linalg.norm(b[i])) for i in range(len(a))])
```

Cross-repo `gh search code` over `doordash/fabricator`, `doordash/sibyl-models`, `doordash/dd-models`, `doordash/feed-service`, `doordash/lucent-training` returns zero hits in any `feature_groups` / `materialize_spec` / `sink` / `online_store` / Cassandra block. The `nvml_job_cluster_example` config tags them `'derived': True` (precomputed in training data, never registered standalone). Online value has been 0.0 since 2026-06-11.

### Three fix options

| option | how | trade-off |
|---|---|---|
| 1. Compute in feed-service | Add inline Kotlin compute; pass 5 scalars as ListFeatures | Re-introduces the train/serve skew class of bug we're investigating |
| 2. Compute in model `forward()` | Feed the 6 underlying embeddings (`saf_st_cuisine_tag_emb`, etc.) as model inputs; PyTorch computes cos_sim | One source of truth; payload grows by 6 vectors per candidate. Prereq: confirm all 6 are served online |
| 3. Sibyl `derived: true` | Register cos_sim in Fabricator/AIMS as derived from the 6 underlying features | Cleanest IF Sibyl's derived-feature DSL supports cross-entity `dot/(norm*norm)` on `LIST<DOUBLE>` — needs a precedent check |

Recommendation: option 2 first; option 3 if Sibyl derived-feature support is mature enough.

Before any of the above, run feature-importance / SHAP on a hold-out batch — if cos_sim contribution is small, drop the features and skip the engineering.

---

## §3 — Bug 3: 18 Fabricator features 100%-defaulted in Sibyl

All 18 are flagged as defaulted in the prediction log AND have valid AIMS catalog metadata (`isActive: true`, `sourceName`, `tableName`, `entities`). For each, the per-feature source-coverage SQL below tells us whether the gap is on the source side (Fabricator under-population) or the Sibyl side (data exists in Iceberg but Sibyl serves the default anyway).

### Method — join each feature's source table on the entity Sibyl would have looked up

For each feature, the source-coverage question is: at the entity tuple Sibyl would have queried, does the source table at D-2 have a row, and is the feature value non-default? Sibyl's `business_id` and `store_id` lookups resolve to O1 per §4, and those O1 values are exactly what the prediction log stamps in its top-level `business_id` / `store_id` columns. So:

- **8 of 18 features** have entity keys entirely in the prediction log (`cs`, `business_id`, `store_id`) → join source directly to pred, denominator = `n_pred = 99,056`. No intermediate table.
- **7 of 18 features** need `submarket_id` or `business_vertical_id`. We derive both from dimension tables keyed on pred columns:
  - `submarket_id` ← `dimension_store(pred.store_id).SUBMARKET_ID` — 99.99% coverage on pred.
  - `business_vertical_id` ← `dimension_business(pred.business_id).BUSINESS_VERTICAL_ID` — 81.8% coverage on pred. (Tried `dimension_store(.store_id).BUSINESS_VERTICAL_ID` first — only 11.9% populated; switched to `dimension_business`.)

  Rationale: `pred.business_id` ≈ O1 (§4 Layer 3, ~99%); business_id → bizv is essentially 1:1; per §4 Sibyl resolves bizv to `seqIndex=0` = O1's bizv. So `dim_business(pred.business_id).bizv` IS the bizv Sibyl actually used. Same denominator (99,056) as the pred-only path.
- **3 of 18 features** (the `baf_o1b_o2bizv_*` cluster) are keyed on O2's `business_vertical_id`. `pred` only stamps O1, so O2's bizv is not derivable from pred alone — these 3 use an auction-context table (`base_instances_v4`) to pick up O2's bizv, with denominator `n_v4_matched = 20,384`, and are flagged separately.

In addition, the **training-time coverage** for each feature is read from `datalake.ml_datasets.doubledash_store_ranking_full_training_instance_v6` (the table `v6.py` writes for the model trainer) at `active_date = 2026-05-08` — most recent partition before the model's training cut, 5,855,362 rows × 2 array slots per row (the v6 join stores each scalar feature as `array<T>` length 2 = `[O1_value, O2_value]`).

**List-feature caveat applied:** the only list-type feature in this set is row 6 (`*_int32_sequence_cutoff_*_i64l`, `array<int>`). For it, the serving-time SQL (§C.1) and the training-time source-table query both check `SIZE(...) > 0` rather than just `IS NOT NULL`. The other 17 are `FloatFeature(null_default=0)` scalars — the v6 join always fills missing entries with 0 (so 100% are technically "non-null"); the model-relevant statistic is the **non-zero element rate**, which is what's reported as "train non-zero %" below. Per the prediction-log default audit at the head of §3, **serving non-zero rate = 0% for all 18** (Sibyl returns the default on every request); so **train/serve skew = train non-zero rate** for each feature, which is the magnitude of the AUC degradation Bug 3 imposes per-feature.

**SQL skeleton — pred-only entities** (e.g. `caf_st_p12w_five_star_rate_upper_bound`, key `(st)`):

```sql
WITH pred AS (
  SELECT consumer_id AS cs, business_id AS biz, store_id AS st, request_id
  FROM datalake.sps_prediction_logging.search
  WHERE predictor_name='doubledash_store_recommendation_ranking'
    AND model_id='dbd_ranker_v0_1_metaflow_1781132405'
    AND prediction_ts_hour BETWEEN '2026-06-11-00' AND '2026-06-11-23'
    AND pmod(hash(request_id), 2570) = 0
)
SELECT COUNT(*) AS n_total,
  SUM(IF(src.active_date IS NOT NULL, 1, 0)) AS n_src_row,
  SUM(IF(src.`caf_st_p12w_five_star_rate_upper_bound` IS NOT NULL, 1, 0)) AS n_feat_nonnull
FROM pred p
LEFT JOIN datalake.ml.store_property_features_v2 src
  ON CAST(src.store_id AS STRING) = p.st
 AND src.active_date = DATE '2026-06-09';
```

**SQL skeleton — dim-augmented entities** (e.g. `baf_b_sm_*`, key `(b, sm)` — `sm` from `dim_store`):

```sql
WITH pred AS (... same as above ...),
dim AS (
  SELECT CAST(store_id AS STRING) AS store_id_str,
         CAST(submarket_id AS BIGINT) AS sm_num
  FROM datalake.edw_merchant.dimension_store
),
joined AS (
  SELECT p.consumer_id AS cs, p.business_id AS biz, p.store_id AS st,
         CAST(d.sm_num AS STRING) AS sm
  FROM pred p
  LEFT JOIN dim d ON p.store_id = d.store_id_str
)
SELECT COUNT(*) AS n_total,
  SUM(IF(sm IS NOT NULL, 1, 0)) AS n_sm_derived,
  SUM(IF(src.active_date IS NOT NULL, 1, 0)) AS n_src_row,
  SUM(IF(src.`baf_b_sm_p84d_doubledash_engagement_num_imps_v2` IS NOT NULL, 1, 0)) AS n_feat_nonnull
FROM joined p
LEFT JOIN datalake.ml.doubledash_store_engagement_b_sm src
  ON CAST(src.business_id AS STRING)  = p.biz
 AND CAST(src.submarket_id AS STRING) = p.sm
 AND src.active_date = DATE '2026-06-09';
```

Full per-table SQL at `/tmp/bug3_no_v4/*.sql` (current path).

### Results: all 18 features (serving source coverage + training non-zero rate + skew)

Sample: ~99,056 raw pred rows from the full 2026-06-11 day. All rows share denominator = 99,056 (`pred` or `pred+dim_*`) except the 3 `o1b_o2bizv` rows which use `n_v4_matched = 20,384`. Serving rates below are conditional on the relevant denominator and isolate source-table data quality. The "train non-zero" column is the v6 element-level rate at 2026-05-08. The "skew" column = train non-zero − serving 0% (since Sibyl always defaults at serving).

| # | feature | source table | entity | serving src has row | serving feat non-null | train non-zero | skew | verdict (AUC impact) |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1  | `caf_st_p12w_five_star_rate_upper_bound`                          | `store_property_features_v2` | st | **100.0%** | 95.3% | **97.9%** | **97.9pp** | **Sibyl-side / major** |
| 2  | `baf_b_sm_p84d_doubledash_engagement_num_imps_v2`                 | `doubledash_store_engagement_b_sm` | b, sm† | **94.2%** | 94.2% | **98.5%** | **98.5pp** | **Sibyl-side / major** |
| 3  | `baf_b_sm_p84d_doubledash_engagement_num_loads_pos_gt1_v2`        | `doubledash_store_engagement_b_sm` | b, sm† | **94.2%** | 91.4% | **96.9%** | **96.9pp** | **Sibyl-side / major** |
| 4  | `baf_b_sm_p84d_doubledash_engagement_num_bundled_v2`              | `doubledash_store_engagement_b_sm` | b, sm† | **94.2%** | 94.2% | **96.5%** | **96.5pp** | **Sibyl-side / major** |
| 5  | `daf_st_p84d_doubledash_business_position_weighted_imp2conv_over_avg` | `doubledash_store_ranking_position_weighted_attach_rate` | st | **95.8%** | 95.8% | **97.4%** | **97.4pp** | **Sibyl-side / major** |
| 6  | `caf_cs_p84d_organic_converted_business_id_int32_sequence_cutoff_v2_i64l` | `organic_consumer_sequence_aggregated_features_cutoff_p13n_v1` | cs | **94.4%** | 91.4% | 60.0%‡ | 60.0pp | **Sibyl-side / moderate** (feeds user-history attention only) |
| 7  | `caf_cs_b_p28d_consumer_business_view_recency`                    | `consumer_business_recency` | cs, b | **80.0%** | 77.4% | 47.0% | 47.0pp | **Sibyl-side / moderate** |
| 8  | `baf_o1b_o2bizv_p84d_doubledash_engagement_num_imps_v2`           | `doubledash_store_engagement_o1b_o2bizv` | o1b, o2bizv§ | 55.3% | 55.3% | 27.4% | 27.4pp | mixed / small-moderate |
| 9  | `baf_o1b_o2bizv_p84d_doubledash_engagement_num_loads_pos_gt1_v2`  | `doubledash_store_engagement_o1b_o2bizv` | o1b, o2bizv§ | 55.3% | 55.3% | 26.5% | 26.5pp | mixed / small-moderate |
| 10 | `baf_o1b_o2bizv_p84d_doubledash_engagement_num_bundled_v2`        | `doubledash_store_engagement_o1b_o2bizv` | o1b, o2bizv§ | 55.3% | 55.3% | 25.3% | 25.3pp | mixed / small-moderate |
| 11 | `daf_cs_bizv_cng_vertical_num_orders`                             | `consumer_business_vertical_cng_business_line_trial` ⚠ "trial" | cs, bizv† | 9.5% | 9.5% | 17.7% | 17.7pp | **Bug-4 binding** (see §3.1) — bound to O1's bizv |
| 12 | `baf_bizv_cs_p84d_doubledash_engagement_num_imps_v2`              | `doubledash_store_engagement_bizv_cs` | bizv, cs† | 5.4% | 5.4% | 16.4% | 16.4pp | **Bug-4 binding** (see §3.1) |
| 13 | `baf_bizv_cs_p84d_doubledash_engagement_num_loads_pos_gt1_v2`     | `doubledash_store_engagement_bizv_cs` | bizv, cs† | 5.4% | 5.2% | 9.5% | 9.5pp | **Bug-4 binding** (see §3.1) |
| 14 | `baf_bizv_cs_p84d_doubledash_engagement_num_bundled_v2`           | `doubledash_store_engagement_bizv_cs` | bizv, cs† | 5.4% | 5.4% | 6.3% | 6.3pp | **Bug-4 binding** (see §3.1) |
| 15 | `baf_cs_b_p84d_doubledash_engagement_num_imps_v2`                 | `doubledash_store_engagement_cs_b` | cs, b | 9.0% | 9.0% | 28.1% | 28.1pp | **Bug-4 binding** (see §3.1) — has working `cs_o2b` twin |
| 16 | `baf_cs_b_p84d_doubledash_engagement_num_loads_pos_gt1_v2`        | `doubledash_store_engagement_cs_b` | cs, b | 9.0% | 8.9% | 16.7% | 16.7pp | **Bug-4 binding** (see §3.1) — has working `cs_o2b` twin |
| 17 | `baf_cs_b_p84d_doubledash_engagement_num_bundled_v2`              | `doubledash_store_engagement_cs_b` | cs, b | 9.0% | 9.0% | 9.0% | 9.0pp | **Bug-4 binding** (see §3.1) — has working `cs_o2b` twin |
| 18 | `daf_st_offers_store_promo_quality_score_v2_cai`                  | `offers_store_promo_quality_v2_store_features` | st | **0.0%** | 0.0% | **99.8%** | **99.8pp** | **pipeline stopped 2026-05-04 / major** (was working at training time, broke after) |

† `sm` from `dim_store(pred.store_id).SUBMARKET_ID` (99.99% derived); `bizv` from `dim_business(pred.business_id).BUSINESS_VERTICAL_ID` (81.8% derived). Both denominators stay at 99,056 (n_pred). **NOTE: these "coverage" numbers join the source on the entity Sibyl currently uses (O1), which is the bug itself — see §3.1. They do NOT mean the source is under-populated.**
‡ For row 6, the training-time number is the **non-empty rate of the sequence column at the source table** (`SIZE > 0` over all rows of `organic_consumer_sequence_aggregated_features_cutoff_p13n_v1` at 2026-05-08); the v6 training table doesn't carry this column directly.
§ Not derivable from pred alone — O2's `business_vertical_id` is not stamped anywhere in pred (only O1 is, per §4 Layer 3). Serving rate uses an auction-context join (`base_instances_v4`) with `n_v4_matched = 20,384` as denominator.

### §3.1 — CORRECTION: rows 11-17 are Bug-4 entity-binding, NOT source under-population

The earlier "source-side" verdict for rows 11-17 was an **artifact of the analysis method**, not a real source-population problem. Verified against the Fabricator Spark jobs and the source tables directly:

- **The `cs_b` / `bizv_cs` source tables are fully populated.** `doubledash_store_engagement_cs_b.py` and `doubledash_store_engagement_cs_o2b.py` are byte-for-byte identical computations (both `grouping_keys=["consumer_id","o2_business_id"]`); they differ ONLY in an `entity_renames` map that renames the key column to `business_id` (cs_b) vs keeps `o2_business_id` (cs_o2b). Direct join of the two tables at active_date 2026-06-09: **64,612,490 rows, 100% value-identical, identical row counts.** So `baf_cs_b_* ≡ baf_cs_o2b_*` exactly.
- **The 9% / 5.4% "coverage" is because §3's probe joined the source on `pred.business_id ≈ O1`, but the table is keyed by `o2_business_id` (the CANDIDATE).** Querying an O2-keyed table with O1 keys hits only when an anchor business was also a candidate for that consumer → ~9%. This is the *serving bug* (Sibyl resolves to O1) measured offline, not source quality.
- **Online, these features are not even materialized.** Prediction-log audit (2026-06-11, n=99,056): `baf_cs_b_*` nonzero **0%**, `baf_b_sm_*` nonzero **0%** — the `cs_b`/`b_sm` feature_groups have no `materialize_spec`, so they're never pushed online. Their O2-bound twins, which ARE materialized, serve fine: `baf_cs_o2b_*` 3.4% nonzero (genuine sparsity), `baf_o2b_sm_*` **99.9%** nonzero.

So rows 11-17 are the **same root cause as §4 (O1/O2 entity binding)**: the feature is bound to the generic `business_id`/`business_vertical_id` entity (→ Sibyl resolves O1) instead of `o2_business_id`/`o2_business_vertical_id` (→ O2). Fix = use the O2-bound entity (existing twin where present, new clone where absent) — see §4 fix. They are NOT under-populated and need NO source backfill of the existing tables.

### Clusters and AUC impact

**Big AUC contributors from Bug 3 (genuine non-materialization / Sibyl-side):**
- **Rows 1-5** — 97-98% training coverage, 0% serving. Source has the data offline but the feature_group lacks `materialize_spec`, so it's never published to the online store (CockroachDB `feature_store_prod_ug` / Boulder — NOT "Cassandra"). **Adding `materialize_spec.sink` recovers the signal.** Highest priority.
- **Row 18** — 99.8% training, 0% serving. The single pipeline-stopped feature. **Revive the upstream pipeline or drop the feature.**
- **Rows 6, 7** — 60%/47% training, 0% serving. Source has the data; not materialized. Second-tier.

**Rows 11-17** — Bug-4 entity-binding (see §3.1), fixed by the §4 O2-binding fix, not by a §3 materialize_spec on the generic source.

**Rows 8-10 (`baf_o1b_o2bizv_*`) update — serve 0% online, worse than the offline §3 number implies.** Prediction-log audit (2026-06-11, n=99,056): `baf_o1b_o2bizv_*` nonzero = **0%** (vs 32.7% in the training parquet). These bind to the `(o1_business_id, o2_business_vertical_id)` entity pair; the `o2_business_vertical_id` half is not served (not emitted/materialized) → full default at serving. Contrast the sibling `baf_o1b_o2b_*` (paired on `(o1_business_id, o2_business_id)`, both unambiguous entities feed-service emits): training 94.5%, **online 80.2%** — correctly bound and served. So the `o1b_o2b` design works; the `o1b_o2bizv` one is broken on the bizv entity and needs the same O2-bizv fix.

### Caveats on the train/serve skew analysis

1. **The v6 array `[O1_value, O2_value]` aggregate averages O1 and O2 coverage.** For CANDIDATE-side features (cs_b, bizv_cs, cs_bizv, o1b_o2bizv), the model consumes position-1 (O2). If O1 has higher source coverage than O2 (e.g., cart-anchor businesses are denser in the engagement source tables), the aggregate biases upward. The Bug 4 fix would route the model to the position the training data was indexed at — so the per-position rate at training matches the per-position rate at serving once Bug 4 is fixed.
2. **All "0% serving" reflects current Sibyl behavior (default returned).** After fixing Bug 3 (`materialize_spec` PRs for the Sibyl-side cluster), the serving rate jumps to ≈ source rate.
3. **`null_default=0` semantics**: the model can't distinguish "feature genuinely 0" from "feature defaulted". For count features, zero-or-defaulted is the same input. For rate/recency features, zero may carry meaning ("never engaged"), which biases the "non-zero" measure slightly. Practical impact is small.

### Likely root cause for the Sibyl-side cluster (6-7 features)

Two candidates:
1. **Cassandra mirror not being written for these specific feature names** — Fabricator's Iceberg-side sink succeeds (we see the data) but the Cassandra-side `materialize_spec` is missing, miswired, or running against a different keyspace.
2. **Sibyl runtime resolver not looking these features up** for the metaflow model's request — could be a registration step the model deployment missed.

Direct Cassandra spot-check is the unblocking step (`cqlsh` the Sibyl-online keyspace on one known-Iceberg-present `(cs, business_id, feature)` tuple). Out of session scope — needs Sibyl/Fabricator owner credentials.

### Side issues

- `daf_cs_bizv_cng_vertical_num_orders` source is `..._business_line_trial` — "trial" suffix is suspicious; confirm it actually publishes online or migrate to a non-trial pipeline.
- `daf_st_offers_store_promo_quality_score_v2_cai` upstream stopped 2026-05-04. AIMS still flags it `isActive: true` (stale). Revive or drop.

---

## §4 — Bug 4: O1/O2 entity-binding default — affects up to 110 served features

### What's being tested

**Hypothesis.** When feed-service emits two values for an entity (e.g. two `business_id`s — O1 at `seqIndex=0`, O2 at `seqIndex=1`), and the feature's AIMS declaration doesn't specify which `seqIndex` to use, Sibyl picks `seqIndex=0` = O1 (cart-anchor). The new DNN expects these features to be CANDIDATE-side (O2). Result: every candidate in a ranking request gets the cart-anchor's history → no candidate discrimination, AUC collapses.

### Three layers of evidence (the chain that nails it)

Each layer is independently checkable; together they constitute proof:

**Layer 1 — The model expects O2.** The model's `feature_config.py` has two lists, `QUERY_FEATURES` and `CANDIDATE_FEATURES`. The training-time code in `_get_feature_config_for_inference` (`feature_config.py:261-298`) assigns `FeatureSource.QUERY` to the former and `FeatureSource.CANDIDATE` to the latter — and the model's `forward()` uses these tags to route each feature into the query tower (consumer + cart-anchor side) vs the candidate tower (the store being scored, O2 side).

| tower | count | what's in it |
|---|---:|---|
| **QUERY** (model: cart-anchor side, expects O1) | 8 | 4 O1-explicit categorical IDs (`o1_business_id_int`, `o1_business_vertical_id`, `o1_primary_category_id`, `o1_is_caviar`), 3 request-context scalars (`submarket_id`, `hour_of_day`, `day_of_week`), 1 sequence (`caf_cs_p84d_organic_converted_business_id_int32_sequence_cutoff_v2_i64l`) |
| **CANDIDATE** (model: candidate side, expects O2) | 172 | every other feature, **including all 33 `*_cs_b_*`, all 39 `baf_b_*`, all 14 `*_cs_st_*`, all 24 `*_st_*`** that Bug 4 says are affected |

The 110 features I'm calling out for Bug 4 are precisely the subset of CANDIDATE features whose entity declaration includes a token (`business_id` or `store_id`) that feed-service emits in two `seqIndex` slots. The model expects O2; Sibyl can't tell it's supposed to be O2 from the AIMS catalog (no `seqIndex` hint), so it picks the default = O1.

**Layer 2 — `prediction_log.business_id` IS the entity Sibyl used to look up business-keyed features.** Direct test: for each business-keyed feature, join its source table on `(pred.consumer_id, pred.business_id)` at `active_date = D-2`, and ask: does `online_val == src.value` exactly? If the column were not what Sibyl used for the lookup, this match rate would be near 0% (chance level). Result for the 6 clean integer-typed `*_cs_b_*` features that have the cleanest match dynamics (full table in §A.5):

| feature | n_online | match @ (pred.cs, pred.biz) | rate |
|---|---:|---:|---:|
| `caf_cs_b_p168d_organic_clicks` | 99,056 | 78,318 | **79.1%** |
| `caf_cs_b_p84d_organic_conversions` | 99,056 | 77,809 | **78.6%** |
| `caf_cs_b_p84d_consumer_business_views` | 99,056 | 76,568 | **77.3%** |
| `caf_cs_b_p168d_consumer_business_views` | 99,056 | 75,834 | **76.6%** |
| `caf_cs_b_p28d_consumer_business_views` | 99,056 | 75,257 | **76.0%** |
| `caf_cs_b_p168d_consumer_business_clicks` | 99,056 | 73,257 | **74.0%** |

Aggregate across all 33 cs_b features × 99k samples: **1,339,598 / 3,268,848 = 41.0%** match at `(pred.cs, pred.biz)`. The 33-row spread (3-79%) is dominated by float-precision loss on ratio-typed features (`*_ctr`, `*_cvr`, `*_recency_discounted_*` → STRING→DOUBLE round-trip drops the last bit, dropping rate to 3-15%) — the integer-typed features cluster cleanly at 74-79%. Same shape for the 39 `baf_b_*` features keyed on `business_id` only (aggregate 14% match, integer top 28-29%, full table in §A.6).

The remaining 20-25% gap on integer features after pred.business_id IS the lookup entity is explained by: (a) source rows missing at D-2 for that (cs, biz) pair, (b) Cassandra freshness drift (Sibyl reading D-1 or D-3 on some requests rather than D-2), (c) defaulting on the row (the audit elsewhere shows 7-16% defaulting on most cs_b features). The key signal — that integer features match at 74-79% vs the ~0% chance level if `pred.business_id` were unrelated to the lookup — closes the question.

```sql
-- Loopback test (cs_b family example, against dimension_consumer_business_engagement_aggregated_metrics)
WITH pred AS (
  SELECT consumer_id AS cs, business_id AS biz, request_id,
    CAST(features['caf_cs_b_p84d_consumer_business_views'] AS DOUBLE) AS on_val
  FROM datalake.sps_prediction_logging.search
  WHERE predictor_name='doubledash_store_recommendation_ranking'
    AND model_id='dbd_ranker_v0_1_metaflow_1781132405'
    AND prediction_ts_hour BETWEEN '2026-06-11-00' AND '2026-06-11-23'
    AND business_id IS NOT NULL
    AND pmod(hash(request_id), 2570) = 0
)
SELECT COUNT(*) AS n_total,
  SUM(IF(on_val IS NOT NULL, 1, 0)) AS n_online,
  SUM(IF(on_val IS NOT NULL AND s_val IS NOT NULL AND on_val = s_val, 1, 0)) AS n_match
FROM pred p
LEFT JOIN (SELECT CAST(consumer_id AS STRING) AS cs, CAST(business_id AS STRING) AS biz,
                  `caf_cs_b_p84d_consumer_business_views` AS s_val
           FROM datalake.ml.dimension_consumer_business_engagement_aggregated_metrics
           WHERE active_date = DATE '2026-06-09') src
  ON src.cs = p.cs AND src.biz = p.biz;
```

**Layer 3 — That entity is O1's `business_id`, not O2's.** Now compare `pred.business_id` against the auction's O1 and O2 business_ids from `base_instance_v2`:

| entity tested | n_joined | n_used_O1 | n_used_O2 | pct_O1 | pct_O2 |
|---|---:|---:|---:|---:|---:|
| `business_id` | 96,645 | 95,550 | 52 | **98.9%** | 0.05% |
| `store_id` | 96,645 | 95,368 | 47 | **98.7%** | 0.05% |

Sample: 100k deterministic random rows from the full 2026-06-11 day (`pmod(hash(request_id), 2570) = 0`), INNER JOIN to `base_instance_v2` on consumer_id + 120s timestamp window. The ~99% / 98.7% O1 rates hold against the larger, day-wide sample.

```sql
-- Same query for both, with column substitution
WITH pred AS (
  SELECT consumer_id, business_id AS sibyl_used, prediction_ts, request_id  -- swap business_id ↔ store_id
  FROM datalake.sps_prediction_logging.search
  WHERE predictor_name='doubledash_store_recommendation_ranking'
    AND model_id='dbd_ranker_v0_1_metaflow_1781132405'
    AND prediction_ts_hour BETWEEN '2026-06-11-00' AND '2026-06-11-23'
    AND business_id IS NOT NULL                                              -- swap
    AND pmod(hash(request_id), 2570) = 0
),
v2 AS (
  SELECT CAST(consumer_id AS STRING) AS cs,
         CAST(CAST(o1_business_id AS BIGINT) AS STRING) AS o1,   -- swap o1/o2_business_id ↔ o1/o2_store_id
         CAST(CAST(o2_business_id AS BIGINT) AS STRING) AS o2,
         o2_timestamp
  FROM datalake.ml_datasets.doubledash_store_ranking_base_instance_v2
  WHERE active_date = DATE '2026-06-11'
)
SELECT COUNT(*) n_joined,
  SUM(IF(p.sibyl_used = v2.o1, 1, 0)) n_used_o1,
  SUM(IF(p.sibyl_used = v2.o2, 1, 0)) n_used_o2
FROM pred p JOIN v2
  ON p.consumer_id = v2.cs
 AND ABS(UNIX_TIMESTAMP(p.prediction_ts) - UNIX_TIMESTAMP(v2.o2_timestamp)) < 120;
```

The remaining few percent is within-auction Cartesian noise (multiple candidates per `(consumer, o2_timestamp)` in `base_instance_v2`). Signal is unambiguous: **Sibyl resolves both entities to O1 ~99% of the time.** The *mechanism* (whether Sibyl explicitly defaults to `seqIndex=0`, or has a different resolution rule that happens to pick O1, or hit some other code path) is not directly inspectable from this session's tooling — but the *outcome* is consistent across both entities at >96%.

**Conclusion of the three layers**: Layer 1 says the model expects O2. Layer 2 says `pred.business_id` is the entity Sibyl used for the lookup. Layer 3 says `pred.business_id = O1`. Therefore Sibyl looks up at O1, the model expects O2, and every CANDIDATE business-keyed feature gets the cart-anchor's history instead of the candidate's. Bug confirmed without smuggling in any unverified assumption about what `pred.business_id` means.

(Below: a separate per-feature "value matches O1 vs O2" comparison via v2 — historical / additional view. Layers 1+2+3 already settle the question.)

### Test methodology: exact equality, no normalization

All tests use `=`. No tolerances, no log-transform, no normalization. A row "matches src(cs, O1)" iff `prediction_log.features['<feat>'] CAST TO DOUBLE` is bit-equal to `datalake.ml.<src>.<feat>` at the joined entity tuple. Low match rates mean one of: (a) source didn't have a row for that (cs, biz) at the D-2 partition; (b) the online side serialized the value as STRING and the `CAST(... AS DOUBLE)` round-trip dropped the last bit (which is what hits the ratio-typed features `*_ctr`, `*_cvr`). The **ratio** `match_O1 / match_O2` is robust to both (a) and (b) because they hit O1 and O2 symmetrically.

### Per-feature value test — 33 `*_cs_b_*` features (6 SQL queries, one per source table)

Layer 3 of the proof above. For each feature, joins prediction_log → v2 (to get O1/O2 per request) → source table TWICE (once on `(cs, O1)`, once on `(cs, O2)`) and asks: does the online value match `src(cs, O1)` or `src(cs, O2)`? Six SQL queries, one per source table. Full per-feature SQL kept at `/tmp/per_feat_v3/*.sql`. The actual (un-truncated) query for the `dimension_consumer_business_engagement_aggregated_metrics` table (9 features in one query — repeated `on_fN` and `s_fN` per feature; full file is 132 lines):

```sql
WITH pred AS (
  SELECT consumer_id, business_id, prediction_ts, request_id,
    CAST(features['caf_cs_b_p168d_consumer_business_clicks'] AS DOUBLE) AS on_f1,
    CAST(features['caf_cs_b_p168d_consumer_business_views']  AS DOUBLE) AS on_f2,
    /* ... 7 more on_fN, one per feature in this source table ... */
    CAST(features['daf_cs_b_p84d_consumer_business_orders']  AS DOUBLE) AS on_f9
  FROM datalake.sps_prediction_logging.search
  WHERE predictor_name='doubledash_store_recommendation_ranking'
    AND model_id='dbd_ranker_v0_1_metaflow_1781132405'
    AND prediction_ts_hour BETWEEN '2026-06-11-00' AND '2026-06-11-23'
    AND business_id IS NOT NULL
    AND pmod(hash(request_id), 2570) = 0
),
v2 AS (
  SELECT CAST(consumer_id AS STRING) AS cs,
         CAST(CAST(o1_business_id AS BIGINT) AS STRING) AS o1_biz,
         CAST(CAST(o2_business_id AS BIGINT) AS STRING) AS o2_biz,
         o2_timestamp
  FROM datalake.ml_datasets.doubledash_store_ranking_base_instance_v2
  WHERE active_date = DATE '2026-06-11'
),
joined AS (
  SELECT p.*, v2.o1_biz, v2.o2_biz
  FROM pred p
  JOIN v2 ON p.consumer_id = v2.cs
         AND ABS(UNIX_TIMESTAMP(p.prediction_ts) - UNIX_TIMESTAMP(v2.o2_timestamp)) < 120
),
src AS (
  SELECT CAST(consumer_id AS STRING) AS cs,
         CAST(business_id AS STRING) AS biz,
    `caf_cs_b_p168d_consumer_business_clicks` AS s_f1,
    `caf_cs_b_p168d_consumer_business_views`  AS s_f2,
    /* ... 7 more s_fN ... */
    `daf_cs_b_p84d_consumer_business_orders`  AS s_f9
  FROM datalake.ml.dimension_consumer_business_engagement_aggregated_metrics
  WHERE active_date = DATE '2026-06-09'
),
final AS (
  SELECT j.*,
    s1.s_f1 AS o1_f1, s2.s_f1 AS o2_f1, s1.s_f2 AS o1_f2, s2.s_f2 AS o2_f2, /* ... s1.s_f9, s2.s_f9 ... */
  FROM joined j
  LEFT JOIN src s1 ON s1.cs = j.consumer_id AND s1.biz = j.o1_biz   -- (cs, O1) lookup
  LEFT JOIN src s2 ON s2.cs = j.consumer_id AND s2.biz = j.o2_biz   -- (cs, O2) lookup
)
SELECT COUNT(*) AS n_joined,
  SUM(IF(on_f1 IS NOT NULL, 1, 0))                                              AS f1_n_online,
  SUM(IF(o1_f1 IS NOT NULL, 1, 0))                                              AS f1_n_src_o1,
  SUM(IF(o2_f1 IS NOT NULL, 1, 0))                                              AS f1_n_src_o2,
  SUM(IF(on_f1 IS NOT NULL AND o1_f1 IS NOT NULL AND on_f1 = o1_f1, 1, 0))      AS f1_match_o1,
  SUM(IF(on_f1 IS NOT NULL AND o2_f1 IS NOT NULL AND on_f1 = o2_f1, 1, 0))      AS f1_match_o2,
  /* ... 5 more aggregates per feature ... */
FROM final
```

Aggregate across all 33 `*_cs_b_*` features × 96,645 joined rows = 3.19M observations:

| metric | count | rate |
|---|---:|---:|
| Online non-null observations (33 features × 96,645 joined rows) | 3,189,285 | — |
| src(cs, O1) has row | 1,965,493 | 61.6% |
| src(cs, O2) has row | 1,335,856 | 41.9% |
| **online == src(cs, O1)** | **1,400,374** | **43.9%** |
| **online == src(cs, O2)** | **152,939** | **4.8%** |
| **O1:O2 match ratio** | — | **9.2×** |

**Every one of the 33 features individually shows more O1-matches than O2-matches.** Per-feature O1:O2 ratio ranges 1.0× (`*_p28d_view_recency` — already 100%-defaulted per Bug 3, both src columns are often 0 → spurious matches) to >1000× (`*_recency_discounted_views`, near-zero O2 matches).

On the 45% absolute rate (not 100%): see `n_joined = 96,645` from a 100k pred sample → 97% of pred rows had a matching v2 auction in the 120s time window. v2 captures ~2% of all DbD predictions per day, but among the 100k random sample, most consumer-time pairs landed in v2's sample window when using a wide JOIN. The 45% is then `(src has row | v2 matched) × (online value matches | src has row)`. Per the audit and breakdown below, the **ratio** O1:O2 is what's meaningful and is robust to all the upstream losses.

Why the per-feature O1 rate is < 100% even though the binding is universally O1:
- **Source-table sparsity** at D-2: `src(cs, O1)` has a row in ~60-90% of joined cases.
- **Feature defaulted on a non-trivial fraction of rows** — see audit below.
- **Float-precision loss** on ratio-typed features (`*_ctr`, `*_cvr`, `*_discounted_*`). Online stores values as STRING in the `features` map → `CAST(... AS DOUBLE)` round-trip drops the last bit. Integer-typed features (`*_clicks`, `*_views`, `*_orders`) cluster cleanly at 75-86% O1-match; float ratios sit at 3-33% O1-match. O2 rates drop proportionally, preserving the ratio evidence.

Full 33-row table in §A.1. **Conclusion: 9.2× more O1-matches than O2-matches across the whole family.** Every feature individually beats O2 by 3-1000+×.

### Per-feature `default_values_used` audit (rules out the "feature is just defaulted" alternative explanation)

A natural objection to the O1 rates above: maybe the low O1-match rates for some features are because Sibyl is just *defaulting* those features, in which case the O1/O2 question is moot for them. Direct check: how often does each feature appear in `default_values_used` on the prediction-log row? Query:

```sql
SELECT COUNT(*) AS n_total,
  SUM(IF(ARRAY_CONTAINS(MAP_KEYS(default_values_used), '<FEATURE>'), 1, 0)) AS n_defaulted
FROM datalake.sps_prediction_logging.search
WHERE predictor_name='doubledash_store_recommendation_ranking'
  AND model_id='dbd_ranker_v0_1_metaflow_1781132405'
  AND prediction_ts_hour='2026-06-22-12'
LIMIT 100000
```

Run for all 72 features (33 cs_b + 39 baf_b) at once via 72 `SUM(IF(...))` columns. Sampled at `prediction_ts_hour='2026-06-22-12'` (audit was run before the §3/§4 day-wide v3 batch; per-feature defaulting rates are steady across days). n_total = 1,647,975 prediction-log rows. Result clusters:

| feature group | defaulting rate range | count |
|---|---|---:|
| **`caf_cs_b_p28d_consumer_business_view_recency`** (special case — also in Bug 3) | **100.0%** | 1 |
| `caf_cs_b_*` engagement (views, clicks, organic) | 6.9% - 16.2% | 14 |
| `caf_cs_b_*` recency-discounted | 14.9% | 5 |
| `caf_cs_b_p1y_order_recency` | 32.0% | 1 |
| `daf_cs_b_*` consumer-business orders / recency | 19.8% - 22.3% | 4 |
| `daf_cs_b_*` global_search | 15.2% - 48.0% | 10 |
| `baf_b_*` engagement (all variants) | 3.05% - 3.85% | 39 |

So among the 72 verified Bug-4 features, only **1** (`caf_cs_b_p28d_consumer_business_view_recency`) is 100%-defaulted — and that one is already in Bug 3. For the other 71, defaulting rates are 3-48%, meaning Sibyl actively returns a value for the majority of requests. The low O1-match rates on float-ratio features (`*_ctr`, `*_cvr`, `*_discounted_*`) are not explained by defaulting; they're explained by float-precision loss. The O1:O2 ratio is robust to either: defaulted rows would match neither src(O1) nor src(O2), so they fall out of both counts equally.

### Test B extension — 39 `baf_b_*` features (business-only key, same source table)

Same SQL template, single source table `datalake.ml.doubledash_store_engagement_b`, joined on `business_id` only (no consumer). Aggregate over 39 features × 96,645 joined rows = 3.77M observations:

| metric | count | rate |
|---|---:|---:|
| Online non-null | 3,769,155 | — |
| **online == src(O1)** | **584,871** | **15.5%** |
| **online == src(O2)** | **59,962** | **1.6%** |
| **O1:O2 match ratio** | — | **9.8×** |

Lower absolute O1 rate on the day-wide 6/11 sample compared with the prior 6/22 single-hour sample (54.8% → 15.5%). Most plausible cause: on 6/11 (model's first serving day) Sibyl's Cassandra mirror was reading a different freshness than D-2, so exact-equality vs `src(D-2)` fails more often. The O1:O2 *ratio* is conserved (still 9.8×) — that's the bug signal. Full 39-row table in §A.2.

### Store-keyed features (38)

Layer 2's store_id test above (98.6% Sibyl used O1's primary store) confirms the same mechanism. Layer 3 per-feature value test for the 38 store-keyed features (`*_cs_st_*`, `*_st_*`) is queued — same SQL pattern with different source tables. Inclusion in "up to 110" rests on layers 1 + 2 alone for these 38.

### Can we know for sure which entity Sibyl served per feature? (Beyond the 3 layers above)

The three-layer chain (model expects O2 + Sibyl recorded O1 + value traces to src(cs,O1)) is conclusive for the verified 72 features (33 cs_b + 39 baf_b). For the remaining 38 store-keyed CANDIDATE features we have layers 1+2 only. The user's question is fair: is there a way to literally observe "for feature X, Sibyl looked up at entity Y" rather than inferring it through these joins? Yes — none yet wired up in this session's tooling:

| approach | cost | what it would prove |
|---|---|---|
| **Sibyl trace logs** — enable per-request feature-resolution logging on the Sibyl side for this predictor. Each log emits `(feature_name, entity_used, value_returned)`. | medium — requires Sibyl team change | Direct, definitive, per-feature, per-request. |
| **Sibyl gRPC replay** — call `sibyl-prediction-service` with a controlled (consumer_id=X, business_id@seqIndex=0=Y, business_id@seqIndex=1=Z) request and inspect what comes back per feature. | medium — needs Sibyl client creds + the right `.proto`s | Direct, controlled, but requires offline replay capability. |
| **Per-feature `default_values_used` audit** — `default_values_used` map in the prediction log includes which features defaulted. Cross-tab against the value match-rate from Layer 3 to confirm "low-O1-match" features aren't actually 100% defaulted (and therefore not affected). | trivial — already runnable in Databricks | Indirect, but rules out the "feature is defaulted everywhere, so the O1/O2 question is moot" alternative explanation. |
| **Cassandra direct read** — `cqlsh` the Sibyl-online keyspace at `(cs, O1, feature)` and `(cs, O2, feature)`, compare to what the prediction log served. | medium — needs Cassandra credentials | Direct, but only tells us what's stored, not what Sibyl resolves at request-time. |

What this report relies on instead is the convergence of 3 independent indirect signals (model config + prediction-log entity column + source-table value match) all pointing to the same answer. Net confidence is high but not absolute.

### Fix options (priority order)

1. **AIMS metadata fix (lowest risk, no retrain)** — add `seqIndex: 1` to the second entity slot in every affected feature's AIMS declaration where the model expects O2 (CANDIDATE-side). Affects `business_id` slot on the 33+39 verified features, plus `store_id` slot on the 38 store-keyed. Single largest expected AUC recovery.
2. **Rename features in the model config** to `caf_cs_o2b_*` / `daf_cs_o2b_*` (following DoorDash's CANDIDATE-side naming convention). Requires retraining. Cleaner long-term.
3. Patch feed-service to add a CANDIDATE-only `business_id` entity. Wrong layer — the binding semantics belong with the feature.

---

## §5 — Shadow expiry follow-up

`shadow_expire_date: 2026-06-15` is 9 days in the past as of 2026-06-24 and the shadow is still running. Worth confirming this is intentional rather than a missed expiry sweep.

---

## §6 — Recommended action sequence

| # | action | risk | expected impact |
|---|---|---|---|
| 1 | Read both `doubledash_store_ranking_config` (`147f2680-…`) and `bundle_ranking_config_postcheckout` (`4ba4fe5e-…`) DVs via `dv` CLI / gRPC. Align if needed. | trivial | recovers ~7 of 12 Bug-1 features |
| 2 | Follow-up PR in `BundleMLFeatureGenerator`: drop `_int` suffix on 3 features; emit the 5 missing features. | small | recovers remaining 5 Bug-1 features |
| 3 | **AIMS metadata fix on every `*_cs_b_*` / `baf_b_*` / store-keyed feature with no `seqIndex` hint**: set `seqIndex: 1` on the second-entity slot for features the DNN treats as CANDIDATE-side. | trivial | **largest single AUC recovery** — fixes up to 110 features at once, no retrain |
| 4 | Cassandra spot-check on the converted-business sequence feature for a known-in-Iceberg consumer_id. Pick a path: Sibyl/Fabricator owner with `cqlsh`, or feed-service trace logging. | trivial | informs Bug 3 fix path |
| 5 | Per-feature value test on the 38 store-keyed features (same SQL pattern as §4 Test B, swap source tables and join on store_id). | trivial | closes the open scope on Bug 4 |
| 6 | Decide cos_sim direction (option 1/2/3 in §2). Run feature-importance first. | depends | unknown |
| 7 | Re-run Phase F AUC after actions 1+3 deploy. Expected ceiling ~0.89. | nil | measures recovery |
| 8 | Investigate why `shadow_expire_date: 2026-06-15` was ignored. | nil | governance |

---

> **Fixes:** the actionable fix plan + system knowledge (architecture, O1/O2 position machinery, code pointers, retrain/backfill plan) lives in the companion doc **`dbd_fixes.md`**.

---

## §A — Appendix: full per-feature verification tables

### §A.1 — 33 `*_cs_b_*` features (Bug 4 Test B, v3, 100k random 6/11 sample)

Sample from prediction log: full 2026-06-11 day, deterministic 100k random sample via `pmod(hash(request_id), 2570) = 0`. INNER JOIN to `base_instance_v2` on consumer_id + 120s timestamp window → 96,645 (pred, v2) pairs. Source tables joined twice at `active_date='2026-06-09'` (D-2). Sorted by feature name.

| # | feature | online | src@O1 | src@O2 | match O1 | match O2 | O1 rate | O2 rate | O1:O2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `caf_cs_b_consumer_business_recency_discounted_clicks` | 96,645 | 51,414 | 24,388 | 5,030 | 285 | **5.2%** | 0.3% | 17.6× |
| 2 | `caf_cs_b_consumer_business_recency_discounted_ctr` | 96,645 | 51,414 | 24,388 | 3,342 | 247 | **3.5%** | 0.3% | 13.5× |
| 3 | `caf_cs_b_consumer_business_recency_discounted_cvr` | 96,645 | 51,414 | 24,388 | 10,234 | 2,929 | **10.6%** | 3.0% | 3.5× |
| 4 | `caf_cs_b_consumer_business_recency_discounted_orders` | 96,645 | 51,414 | 24,388 | 11,686 | 2,947 | **12.1%** | 3.0% | 4.0× |
| 5 | `caf_cs_b_consumer_business_recency_discounted_views` | 96,645 | 51,414 | 24,388 | 3,716 | 3 | **3.8%** | 0.0% | 1238.7× |
| 6 | `caf_cs_b_p168d_consumer_business_clicks` | 96,645 | 76,039 | 52,264 | 74,889 | 3,839 | **77.5%** | 4.0% | 19.5× |
| 7 | `caf_cs_b_p168d_consumer_business_views` | 96,645 | 79,051 | 68,435 | 77,671 | 556 | **80.4%** | 0.6% | 139.7× |
| 8 | `caf_cs_b_p168d_organic_clicks` | 96,645 | 81,482 | 68,108 | 80,410 | 6,357 | **83.2%** | 6.6% | 12.6× |
| 9 | `caf_cs_b_p168d_organic_ctr` | 96,645 | 81,482 | 68,108 | 17,190 | 3,129 | **17.8%** | 3.2% | 5.5× |
| 10 | `caf_cs_b_p168d_organic_cvr` | 96,645 | 81,482 | 68,108 | 29,774 | 13,778 | **30.8%** | 14.3% | 2.2× |
| 11 | `caf_cs_b_p1y_order_recency` | 96,645 | 60,215 | 31,711 | 58,333 | 454 | **60.4%** | 0.5% | 128.5× |
| 12 | `caf_cs_b_p28d_consumer_business_clicks` | 96,645 | 58,028 | 28,895 | 57,016 | 5,961 | **59.0%** | 6.2% | 9.6× |
| 13 | `caf_cs_b_p28d_consumer_business_view_recency` | 96,645 | 80,156 | 70,892 | 32,696 | 20,482 | **33.8%** | 21.2% | 1.6× |
| 14 | `caf_cs_b_p28d_consumer_business_views` | 96,645 | 79,568 | 69,083 | 78,023 | 2,551 | **80.7%** | 2.6% | 30.6× |
| 15 | `caf_cs_b_p84d_consumer_business_click_recency` | 96,645 | 72,100 | 46,188 | 69,915 | 3,663 | **72.3%** | 3.8% | 19.1× |
| 16 | `caf_cs_b_p84d_consumer_business_clicks` | 96,645 | 70,293 | 43,306 | 69,191 | 4,601 | **71.6%** | 4.8% | 15.0× |
| 17 | `caf_cs_b_p84d_consumer_business_views` | 96,645 | 81,119 | 69,572 | 79,780 | 1,050 | **82.5%** | 1.1% | 76.0× |
| 18 | `caf_cs_b_p84d_organic_conversions` | 96,645 | 80,545 | 67,577 | 79,587 | 21,382 | **82.3%** | 22.1% | 3.7× |
| 19 | `caf_cs_b_p84d_organic_cvr` | 96,645 | 80,545 | 67,577 | 35,430 | 19,201 | **36.7%** | 19.9% | 1.8× |
| 20 | `daf_cs_b_p168d_consumer_business_orders` | 96,645 | 62,207 | 35,746 | 61,186 | 4,079 | **63.3%** | 4.2% | 15.0× |
| 21 | `daf_cs_b_p168d_global_search_clicks` | 96,645 | 49,930 | 33,071 | 47,122 | 4,526 | **48.8%** | 4.7% | 10.4× |
| 22 | `daf_cs_b_p168d_global_search_conversions` | 96,645 | 36,138 | 19,680 | 34,462 | 2,192 | **35.7%** | 2.3% | 15.7× |
| 23 | `daf_cs_b_p168d_global_search_ctr` | 96,645 | 49,930 | 33,071 | 17,319 | 522 | **17.9%** | 0.5% | 33.2× |
| 24 | `daf_cs_b_p168d_global_search_cvr` | 96,645 | 36,138 | 19,680 | 12,085 | 153 | **12.5%** | 0.2% | 79.0× |
| 25 | `daf_cs_b_p168d_global_search_views` | 96,645 | 74,861 | 65,476 | 65,779 | 4,494 | **68.1%** | 4.7% | 14.6× |
| 26 | `daf_cs_b_p28d_consumer_business_orders` | 96,645 | 42,084 | 17,959 | 41,454 | 4,511 | **42.9%** | 4.7% | 9.2× |
| 27 | `daf_cs_b_p56d_global_search_clicks` | 96,645 | 34,967 | 19,845 | 32,014 | 4,202 | **33.1%** | 4.3% | 7.6× |
| 28 | `daf_cs_b_p56d_global_search_conversions` | 96,645 | 23,821 | 11,144 | 22,034 | 1,787 | **22.8%** | 1.8% | 12.3× |
| 29 | `daf_cs_b_p56d_global_search_ctr` | 96,645 | 34,967 | 19,845 | 17,121 | 832 | **17.7%** | 0.9% | 20.6× |
| 30 | `daf_cs_b_p56d_global_search_cvr` | 96,645 | 23,821 | 11,144 | 10,899 | 216 | **11.3%** | 0.2% | 50.5× |
| 31 | `daf_cs_b_p56d_global_search_views` | 96,645 | 64,462 | 51,204 | 54,618 | 6,650 | **56.5%** | 6.9% | 8.2× |
| 32 | `daf_cs_b_p84d_consumer_business_order_recency` | 96,645 | 56,440 | 28,117 | 54,525 | 1,083 | **56.4%** | 1.1% | 50.3× |
| 33 | `daf_cs_b_p84d_consumer_business_orders` | 96,645 | 56,552 | 28,110 | 55,843 | 4,277 | **57.8%** | 4.4% | 13.1× |

### §A.2 — 39 `baf_b_*` features (Bug 4 Test B extension, v3, 100k random 6/11 sample)

Same join pattern as §A.1, single source table `datalake.ml.doubledash_store_engagement_b` keyed on `business_id` only. Sorted by O1 match rate descending. Many triplets (p7d/p28d/p84d of the same metric) produced identical match counts — same source row, same online value rounded.

| # | feature | online | src@O1 | src@O2 | match O1 | match O2 | O1 rate | O2 rate | O1:O2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `baf_b_p28d_doubledash_engagement_num_bundled_v2` | 96,645 | 95,697 | 96,549 | 30,635 | 2,644 | **31.7%** | 2.7% | 11.6× |
| 2 | `baf_b_p7d_doubledash_engagement_num_bundled_v2` | 96,645 | 95,697 | 96,549 | 30,635 | 2,644 | **31.7%** | 2.7% | 11.6× |
| 3 | `baf_b_p84d_doubledash_engagement_num_bundled_v2` | 96,645 | 95,697 | 96,549 | 30,635 | 2,642 | **31.7%** | 2.7% | 11.6× |
| 4 | `baf_b_p28d_doubledash_engagement_num_bundled_wo_pin_v2` | 96,645 | 95,079 | 96,075 | 28,239 | 3,960 | **29.2%** | 4.1% | 7.1× |
| 5 | `baf_b_p7d_doubledash_engagement_num_bundled_wo_pin_v2` | 96,645 | 95,079 | 96,075 | 28,239 | 3,960 | **29.2%** | 4.1% | 7.1× |
| 6 | `baf_b_p84d_doubledash_engagement_num_bundled_wo_pin_v2` | 96,645 | 95,079 | 96,075 | 28,239 | 3,958 | **29.2%** | 4.1% | 7.1× |
| 7 | `baf_b_p28d_doubledash_engagement_num_bundled_pos_gt1_v2` | 96,645 | 95,486 | 96,549 | 27,161 | 3,012 | **28.1%** | 3.1% | 9.0× |
| 8 | `baf_b_p7d_doubledash_engagement_num_bundled_pos_gt1_v2` | 96,645 | 95,486 | 96,549 | 27,161 | 3,012 | **28.1%** | 3.1% | 9.0× |
| 9 | `baf_b_p84d_doubledash_engagement_num_bundled_pos_gt1_v2` | 96,645 | 95,486 | 96,549 | 27,161 | 3,012 | **28.1%** | 3.1% | 9.0× |
| 10 | `baf_b_p28d_doubledash_engagement_num_loads_wo_pin_v2` | 96,645 | 95,079 | 96,075 | 18,636 | 1,277 | **19.3%** | 1.3% | 14.6× |
| 11 | `baf_b_p7d_doubledash_engagement_num_loads_wo_pin_v2` | 96,645 | 95,079 | 96,075 | 18,636 | 1,277 | **19.3%** | 1.3% | 14.6× |
| 12 | `baf_b_p84d_doubledash_engagement_num_loads_wo_pin_v2` | 96,645 | 95,079 | 96,075 | 18,636 | 1,285 | **19.3%** | 1.3% | 14.5× |
| 13 | `baf_b_p28d_doubledash_engagement_cvr_wo_pin_v2` | 96,645 | 95,697 | 96,549 | 16,331 | 3,080 | **16.9%** | 3.2% | 5.3× |
| 14 | `baf_b_p7d_doubledash_engagement_cvr_wo_pin_v2` | 96,645 | 95,697 | 96,549 | 16,331 | 3,080 | **16.9%** | 3.2% | 5.3× |
| 15 | `baf_b_p84d_doubledash_engagement_cvr_wo_pin_v2` | 96,645 | 95,697 | 96,549 | 16,331 | 3,074 | **16.9%** | 3.2% | 5.3× |
| 16 | `baf_b_p28d_doubledash_engagement_num_loads_v2` | 96,645 | 95,697 | 96,549 | 15,453 | 521 | **16.0%** | 0.5% | 29.7× |
| 17 | `baf_b_p7d_doubledash_engagement_num_loads_v2` | 96,645 | 95,697 | 96,549 | 15,453 | 521 | **16.0%** | 0.5% | 29.7× |
| 18 | `baf_b_p84d_doubledash_engagement_num_loads_v2` | 96,645 | 95,697 | 96,549 | 15,453 | 521 | **16.0%** | 0.5% | 29.7× |
| 19 | `baf_b_p28d_doubledash_engagement_num_loads_pos_gt1_v2` | 96,645 | 95,486 | 96,549 | 15,205 | 607 | **15.7%** | 0.6% | 25.0× |
| 20 | `baf_b_p7d_doubledash_engagement_num_loads_pos_gt1_v2` | 96,645 | 95,486 | 96,549 | 15,205 | 607 | **15.7%** | 0.6% | 25.0× |
| 21 | `baf_b_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 96,645 | 95,486 | 96,549 | 15,205 | 607 | **15.7%** | 0.6% | 25.0× |
| 22 | `baf_b_p28d_doubledash_engagement_cvr_pos_gt1_v2` | 96,645 | 95,697 | 96,549 | 14,382 | 2,277 | **14.9%** | 2.4% | 6.3× |
| 23 | `baf_b_p7d_doubledash_engagement_cvr_pos_gt1_v2` | 96,645 | 95,697 | 96,549 | 14,382 | 2,277 | **14.9%** | 2.4% | 6.3× |
| 24 | `baf_b_p84d_doubledash_engagement_cvr_pos_gt1_v2` | 96,645 | 95,697 | 96,549 | 14,382 | 2,273 | **14.9%** | 2.4% | 6.3× |
| 25 | `baf_b_p28d_doubledash_engagement_cvr_v2` | 96,645 | 95,697 | 96,549 | 12,423 | 1,784 | **12.9%** | 1.8% | 7.0× |
| 26 | `baf_b_p7d_doubledash_engagement_cvr_v2` | 96,645 | 95,697 | 96,549 | 12,423 | 1,784 | **12.9%** | 1.8% | 7.0× |
| 27 | `baf_b_p84d_doubledash_engagement_cvr_v2` | 96,645 | 95,697 | 96,549 | 12,423 | 1,780 | **12.9%** | 1.8% | 7.0× |
| 28 | `baf_b_p28d_doubledash_engagement_ctr_wo_pin_v2` | 96,645 | 95,697 | 96,549 | 6,729 | 616 | **7.0%** | 0.6% | 10.9× |
| 29 | `baf_b_p7d_doubledash_engagement_ctr_wo_pin_v2` | 96,645 | 95,697 | 96,549 | 6,729 | 616 | **7.0%** | 0.6% | 10.9× |
| 30 | `baf_b_p84d_doubledash_engagement_ctr_wo_pin_v2` | 96,645 | 95,697 | 96,549 | 6,729 | 609 | **7.0%** | 0.6% | 11.0× |
| 31 | `baf_b_p28d_doubledash_engagement_ctr_pos_gt1_v2` | 96,645 | 95,697 | 96,549 | 3,540 | 91 | **3.7%** | 0.1% | 38.9× |
| 32 | `baf_b_p7d_doubledash_engagement_ctr_pos_gt1_v2` | 96,645 | 95,697 | 96,549 | 3,540 | 91 | **3.7%** | 0.1% | 38.9× |
| 33 | `baf_b_p84d_doubledash_engagement_ctr_pos_gt1_v2` | 96,645 | 95,697 | 96,549 | 3,540 | 91 | **3.7%** | 0.1% | 38.9× |
| 34 | `baf_b_p28d_doubledash_engagement_ctr_v2` | 96,645 | 95,697 | 96,549 | 3,514 | 87 | **3.6%** | 0.1% | 40.4× |
| 35 | `baf_b_p7d_doubledash_engagement_ctr_v2` | 96,645 | 95,697 | 96,549 | 3,514 | 87 | **3.6%** | 0.1% | 40.4× |
| 36 | `baf_b_p84d_doubledash_engagement_ctr_v2` | 96,645 | 95,697 | 96,549 | 3,514 | 87 | **3.6%** | 0.1% | 40.4× |
| 37 | `baf_b_p28d_doubledash_engagement_num_imps_v2` | 96,645 | 95,697 | 96,549 | 2,709 | 37 | **2.8%** | 0.0% | 73.2× |
| 38 | `baf_b_p7d_doubledash_engagement_num_imps_v2` | 96,645 | 95,697 | 96,549 | 2,709 | 37 | **2.8%** | 0.0% | 73.2× |
| 39 | `baf_b_p84d_doubledash_engagement_num_imps_v2` | 96,645 | 95,697 | 96,549 | 2,709 | 37 | **2.8%** | 0.0% | 73.2× |

### §A.3 — Feed-service entity emission (the structural cause)

[`Constants.kt`](https://github.com/doordash/feed-service/blob/master/libraries/domain-util/src/main/kotlin/com/doordash/consumer/feed/domainUtil/bundle/jobs/ranking/Constants.kt):
```kotlin
const val SIBYL_PREDICTOR_ENTITY_BUSINESS    = "business_id"      // canonical
const val SIBYL_PREDICTOR_ENTITY_STORE       = "store_id"          // canonical
const val SIBYL_PREDICTOR_PRIMARY_BUSINESS_ID = "o1_business_id"   // O1 tag (single emit)
const val SIBYL_PREDICTOR_O2_BUSINESS_ID     = "o2_business_id"    // O2 tag (single emit)
const val SIBYL_PREDICTOR_PRIMARY_STORE_ID   = "o1_store_id"       // O1 tag
const val SIBYL_PREDICTOR_O2_STORE_ID        = "o2_store_id"       // O2 tag
```

[`BundleMLFeatureGenerator.kt:221-225, 262-265`](https://github.com/doordash/feed-service/blob/master/libraries/domain-util/src/main/kotlin/com/doordash/consumer/feed/domainUtil/bundle/jobs/ranking/BundleMLFeatureGenerator.kt#L221) (the new path):
```kotlin
.addEntityIds(buildEntityId("business_id", primaryBusinessId,   seqIndex = 0))   // O1 at slot 0
.addEntityIds(buildEntityId("business_id", candidateBusinessId, seqIndex = 1))   // O2 at slot 1
.addEntityIds(buildEntityId("store_id",    primaryStoreId,      seqIndex = 0))   // O1 store at slot 0
.addEntityIds(buildEntityId("store_id",    candidateStoreId,    seqIndex = 1))   // O2 store at slot 1
```

Two `business_id` entries + no `seqIndex` hint in the AIMS feature catalog → Sibyl defaults to `seqIndex=0` → cart-anchor.

---

### §A.5 — Loopback test for 33 `*_cs_b_*` features (`online_val == src(pred.cs, pred.business_id)` at D-2)

Sample: 100k random pred rows from 2026-06-11, source @ active_date='2026-06-09'. Match rate measured per feature; if `pred.business_id` were not the entity Sibyl used, match rate would be near 0%.

| # | feature | n_online | match | rate |
|---:|---|---:|---:|---:|
| 1 | `caf_cs_b_p168d_organic_clicks` | 99,056 | 78,318 | **79.1%** |
| 2 | `caf_cs_b_p84d_organic_conversions` | 99,056 | 77,809 | **78.6%** |
| 3 | `caf_cs_b_p84d_consumer_business_views` | 99,056 | 76,568 | **77.3%** |
| 4 | `caf_cs_b_p168d_consumer_business_views` | 99,056 | 75,834 | **76.6%** |
| 5 | `caf_cs_b_p28d_consumer_business_views` | 99,056 | 75,257 | **76.0%** |
| 6 | `caf_cs_b_p168d_consumer_business_clicks` | 99,056 | 72,316 | **73.0%** |
| 7 | `caf_cs_b_p84d_consumer_business_clicks` | 99,056 | 66,237 | **66.9%** |
| 8 | `caf_cs_b_p84d_consumer_business_click_recency` | 99,056 | 65,766 | **66.4%** |
| 9 | `daf_cs_b_p168d_global_search_views` | 99,056 | 64,143 | **64.8%** |
| 10 | `daf_cs_b_p168d_consumer_business_orders` | 99,056 | 63,640 | **64.2%** |
| 11 | `caf_cs_b_p1y_order_recency` | 99,056 | 59,576 | **60.1%** |
| 12 | `daf_cs_b_p84d_consumer_business_orders` | 99,056 | 56,787 | **57.3%** |
| 13 | `daf_cs_b_p84d_consumer_business_order_recency` | 99,056 | 54,370 | **54.9%** |
| 14 | `caf_cs_b_p28d_consumer_business_clicks` | 99,056 | 51,606 | **52.1%** |
| 15 | `daf_cs_b_p56d_global_search_views` | 99,056 | 48,265 | **48.7%** |
| 16 | `daf_cs_b_p168d_global_search_clicks` | 99,056 | 43,216 | **43.6%** |
| 17 | `daf_cs_b_p28d_consumer_business_orders` | 99,056 | 41,719 | **42.1%** |
| 18 | `daf_cs_b_p168d_global_search_conversions` | 99,056 | 33,499 | **33.8%** |
| 19 | `caf_cs_b_p84d_organic_cvr` | 99,056 | 33,283 | **33.6%** |
| 20 | `daf_cs_b_p56d_global_search_clicks` | 99,056 | 26,844 | **27.1%** |
| 21 | `caf_cs_b_p168d_organic_cvr` | 99,056 | 25,426 | **25.7%** |
| 22 | `caf_cs_b_p28d_consumer_business_view_recency` | 99,056 | 22,241 | **22.5%** |
| 23 | `daf_cs_b_p168d_global_search_ctr` | 99,056 | 20,606 | **20.8%** |
| 24 | `daf_cs_b_p56d_global_search_conversions` | 99,056 | 19,811 | **20.0%** |
| 25 | `caf_cs_b_p168d_organic_ctr` | 99,056 | 16,703 | **16.9%** |
| 26 | `daf_cs_b_p56d_global_search_ctr` | 99,056 | 15,899 | **16.1%** |
| 27 | `daf_cs_b_p168d_global_search_cvr` | 99,056 | 14,739 | **14.9%** |
| 28 | `daf_cs_b_p56d_global_search_cvr` | 99,056 | 11,127 | **11.2%** |
| 29 | `caf_cs_b_consumer_business_recency_discounted_orders` | 99,056 | 9,075 | **9.2%** |
| 30 | `caf_cs_b_consumer_business_recency_discounted_cvr` | 99,056 | 8,220 | **8.3%** |
| 31 | `caf_cs_b_consumer_business_recency_discounted_clicks` | 99,056 | 4,185 | **4.2%** |
| 32 | `caf_cs_b_consumer_business_recency_discounted_views` | 99,056 | 3,264 | **3.3%** |
| 33 | `caf_cs_b_consumer_business_recency_discounted_ctr` | 99,056 | 3,249 | **3.3%** |

Aggregate: 1,339,598 / 3,268,848 = **40.98%**

### §A.6 — Loopback test for 39 `baf_b_*` features (`online_val == src(pred.business_id)` at D-2)

Same idea, single source table `datalake.ml.doubledash_store_engagement_b` joined on business_id only.

| # | feature | n_online | match | rate |
|---:|---|---:|---:|---:|
| 1 | `baf_b_p28d_doubledash_engagement_num_bundled_v2` | 99,056 | 28,469 | **28.7%** |
| 2 | `baf_b_p7d_doubledash_engagement_num_bundled_v2` | 99,056 | 28,469 | **28.7%** |
| 3 | `baf_b_p84d_doubledash_engagement_num_bundled_v2` | 99,056 | 28,469 | **28.7%** |
| 4 | `baf_b_p28d_doubledash_engagement_num_bundled_wo_pin_v2` | 99,056 | 27,299 | **27.6%** |
| 5 | `baf_b_p7d_doubledash_engagement_num_bundled_wo_pin_v2` | 99,056 | 27,299 | **27.6%** |
| 6 | `baf_b_p84d_doubledash_engagement_num_bundled_wo_pin_v2` | 99,056 | 27,299 | **27.6%** |
| 7 | `baf_b_p28d_doubledash_engagement_num_bundled_pos_gt1_v2` | 99,056 | 25,633 | **25.9%** |
| 8 | `baf_b_p7d_doubledash_engagement_num_bundled_pos_gt1_v2` | 99,056 | 25,633 | **25.9%** |
| 9 | `baf_b_p84d_doubledash_engagement_num_bundled_pos_gt1_v2` | 99,056 | 25,633 | **25.9%** |
| 10 | `baf_b_p28d_doubledash_engagement_num_loads_wo_pin_v2` | 99,056 | 17,056 | **17.2%** |
| 11 | `baf_b_p7d_doubledash_engagement_num_loads_wo_pin_v2` | 99,056 | 17,056 | **17.2%** |
| 12 | `baf_b_p84d_doubledash_engagement_num_loads_wo_pin_v2` | 99,056 | 17,056 | **17.2%** |
| 13 | `baf_b_p28d_doubledash_engagement_cvr_wo_pin_v2` | 99,056 | 16,516 | **16.7%** |
| 14 | `baf_b_p7d_doubledash_engagement_cvr_wo_pin_v2` | 99,056 | 16,516 | **16.7%** |
| 15 | `baf_b_p84d_doubledash_engagement_cvr_wo_pin_v2` | 99,056 | 16,516 | **16.7%** |
| 16 | `baf_b_p28d_doubledash_engagement_cvr_pos_gt1_v2` | 99,056 | 13,285 | **13.4%** |
| 17 | `baf_b_p7d_doubledash_engagement_cvr_pos_gt1_v2` | 99,056 | 13,285 | **13.4%** |
| 18 | `baf_b_p84d_doubledash_engagement_cvr_pos_gt1_v2` | 99,056 | 13,285 | **13.4%** |
| 19 | `baf_b_p28d_doubledash_engagement_num_loads_pos_gt1_v2` | 99,056 | 13,220 | **13.3%** |
| 20 | `baf_b_p7d_doubledash_engagement_num_loads_pos_gt1_v2` | 99,056 | 13,220 | **13.3%** |
| 21 | `baf_b_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 99,056 | 13,220 | **13.3%** |
| 22 | `baf_b_p28d_doubledash_engagement_num_loads_v2` | 99,056 | 12,690 | **12.8%** |
| 23 | `baf_b_p7d_doubledash_engagement_num_loads_v2` | 99,056 | 12,690 | **12.8%** |
| 24 | `baf_b_p84d_doubledash_engagement_num_loads_v2` | 99,056 | 12,690 | **12.8%** |
| 25 | `baf_b_p28d_doubledash_engagement_cvr_v2` | 99,056 | 11,766 | **11.9%** |
| 26 | `baf_b_p7d_doubledash_engagement_cvr_v2` | 99,056 | 11,766 | **11.9%** |
| 27 | `baf_b_p84d_doubledash_engagement_cvr_v2` | 99,056 | 11,766 | **11.9%** |
| 28 | `baf_b_p28d_doubledash_engagement_ctr_wo_pin_v2` | 99,056 | 6,857 | **6.9%** |
| 29 | `baf_b_p7d_doubledash_engagement_ctr_wo_pin_v2` | 99,056 | 6,857 | **6.9%** |
| 30 | `baf_b_p84d_doubledash_engagement_ctr_wo_pin_v2` | 99,056 | 6,857 | **6.9%** |
| 31 | `baf_b_p28d_doubledash_engagement_num_imps_v2` | 99,056 | 3,424 | **3.5%** |
| 32 | `baf_b_p7d_doubledash_engagement_num_imps_v2` | 99,056 | 3,424 | **3.5%** |
| 33 | `baf_b_p84d_doubledash_engagement_num_imps_v2` | 99,056 | 3,424 | **3.5%** |
| 34 | `baf_b_p28d_doubledash_engagement_ctr_pos_gt1_v2` | 99,056 | 3,158 | **3.2%** |
| 35 | `baf_b_p7d_doubledash_engagement_ctr_pos_gt1_v2` | 99,056 | 3,158 | **3.2%** |
| 36 | `baf_b_p84d_doubledash_engagement_ctr_pos_gt1_v2` | 99,056 | 3,158 | **3.2%** |
| 37 | `baf_b_p28d_doubledash_engagement_ctr_v2` | 99,056 | 3,006 | **3.0%** |
| 38 | `baf_b_p7d_doubledash_engagement_ctr_v2` | 99,056 | 3,006 | **3.0%** |
| 39 | `baf_b_p84d_doubledash_engagement_ctr_v2` | 99,056 | 3,006 | **3.0%** |

Aggregate: 547,137 / 3,863,184 = **14.16%**


## §B — Method appendix

All SQL ran via `databricks --profile dev api post /api/2.0/sql/statements` on warehouse `68894e6235640a87` (workspace `doordash-dev.cloud.databricks.com`, NOT `dash`). AIMS lookups via devbox tunnel:

```bash
devbox tunnel -c cell2 --new-window
grpcurl -plaintext -max-time 5 aims-web.service.prod.ddsd:50051 grpc.health.v1.Health/Check  # {"status":"SERVING"}
grpcurl -plaintext -d '{"featureName":"<NAME>"}' aims-web.service.prod.ddsd:50051 \
  aims.v1.AIMetadataService/GetFeatureByFeatureName
```

Tables used:
- `datalake.sps_prediction_logging.search` — predictions (NOT `iceberg.*`)
- `datalake.ml_datasets.doubledash_store_ranking_base_instance_v2` — auction context table that `v6.py` (the Fabricator job producing the model's training data) reads. Max active_date = 2026-06-23, 1,480 dates, still being written. Has o1/o2 business_id and store_id.
- `datalake.ml.<source_name>` — Fabricator feature source tables, partitioned by `active_date`. Sibyl reads ~D-2 (verified by per-row match for one feature).

Feed-service code via `gh api repos/doordash/feed-service/contents/<path> -H "Accept: application/vnd.github.v3.raw"`.

SQL artifacts kept at `/tmp/per_feat_queries/*.sql`, `/tmp/per_feat_results/*.json`, `/tmp/extra_results/*.json`, `/tmp/bug3_no_v4/*.sql` for the duration of this session.

---

## §C — Appendix: Bug 3 SQL (verbatim)

Three queries, one per source table. All ran on warehouse `2224fa0cd749c5f8` (Ad-hoc Serverless) with `wait_timeout=50s`. Results in the §3 table use these exact queries.

### §C.1 — `doubledash_store_engagement_b_sm` (3 features, key = b, sm)

`sm` derived from `dim_store(pred.store_id).SUBMARKET_ID` (99.99% derived on 99,056 pred rows).

```sql
-- Bug 3 rerun: doubledash_store_engagement_b_sm  (3 features, key = b, sm)
-- derive consumer's submarket from dim_store(pred.store_id)
-- Rationale: pred.business_id ~= O1 (§4 Layer 3, ~99%). pred.store_id is the same
--            request's anchor store, whose submarket is the consumer's delivery submarket.
WITH pred AS (
  SELECT consumer_id, business_id, store_id, prediction_ts, request_id
  FROM datalake.sps_prediction_logging.search
  WHERE predictor_name='doubledash_store_recommendation_ranking'
    AND model_id='dbd_ranker_v0_1_metaflow_1781132405'
    AND prediction_ts_hour BETWEEN '2026-06-11-00' AND '2026-06-11-23'
    AND pmod(hash(request_id), 2570) = 0
),
dim AS (
  SELECT CAST(STORE_ID AS STRING) AS store_id_str,
         CAST(SUBMARKET_ID AS BIGINT) AS sm_num
  FROM datalake.edw_merchant.dimension_store
),
joined AS (
  SELECT p.consumer_id AS cs,
         p.business_id AS biz,
         p.store_id    AS st,
         CAST(d.sm_num AS STRING) AS sm
  FROM pred p
  LEFT JOIN dim d ON p.store_id = d.store_id_str
)
SELECT COUNT(*) AS n_total,
  SUM(IF(sm IS NOT NULL, 1, 0)) AS n_sm_derived,
  SUM(IF(src.active_date IS NOT NULL, 1, 0))                                                    AS f1_n_src_row,
  SUM(IF(src.`baf_b_sm_p84d_doubledash_engagement_num_imps_v2`         IS NOT NULL, 1, 0))     AS f1_n_feat_nonnull,
  SUM(IF(src.active_date IS NOT NULL, 1, 0))                                                    AS f2_n_src_row,
  SUM(IF(src.`baf_b_sm_p84d_doubledash_engagement_num_loads_pos_gt1_v2` IS NOT NULL, 1, 0))    AS f2_n_feat_nonnull,
  SUM(IF(src.active_date IS NOT NULL, 1, 0))                                                    AS f3_n_src_row,
  SUM(IF(src.`baf_b_sm_p84d_doubledash_engagement_num_bundled_v2`      IS NOT NULL, 1, 0))     AS f3_n_feat_nonnull
FROM joined p
LEFT JOIN datalake.ml.doubledash_store_engagement_b_sm src
  ON CAST(src.business_id   AS STRING) = p.biz
 AND CAST(src.submarket_id  AS STRING) = p.sm
 AND src.active_date = DATE '2026-06-09';
```

Result row (`statement_id 01f170ee-fc1e-16b1-b8b7-26356bab63dd`):

| n_total | n_sm_derived | f1_src_row | f1_feat_nn | f2_src_row | f2_feat_nn | f3_src_row | f3_feat_nn |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 99,056 | 99,045 | 93,339 | 93,339 | 93,339 | 90,568 | 93,339 | 93,339 |

### §C.2 — `doubledash_store_engagement_bizv_cs` (3 features, key = bizv, cs)

`bizv` derived from `dim_business(pred.business_id).BUSINESS_VERTICAL_ID` (81.8% derived on 99,056 pred rows). First tried `dim_store(pred.store_id).BUSINESS_VERTICAL_ID` — only 11.9% populated; switched to `dim_business`.

```sql
-- Bug 3 rerun: doubledash_store_engagement_bizv_cs  (3 features, key = bizv, cs)
-- derive business_vertical_id from dim_business keyed on pred.business_id.
-- (dim_store's BUSINESS_VERTICAL_ID column is only ~12% populated; dim_business
-- gives 81.8% bizv coverage on pred rows — verified separately.)
-- Note: pred.business_id ~= O1 (§4 Layer 3, ~99%); business_id -> business_vertical_id
-- is essentially 1:1, so dim_business(pred.business_id).bizv IS O1's bizv = the bizv
-- Sibyl actually resolves (Sibyl picks seqIndex=0 per §4). Sibyl-side analysis asks
-- "did the source table have a row at the entity Sibyl ACTUALLY queried?" — that's
-- exactly this join.
WITH pred AS (
  SELECT consumer_id, business_id, store_id, prediction_ts, request_id
  FROM datalake.sps_prediction_logging.search
  WHERE predictor_name='doubledash_store_recommendation_ranking'
    AND model_id='dbd_ranker_v0_1_metaflow_1781132405'
    AND prediction_ts_hour BETWEEN '2026-06-11-00' AND '2026-06-11-23'
    AND pmod(hash(request_id), 2570) = 0
),
dim AS (
  SELECT CAST(BUSINESS_ID AS STRING) AS business_id_str,
         CAST(BUSINESS_VERTICAL_ID AS BIGINT) AS bizv_num
  FROM datalake.edw_merchant.dimension_business
),
joined AS (
  SELECT p.consumer_id AS cs,
         p.business_id AS biz,
         p.store_id    AS st,
         CAST(d.bizv_num AS STRING) AS bizv
  FROM pred p
  LEFT JOIN dim d ON p.business_id = d.business_id_str
)
SELECT COUNT(*) AS n_total,
  SUM(IF(bizv IS NOT NULL, 1, 0)) AS n_bizv_derived,
  SUM(IF(src.active_date IS NOT NULL, 1, 0))                                                       AS f1_n_src_row,
  SUM(IF(src.`baf_bizv_cs_p84d_doubledash_engagement_num_imps_v2`         IS NOT NULL, 1, 0))     AS f1_n_feat_nonnull,
  SUM(IF(src.active_date IS NOT NULL, 1, 0))                                                       AS f2_n_src_row,
  SUM(IF(src.`baf_bizv_cs_p84d_doubledash_engagement_num_loads_pos_gt1_v2` IS NOT NULL, 1, 0))    AS f2_n_feat_nonnull,
  SUM(IF(src.active_date IS NOT NULL, 1, 0))                                                       AS f3_n_src_row,
  SUM(IF(src.`baf_bizv_cs_p84d_doubledash_engagement_num_bundled_v2`      IS NOT NULL, 1, 0))     AS f3_n_feat_nonnull
FROM joined p
LEFT JOIN datalake.ml.doubledash_store_engagement_bizv_cs src
  ON CAST(src.business_vertical_id AS STRING) = p.bizv
 AND CAST(src.consumer_id          AS STRING) = p.cs
 AND src.active_date = DATE '2026-06-09';
```

Result row (`statement_id 01f170ef-ecf4-1f33-acce-13eb59cb6b92`):

| n_total | n_bizv_derived | f1_src_row | f1_feat_nn | f2_src_row | f2_feat_nn | f3_src_row | f3_feat_nn |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 99,056 | 81,075 | 5,310 | 5,310 | 5,310 | 5,124 | 5,310 | 5,310 |

### §C.3 — `consumer_business_vertical_cng_business_line_trial` (1 feature, key = cs, bizv)

Same dim_business path as §C.2.

```sql
-- Bug 3 rerun: consumer_business_vertical_cng_business_line_trial  (1 feature, key = cs, bizv)
-- derive business_vertical_id from dim_business keyed on pred.business_id
-- (same approach as bizv_cs; 81.8% bizv coverage vs ~12% via dim_store).
WITH pred AS (
  SELECT consumer_id, business_id, store_id, prediction_ts, request_id
  FROM datalake.sps_prediction_logging.search
  WHERE predictor_name='doubledash_store_recommendation_ranking'
    AND model_id='dbd_ranker_v0_1_metaflow_1781132405'
    AND prediction_ts_hour BETWEEN '2026-06-11-00' AND '2026-06-11-23'
    AND pmod(hash(request_id), 2570) = 0
),
dim AS (
  SELECT CAST(BUSINESS_ID AS STRING) AS business_id_str,
         CAST(BUSINESS_VERTICAL_ID AS BIGINT) AS bizv_num
  FROM datalake.edw_merchant.dimension_business
),
joined AS (
  SELECT p.consumer_id AS cs,
         p.business_id AS biz,
         p.store_id    AS st,
         CAST(d.bizv_num AS STRING) AS bizv
  FROM pred p
  LEFT JOIN dim d ON p.business_id = d.business_id_str
)
SELECT COUNT(*) AS n_total,
  SUM(IF(bizv IS NOT NULL, 1, 0)) AS n_bizv_derived,
  SUM(IF(src.active_date IS NOT NULL, 1, 0))                                                AS f1_n_src_row,
  SUM(IF(src.`daf_cs_bizv_cng_vertical_num_orders` IS NOT NULL, 1, 0))                      AS f1_n_feat_nonnull
FROM joined p
LEFT JOIN datalake.ml.consumer_business_vertical_cng_business_line_trial src
  ON CAST(src.consumer_id          AS STRING) = p.cs
 AND CAST(src.business_vertical_id AS STRING) = p.bizv
 AND src.active_date = DATE '2026-06-09';
```

Result row (`statement_id 01f170ef-f1ac-15a6-989d-8ef3d7dbad01`):

| n_total | n_bizv_derived | f1_src_row | f1_feat_nn |
|---:|---:|---:|---:|
| 99,056 | 81,075 | 9,430 | 9,430 |

### §C.4 — dimension-table sanity-check

The bizv-via-`dim_store` vs bizv-via-`dim_business` choice was settled by this one-shot:

```sql
WITH pred AS (
  SELECT business_id, store_id, request_id
  FROM datalake.sps_prediction_logging.search
  WHERE predictor_name='doubledash_store_recommendation_ranking'
    AND model_id='dbd_ranker_v0_1_metaflow_1781132405'
    AND prediction_ts_hour BETWEEN '2026-06-11-00' AND '2026-06-11-23'
    AND pmod(hash(request_id), 2570) = 0
)
SELECT COUNT(*) AS n_pred,
       SUM(CASE WHEN db.BUSINESS_VERTICAL_ID IS NOT NULL THEN 1 ELSE 0 END) AS n_bizv_from_dimbiz,
       SUM(CASE WHEN ds.BUSINESS_VERTICAL_ID IS NOT NULL THEN 1 ELSE 0 END) AS n_bizv_from_dimstore
FROM pred p
LEFT JOIN datalake.edw_merchant.dimension_business db
  ON CAST(db.BUSINESS_ID AS STRING) = p.business_id
LEFT JOIN datalake.edw_merchant.dimension_store ds
  ON CAST(ds.STORE_ID AS STRING) = p.store_id;
-- 99,056   81,075   11,817   →  dim_business wins, 81.8% vs 11.9%
```

### §C.5 — training-time coverage (v6 training table, active_date 2026-05-08)

For each scalar feature, count non-zero array elements across `array<T>` length 2 over one full active_date (5,855,362 rows × 2 = 11,710,724 elements). Aggregated into two queries because column-count limits — split into f1/f2/f3/f4/f5/f7 and f8-f18. Result rows below.

```sql
-- Group A: features 1, 2, 3, 4, 5, 7  (per-feature triple: n_elts / n_nn / n_pos)
WITH t AS (SELECT * FROM datalake.ml_datasets.doubledash_store_ranking_full_training_instance_v6
           WHERE active_date = DATE '2026-05-08')
SELECT COUNT(*) AS n_rows,
  SUM(SIZE(caf_st_p12w_five_star_rate_upper_bound)) AS f1_n_elts,
  SUM(SIZE(FILTER(caf_st_p12w_five_star_rate_upper_bound, x -> x IS NOT NULL))) AS f1_n_nn,
  SUM(SIZE(baf_b_sm_p84d_doubledash_engagement_num_imps_v2)) AS f2_n_elts,
  SUM(SIZE(FILTER(baf_b_sm_p84d_doubledash_engagement_num_imps_v2, x -> x IS NOT NULL))) AS f2_n_nn,
  SUM(SIZE(FILTER(baf_b_sm_p84d_doubledash_engagement_num_imps_v2, x -> x IS NOT NULL AND x > 0))) AS f2_n_pos,
  -- …same triple for baf_b_sm_p84d_doubledash_engagement_num_loads_pos_gt1_v2 (f3),
  --                  baf_b_sm_p84d_doubledash_engagement_num_bundled_v2 (f4),
  --                  daf_st_p84d_doubledash_business_position_weighted_imp2conv_over_avg (f5),
  --                  caf_cs_b_p28d_consumer_business_view_recency (f7) …
FROM t;
-- Then a follow-up:
SELECT
  SUM(SIZE(FILTER(caf_st_p12w_five_star_rate_upper_bound,           x -> x IS NOT NULL AND x > 0))) AS f1_n_pos,
  SUM(SIZE(FILTER(caf_cs_b_p28d_consumer_business_view_recency,     x -> x IS NOT NULL AND x > 0))) AS f7_n_pos
FROM t;
```

Result:

| f1_n_pos    | f2_n_pos    | f3_n_pos    | f4_n_pos    | f5_n_pos    | f7_n_pos   |
|---:|---:|---:|---:|---:|---:|
| 11,466,784  | 11,530,760  | 11,346,208  | 11,298,747  | 11,410,046  | 5,502,712  |
| → 97.9%     | → 98.5%     | → 96.9%     | → 96.5%     | → 97.4%     | → 47.0%    |

```sql
-- Group B: features 8-18 (per-feature triple — abridged similarly to group A; query text in /tmp/probe_train_cov2.json)
```

Result:

| feature | n_pos | rate |
|---|---:|---:|
| f8  baf_o1b_o2bizv_*_num_imps          | 3,210,736  | 27.4% |
| f9  baf_o1b_o2bizv_*_num_loads_pos_gt1 | 3,107,444  | 26.5% |
| f10 baf_o1b_o2bizv_*_num_bundled       | 2,963,012  | 25.3% |
| f11 daf_cs_bizv_cng_vertical_num_orders | 2,076,174 | 17.7% |
| f12 baf_bizv_cs_*_num_imps             | 1,921,368  | 16.4% |
| f13 baf_bizv_cs_*_num_loads_pos_gt1    | 1,110,294  | 9.5% |
| f14 baf_bizv_cs_*_num_bundled          | 736,642    | 6.3% |
| f15 baf_cs_b_*_num_imps                | 3,296,037  | 28.1% |
| f16 baf_cs_b_*_num_loads_pos_gt1       | 1,956,037  | 16.7% |
| f17 baf_cs_b_*_num_bundled             | 1,054,177  | 9.0% |
| f18 daf_st_offers_store_promo_quality_score_v2_cai | 11,685,454 | **99.8%** |

For row 6 (sequence — list feature, not in v6 directly), source-table non-empty rate at 2026-05-08:

```sql
SELECT COUNT(*) AS n_rows,
  SUM(IF(`caf_cs_p84d_organic_converted_business_id_int32_sequence_cutoff_v2_i64l` IS NOT NULL, 1, 0)) AS n_nn,
  SUM(IF(`caf_cs_p84d_organic_converted_business_id_int32_sequence_cutoff_v2_i64l` IS NOT NULL
         AND SIZE(`caf_cs_p84d_organic_converted_business_id_int32_sequence_cutoff_v2_i64l`) > 0, 1, 0)) AS n_nonempty
FROM datalake.ml.organic_consumer_sequence_aggregated_features_cutoff_p13n_v1
WHERE active_date = DATE '2026-05-08';
-- 90,794,135   74,000,646 (81.5% non-null)   54,487,878 (60.0% non-empty)
```
