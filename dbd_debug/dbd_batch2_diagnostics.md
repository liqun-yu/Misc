# DbD Online vs Offline — Debugging Diagnostics Summary

**Run window:** 2026-06-10 → 2026-06-12 (online cache covers 2026-06-11 in practice)
**Sample:** 506,064 joined impressions; 180 model-input features present on both sides
**Online predictions in AB window:** 591,508,964
**Model:** `1781911252` (`dbd_ranker_v0_metaflow_liqun_test`) — same model scored on both sides
**Drift definition this run:** `|online − offline| / (|online| + 1e-16) ≤ 1e-1` → a row counts as "drift" when online and offline disagree by more than **10% relative** on the both-real cohort.

---

## 1. Headline — model is fine, serving feeds it skewed inputs

Same retrained model, same 506K joined rows, same online `is_store_bundled` labels. Only the feature values differ between online (Sibyl-served) and offline (regenerated from upstream Fabricator tables).

| Setting | n | AUC | mean_pred | actual_rate | ECE |
|---|---:|---:|---:|---:|---:|
| Joined, **online** features → online labels  | 506,064 | **0.6378** | 0.2027 | 0.0404 | **0.1623** |
| Joined, **offline** features → online labels | 506,064 | **0.8923** | 0.0365 | 0.0404 | **0.0045** |
| Online non-joined → online labels   |  96,884 | 0.5503 | — | — | — |
| Offline non-joined → offline labels | 499,633 | 0.9048 | — | — | — |

- **Δ AUC (offline − online features) = +0.2545** on identical rows + labels. The retrained model can rank at 0.89 AUC when fed the offline-regenerated features; on the same rows with Sibyl-served features it drops to 0.64.
- **The online-features path is wildly over-predicting** — mean predicted pConv = 0.20, actual rate = 0.04. ECE is 36× higher than the offline-features path.
- Label match rate on the join = 99.71%, so the join is sound and labels are not the problem.

---

## 2. Calibration on the joined sample

Reliability table (online vs offline features, both against online labels):

| Bucket | n_on | mean_pred_on | actual_on | gap_on | n_off | mean_pred_off | actual_off | gap_off |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [0, 0.1]  | 152,379 | 0.0604 | 0.0222 | **+0.038** | **463,512** | 0.0160 | 0.0187 | −0.003 |
| [0.1, 0.2] | 160,462 | 0.1465 | 0.0337 | **+0.113** |  24,122 | 0.1387 | 0.1592 | −0.020 |
| [0.2, 0.3] |  91,570 | 0.2442 | 0.0462 | **+0.198** |   7,131 | 0.2424 | 0.2761 | −0.034 |
| [0.3, 0.4] |  44,779 | 0.3446 | 0.0546 | **+0.290** |   3,418 | 0.3452 | 0.3736 | −0.028 |
| [0.4, 0.5] |  25,799 | 0.4467 | 0.0628 | **+0.384** |   2,193 | 0.4483 | 0.4569 | −0.009 |
| [0.5, 0.6] |  13,981 | 0.5446 | 0.0755 | **+0.469** |   1,797 | 0.5489 | 0.5342 | +0.015 |
| [0.6, 0.7] |   7,001 | 0.6440 | 0.0994 | **+0.545** |   1,619 | 0.6489 | 0.6387 | +0.010 |
| [0.7, 0.8] |   3,761 | 0.7465 | 0.1391 | **+0.607** |   1,317 | 0.7462 | 0.7289 | +0.017 |
| [0.8, 0.9] |   3,675 | 0.8514 | 0.1516 | **+0.700** |     831 | 0.8419 | 0.7449 | +0.097 |
| [0.9, 1.0] |   2,657 | 0.9353 | 0.2017 | **+0.734** |     124 | 0.9206 | 0.8710 | +0.050 |

Two things jump out:
- **Online: half of all rows (66%) land in buckets [0.1, 0.5]** — at mean_pred ≈ 0.25–0.45 — but the actual conv rate in those rows is only 4-6%. The model thinks it's confident, but the features it's seeing don't actually predict.
- **Offline: 91.6% of rows land in [0, 0.1]** — at mean_pred 0.016 — and the actual rate matches (0.019). The model knows when it doesn't know.

The shape of the failure: Sibyl-served features push pConv to the middle of the distribution where the model has no resolving power, and the model retains its overall confidence because each individual feature looks "plausible" (just shifted/scaled wrong).

---

## 3. Cohorts on the online-serving side

From `default_values_used` (the per-prediction MAP Sibyl logs whenever it substitutes a default):

| Cohort | # features | Definition |
|---|---:|---|
| `pct_defaulted = 1.0` (always defaulted) | **35** | Sibyl substitutes the per-feature default in 100% of online predictions |
| `0.5 ≤ pct_defaulted < 1.0` (mostly defaulted) | **9** | Sibyl substitutes the default in 50–99% of online predictions |
| `pct_defaulted < 0.5` (mostly served real) | **136** | Sibyl serves a real value most of the time |
| Absent from `default_values_used` (always served real) | 0 | — |

Per-side feature-quality summary (over the 506K joined sample):
- Online side: zero nulls on any feature (`null_on` p50=0, p99=0). The zero/default rate on online is what fluctuates (mean 0.38, p90=1.0).
- Offline side: nulls and zeros are real and informative (mean null_off=0.31, mean zero_off=0.05).
- Online sample looks "complete" but is constructed from defaults for a meaningful subset of features.

---

## 4. The drift story — and the two distinct patterns

5-state distribution per feature, eps = 10% relative tolerance:

- **3 categorical/scalar features** that are 100% defaulted online: 100% online_drop, 0% drift (no real comparison possible).
- **58 features** with `drift% > 50%` of rows — most of the engagement family (`baf_b_p{7,28,84}d_*_v2` × {ctr, cvr, num_loads, num_bundled, num_imps, _wo_pin, _pos_gt1}). These are served (pct_defaulted 3.6-4.4%) but the served values disagree with offline regen on **>85% of rows** by ≥10% relative.
- **55 features** with `10% ≤ drift% ≤ 50%` — primarily consumer-anchored features (`caf_cs_*`, `daf_cs_*`).

The distribution-diff analysis on the both-real cohort splits the drifting features into two visually distinct patterns:

### Pattern A — store-aggregate, near-perfect agreement (Spearman ≥ 0.99)

These are features at store / store-pair / vertical granularity. Online and offline read the same upstream table and produce the same value. Top examples:

| Feature | off_mean | on_mean | ratio | Spearman |
|---|---:|---:|---:|---:|
| `baf_o2b_sm_p84d_doubledash_engagement_num_imps_v2` | 47,729 | 47,963 | 1.00× | **0.99997** |
| `baf_o1b_o2b_p84d_doubledash_engagement_num_imps_v2` | 31,597 | 31,766 | 1.01× | 0.99984 |
| `daf_cs_p84d_organic_food_vertical_repurchased_orders` | 39.0 | 38.8 | 1.00× | 0.9991 |
| `saf_cs_p84d_organic_ctr` | 0.0641 | 0.0641 | 1.00× | 0.9914 |
| `daf_cs_dprg_p84d_organic_food_vertical_total_purchased_business_count` | 8.11 | 8.06 | 0.99× | 0.9889 |

Spearman ~ 1 means the **ranks agree** even when absolute drift exceeds 10%. These features aren't actually problematic for ranking.

### Pattern B — consumer-anchored, systematic 1.5-2× inflation (Spearman 0.1-0.3)

Anything aggregated at the consumer level (`caf_cs_*`, `daf_cs_*`) is consistently **larger on the online side** than the offline regen — by ~1.5-2×, with very low rank correlation:

| Feature | off_mean | on_mean | online inflation | Spearman |
|---|---:|---:|---:|---:|
| `caf_cs_st_p6m_consumer_store_views` | 486 | **973** | **2.00×** | 0.247 |
| `caf_cs_st_p6m_consumer_store_clicks` | 17.5 | 34.6 | 1.98× | 0.207 |
| `caf_cs_st_p28d_consumer_store_views` | 128 | 240 | 1.87× | 0.208 |
| `caf_cs_b_p168d_organic_clicks` | 11.9 | 22.0 | 1.85× | 0.218 |
| `caf_cs_b_p168d_consumer_business_clicks` | 10.7 | 19.1 | 1.79× | 0.214 |
| `daf_cs_b_p168d_global_search_views` | 21.2 | 35.4 | 1.67× | 0.329 |
| `caf_cs_b_p84d_organic_conversions` | 6.02 | 9.80 | 1.63× | 0.146 |
| `daf_cs_b_p168d_consumer_business_orders` | 7.71 | 12.5 | 1.62× | 0.188 |
| `caf_cs_b_p28d_consumer_business_views` | 26.0 | 41.9 | 1.61× | 0.279 |
| `caf_cs_st_p28d_consumer_store_clicks` | 6.49 | 9.61 | 1.48× | 0.158 |

A couple of consumer-anchored *recency* features show the opposite pattern — online is much **smaller** than offline (more recent):

| Feature | off_mean | on_mean | online/offline | Spearman |
|---|---:|---:|---:|---:|
| `caf_cs_bizv_p84d_consumer_business_vertical_click_recency` | 36.8 | 13.4 | **0.36×** | 0.150 |
| `daf_cs_bizv_p84d_consumer_business_vertical_order_recency` | 49.4 | 20.2 | 0.41× | 0.143 |
| `caf_cs_b_p1y_order_recency` | 78.6 | 56.0 | 0.71× | 0.157 |

The consistent fingerprint — *systematic shift up on counts, systematic shift down on recencies, Spearman 0.1-0.3* — strongly suggests the online and offline reads use **different temporal cutoffs** for consumer-side aggregates. Online may be reading a "live up-to-now" snapshot while offline reads training-cutoff-1d, which would inflate counts (more days included) and shrink recencies (more recent events).

---

## 5. The converted-business sequence feature

`caf_cs_p84d_organic_converted_business_id_int32_sequence_cutoff_v2_i64l` (the user's past-converted business sequence):

| Statistic | Online | Offline |
|---|---:|---:|
| `pct_defaulted_online` | 1.0 | — |
| length p50 / p90 / p99 / max | 0 / 0 / 0 / 0 | 33 / 115 / 286 / 500 |
| Target o2_business_id ∈ sequence (all rows) | 0.000 | 0.277 |
| ... when conv = 1 | 0.000 | **0.761** |
| ... when conv = 0 | 0.000 | 0.257 |

**On the online "is it empty vs is it defaulted" question:** both are true at the same time. Sibyl substitutes the per-feature default in 100% of rows (the prediction-log `default_values_used` map flags this feature on every row), AND the registered default value for this list feature is the empty list `[]`. So online predictions see `[]`, and that `[]` is the configured default — not a missing-data fallback or a serialization artifact. Upstream is healthy (`organic_consumer_sequence_aggregated_features_cutoff_p13n_v1`, DAGIT ✅), and offline length p50=33 confirms the data IS there at the source. The breakage is between Sibyl's lookup and the source table.

Offline shows a real repeat-buyer signal — **76% of converted rows have the target business already in the user's past-converted sequence**, vs 26% for non-converted rows.

**Offset scan** (re-pulled the offline sequence at active_date − {0, 1, 2, 3} to test for lag-bug leakage):

| offset_days | overlap rate |
|---:|---:|
| 0 | 0.278 |
| 1 | 0.268 |
| 2 | 0.270 |
| 3 | 0.266 |

Flat across offsets → **no lag-bug leakage**. The cutoff_v2 strategy correctly excludes the impression day, and the 27% overlap is genuine repeat-buyer behavior, not the FI-style today-included join bug.

2×2×2 by sequence length (len=0 vs len>0) on the offline side: conv_rate is 3.8% in the empty-seq cohort vs 4.07% in the non-empty cohort. `lenzero_rate_pos_over_neg_offline = 0.93` — barely any correlation between sequence presence and label. The leakage hunt comes back **clean** for this feature.

**Implication:** offline learns a real signal that online cannot use (sequence is served as `[]`). The cross-attention block is also a dedicated piece of the model graph that effectively becomes a constant input online. Whether fixing this one feature recovers the largest share of the 0.25 AUC gap is **not measured here** — to know that we'd need a one-feature-at-a-time ablation (patch this feature's offline value into an otherwise-online row, re-score, compare AUC). The architectural intuition + the size of the offline conv-rate split make it a strong candidate for first-fix, but it's still a candidate, not a measurement.

---

## 6. Stratified AUC suspects (Phase D.3)

Two features show a sizable real/null cohort AUC gap that survives the served-cohort filter:

| Feature | pct_default | Δ_online (real−null) | Δ_offline (real−null) | stratified_gap |
|---|---:|---:|---:|---:|
| `baf_o2b_sm_p84d_doubledash_engagement_num_imps_v2` | 0.2% | −0.316 | −0.086 | **+0.230** |
| `daf_st_p4w_no_recent_orders_v2`                    | 0.8% | −0.024 | +0.068 | **+0.092** |

For these, the model gets a much bigger AUC boost from the offline real-cohort than the online real-cohort, even though their values agree (Spearman ≈ 1). Worth a closer per-cohort look, but they cannot account for more than ~0.05 of the 0.25 AUC gap.

---

## 7. So what's actually wrong

1. **Sibyl serving is the dominant root cause.** Two distinct failure modes:
   - **35 features always served as default** (categorical ids, cos-sim features, some engagement counts, the converted-business sequence). The upstream Fabricator pipelines for all of these are healthy; this is a Sibyl-side lookup or feature-name mismatch.
   - **~115 features have value drift** between Sibyl-served and offline-regenerated values, dominated by:
     - **Store-level engagement CTR/CVR/load/bundle counts** (the `baf_b_p{7,28,84}d_*_v2` family) — high drift on ≥85% of rows but rank correlation is OK enough that this isn't the biggest ranking problem. Likely a units/scaling/zero-handling mismatch.
     - **Consumer-anchored aggregates** (`caf_cs_*`, `daf_cs_*`) — systematic 1.5-2× inflation on counts and 0.4× compression on recencies, Spearman 0.1-0.3. This is what most hurts ranking.
2. **No leakage in the offline training data.** The converted-business sequence carries a real 3× repeat-buyer signal but offset scan rules out lag-bug leakage. Phase D.5 outrageous-leakage scan returns target ∈ sequence = 0% online (sequence empty) vs 27% offline (genuine repurchase). No degenerate prediction signature in the null cohorts.
3. **The model trains well** (offline AUC 0.89, ECE 0.005 on independent 500K hold-out). Fixing serving — not retraining — is the highest-leverage move.

---

## 8. Recommended next steps

1. **End-to-end validate one always-defaulted feature.** `caf_cs_p84d_organic_converted_business_id_int32_sequence_cutoff_v2_i64l` is the natural starting candidate (it powers the cross-attention block, upstream is healthy, offline carries a sizeable conv-rate split). Whether it actually recovers more AUC than other always-defaulted features is not yet measured — run a one-feature-at-a-time ablation (patch each defaulted feature's offline value into an otherwise-online row and re-score) before committing to a fix ordering. Then trace the chosen feature from `predictor_config` through Sibyl's feature lookup to identify why the default is fired in 100% of rows.
2. **Audit the consumer-anchored feature cutoffs.** Pick `caf_cs_b_p84d_consumer_business_views` and compare the `cutoff_date` used at prediction time vs the cutoff baked into the training data. The 1.5-2× ratio is the signature.
3. **Audit the engagement-CTR/CVR family** for zero/null handling. Pick `baf_b_p84d_doubledash_engagement_ctr_v2` (95.9% drift, 4% online_drop). Check whether the engagement aggregation runs over the same denominator online and offline.
4. **Re-run Phase F after a Sibyl fix** to measure the AUC recovery without retraining. That gives the unambiguous ceiling for the current model.
5. **Do not drop the converted-business sequence** from training. It carries a real signal; the issue is serving, not the feature.

---

## 9. Master per-feature diagnostic table

180 features, sorted by **severity** then **diagnostic** (most severe first). Within each group, sorted by the relevant signal magnitude.

### 9.1 Severity counts

| Severity | Count | Definition |
|---|---:|---|
| 🟥 **CRITICAL** | 35 | Feature is `always_defaulted` — model receives the per-feature default in essentially every prediction. |
| 🟧 **HIGH**     | 67 | `mostly_defaulted` (9), OR high drift (≥50% of rows differ >10% relative) with low/unknown rank correlation. |
| 🟨 **MEDIUM**   | 74 | High drift but rank correlation preserved (Spearman ≥ 0.5), OR moderate drift with broken rank correlation, OR stratified-AUC outlier, OR partially defaulted (20-50%) even when both-real cohort agrees. |
| ⬜ **NONE**     |  4 | Online ≈ offline AND not partially defaulted; no further action. |

### 9.2 Diagnostic counts

| Diagnostic | Count | Meaning |
|---|---:|---|
| `always_defaulted` | 35 | `pct_defaulted_online ≥ 99%` — Sibyl substitutes the per-feature default in essentially every row. |
| `mostly_defaulted` |  9 | `50% ≤ pct_defaulted < 99%` — partial serving. |
| `high_drift`       | 58 | `drift% ≥ 50%` of rows disagree by >10% relative. |
| `moderate_drift`   | 55 | `10% ≤ drift% < 50%` of rows disagree. |
| `low_rank_corr`    |  1 | Spearman < 0.5 and drift ≥ 1% (and not already in a higher category). |
| `stratified_AUC`   | 16 | Phase D.3 stratified-AUC gap ≥ 0.05 between online and offline. |
| `aligned`          |  6 | Match on the joined sample. |

### 9.3 Column legend

- **feature** — Sibyl feature name as it appears in the model config.
- **severity** — single ranked priority bucket per row (see 9.1).
- **diagnostic** — primary failure mode (see 9.2).
- **pipeline** — the Fabricator pipeline (DAGIT job name) that generates the source table. `(computed in v6 construct_dataset; no Fabricator pipeline)` means the value is a model-side transform of two embedding columns, not a pipeline output. `doubledash_store_ranking_base_instance_v2` means the column comes directly from the impression base instance (id/categorical columns).
- **pipeline_status** — `running` = DAGIT ✅ as of 2026-05-22 (Xun Yi's audit). `STOPPED 2026-05-04` = your comment in the Master Doc. `SOURCE-BUG` = pipeline runs but data has a known integrity issue (e.g. `doubledash_store_engagement_b` and `consumer_business_vertical_aggregate_engagement` have byte-identical p7d/p28d/p84d columns — Comment #8). `running (base instance)` means the upstream source is the impression base instance, not a feature pipeline.
- **%default_on** — fraction of online predictions where Sibyl substitutes the default value for this feature (from `default_values_used`).
- **drift%** — fraction of joined rows where `|online − offline| / (|online| + 1e-16) > 0.1` AND both sides have a real value (`eps=10%`).
- **online_drop%** — fraction of joined rows where offline has a real value but online is null / zero / empty.
- **Spearman** — rank correlation between online and offline values on the both-real cohort. Reported only for features that made the top-drift inspection.
- **off_mean / on_mean** — means on the both-real cohort.
- **on/off** — ratio of online mean to offline mean on the both-real cohort. `2.00×` means online values are on average twice as large as offline; `0.36×` means online is roughly a third of offline.
- **note** — what is likely wrong + suggested next step, per feature.

### 9.4 Table

| # | feature | severity | diagnostic | pipeline | pipeline_status | %default_on | drift% | online_drop% | Spearman | off_mean | on_mean | on/off | note |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `baf_b_sm_p84d_doubledash_engagement_num_bundled_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_b_sm` | running | 100.0% | 0.0% | 100.0% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 2 | `baf_b_sm_p84d_doubledash_engagement_num_imps_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_b_sm` | running | 100.0% | 0.0% | 100.0% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 3 | `baf_b_sm_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_b_sm` | running | 100.0% | 0.0% | 100.0% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 4 | `baf_bizv_cs_p84d_doubledash_engagement_num_bundled_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_bizv_cs` | running | 100.0% | 0.0% | 19.5% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 5 | `baf_bizv_cs_p84d_doubledash_engagement_num_imps_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_bizv_cs` | running | 100.0% | 0.0% | 19.5% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 6 | `baf_bizv_cs_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_bizv_cs` | running | 100.0% | 0.0% | 19.2% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 7 | `baf_cs_b_p84d_doubledash_engagement_num_bundled_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_cs_b` | running | 100.0% | 0.0% | 29.6% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 8 | `baf_cs_b_p84d_doubledash_engagement_num_imps_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_cs_b` | running | 100.0% | 0.0% | 29.6% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 9 | `baf_cs_b_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_cs_b` | running | 100.0% | 0.0% | 29.4% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 10 | `baf_o1b_o2bizv_p84d_doubledash_engagement_num_bundled_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_o1b_o2bizv` | running | 100.0% | 0.0% | 29.6% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 11 | `baf_o1b_o2bizv_p84d_doubledash_engagement_num_imps_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_o1b_o2bizv` | running | 100.0% | 0.0% | 29.6% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 12 | `baf_o1b_o2bizv_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟥 critical | always_defaulted | `doubledash_store_engagement_o1b_o2bizv` | running | 100.0% | 0.0% | 29.6% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 13 | `caf_cs_b_p28d_consumer_business_view_recency` | 🟥 critical | always_defaulted | `consumer_business_recency` | running | 100.0% | 0.0% | 50.8% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 14 | `caf_cs_p84d_organic_converted_business_id_int32_sequence_cutoff_v2_i64l` | 🟥 critical | always_defaulted | `organic_consumer_sequence_aggregated_features_cutoff_p13n_v1` | running | 100.0% | 0.0% | 99.1% | — | — | — | — | Sibyl substitutes the per-feature default — which IS the empty list `[]`. Upstream is healthy and offline length p50=33. Feeds the cross-attention block; serving fix here could yield meaningful AUC recovery, though magnitude is unmeasured. Next: end-to-end one-feature serving validation. |
| 15 | `caf_st_p12w_five_star_rate_upper_bound` | 🟥 critical | always_defaulted | `store_property_features_v2` | running (upper_bound has Sibyl publishing bug per Comment #7) | 100.0% | 0.0% | 97.3% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 16 | `cos_sim_copurchase_cx_o2` | 🟥 critical | always_defaulted | `(computed in v6 construct_dataset; no Fabricator pipeline)` | N/A — model-side transform | 100.0% | 0.0% | 68.5% | — | — | — | — | Cos-sim recipe; needs the two embedding columns served correctly. The 0/0 mismatch likely propagates from an upstream embedding default. Next: end-to-end one-feature serving validation. |
| 17 | `cos_sim_cuisine_tag_cx_o2` | 🟥 critical | always_defaulted | `(computed in v6 construct_dataset; no Fabricator pipeline)` | N/A — model-side transform | 100.0% | 0.0% | 82.9% | — | — | — | — | Cos-sim recipe; needs the two embedding columns served correctly. The 0/0 mismatch likely propagates from an upstream embedding default. Next: end-to-end one-feature serving validation. |
| 18 | `cos_sim_food_labels_cx_o2` | 🟥 critical | always_defaulted | `(computed in v6 construct_dataset; no Fabricator pipeline)` | N/A — model-side transform | 100.0% | 0.0% | 64.0% | — | — | — | — | Cos-sim recipe; needs the two embedding columns served correctly. The 0/0 mismatch likely propagates from an upstream embedding default. Next: end-to-end one-feature serving validation. |
| 19 | `cos_sim_food_labels_o1_o2` | 🟥 critical | always_defaulted | `(computed in v6 construct_dataset; no Fabricator pipeline)` | N/A — model-side transform | 100.0% | 0.0% | 44.3% | — | — | — | — | Cos-sim recipe; needs the two embedding columns served correctly. The 0/0 mismatch likely propagates from an upstream embedding default. Next: end-to-end one-feature serving validation. |
| 20 | `cos_sim_saf_st_cuisine_tag_o1_o2` | 🟥 critical | always_defaulted | `(computed in v6 construct_dataset; no Fabricator pipeline)` | N/A — model-side transform | 100.0% | 0.0% | 60.0% | — | — | — | — | Cos-sim recipe; needs the two embedding columns served correctly. The 0/0 mismatch likely propagates from an upstream embedding default. Next: end-to-end one-feature serving validation. |
| 21 | `daf_cs_bizv_cng_vertical_num_orders` | 🟥 critical | always_defaulted | `doubledash_consumer_engagement_features` | running (likely) | 100.0% | 0.0% | 21.3% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 22 | `daf_st_offers_store_promo_quality_score_v2_cai` | 🟥 critical | always_defaulted | `offers_store_promo_quality_v2_store_features` | STOPPED 2026-05-04 | 100.0% | 0.0% | 0.0% | — | — | — | — | Upstream pipeline stopped 2026-05-04 — feature has no source; deprecate or replace. Next: end-to-end one-feature serving validation. |
| 23 | `daf_st_p84d_doubledash_business_position_weighted_imp2conv_over_avg` | 🟥 critical | always_defaulted | `store_rx_delivery_metrics` | running | 100.0% | 0.0% | 98.5% | — | — | — | — | Sibyl substitutes the per-feature default in 100% of rows — upstream pipeline is healthy (✅ DAGIT). Trace from predictor_config → Sibyl feature lookup for the name/alias mismatch. Next: end-to-end one-feature serving validation. |
| 24 | `day_of_week` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 100.0% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 25 | `hour_of_day` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 99.0% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 26 | `is_promoted` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 100.0% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 27 | `o1_business_id_int` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 100.0% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 28 | `o1_business_vertical_id` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 32.5% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 29 | `o1_is_caviar` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 0.0% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 30 | `o1_primary_category_id` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 96.0% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 31 | `o2_business_id_int` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 100.0% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 32 | `o2_business_vertical_id` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 29.8% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 33 | `o2_is_caviar` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 0.0% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 34 | `o2_primary_category_id` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 96.4% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 35 | `submarket_id` | 🟥 critical | always_defaulted | `doubledash_store_ranking_base_instance_v2` | running (base instance) | 100.0% | 0.0% | 100.0% | — | — | — | — | Categorical/id from base instance — Sibyl substitutes default in every row even though base instance is healthy. Likely a serving-config alias or registration miss. Next: end-to-end one-feature serving validation. |
| 36 | `baf_cs_o2b_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟧 high | mostly_defaulted | `doubledash_store_engagement_cs_o2b` | running | 96.5% | 3.0% | 12.1% | — | — | — | — | Sibyl serves the default in 97% of rows. Upstream pipeline is healthy. Next: investigate which lookup-key subset is failing. |
| 37 | `baf_cs_o2b_p84d_doubledash_engagement_num_imps_v2` | 🟧 high | mostly_defaulted | `doubledash_store_engagement_cs_o2b` | running | 96.5% | 6.4% | 2.8% | 0.98 | 13.1 | 12.7 | 0.97× | Sibyl serves the default in 97% of rows. Upstream pipeline is healthy. Next: investigate which lookup-key subset is failing. |
| 38 | `baf_cs_o2b_p84d_doubledash_engagement_num_bundled_v2` | 🟧 high | mostly_defaulted | `doubledash_store_engagement_cs_o2b` | running | 96.5% | 0.8% | 19.8% | — | — | — | — | Sibyl serves the default in 97% of rows. Upstream pipeline is healthy. Next: investigate which lookup-key subset is failing. |
| 39 | `daf_cs_st_p28d_doubledash_num_views` | 🟧 high | mostly_defaulted | `consumer_store_engagement_feature_uploads` | running | 96.5% | 0.4% | 2.5% | — | — | — | — | Sibyl serves the default in 96% of rows. Upstream pipeline is healthy. Next: investigate which lookup-key subset is failing. |
| 40 | `daf_cs_st_p28d_doubledash_num_orders` | 🟧 high | mostly_defaulted | `consumer_store_engagement_feature_uploads` | running | 96.5% | 0.3% | 3.0% | — | — | — | — | Sibyl serves the default in 96% of rows. Upstream pipeline is healthy. Next: investigate which lookup-key subset is failing. |
| 41 | `daf_cs_b_p56d_global_search_cvr` | 🟧 high | mostly_defaulted | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 61.1% | 6.6% | 5.1% | 0.26 | 0.316 | 0.316 | 1.00× | Sibyl serves the default in 61% of rows. Upstream pipeline is healthy. Next: investigate which lookup-key subset is failing. |
| 42 | `daf_cs_b_p56d_global_search_conversions` | 🟧 high | mostly_defaulted | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 61.1% | 5.6% | 5.1% | 0.13 | 5.07 | 6.06 | 1.19× | Sibyl serves the default in 61% of rows. Upstream pipeline is healthy. Next: investigate which lookup-key subset is failing. |
| 43 | `daf_cs_b_p168d_global_search_cvr` | 🟧 high | mostly_defaulted | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 56.4% | 12.2% | 8.4% | 0.24 | 0.225 | 0.241 | 1.07× | Sibyl serves the default in 56% of rows. Upstream pipeline is healthy. Next: investigate which lookup-key subset is failing. |
| 44 | `daf_cs_b_p168d_global_search_conversions` | 🟧 high | mostly_defaulted | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 56.4% | 11.1% | 8.4% | 0.15 | 7.57 | 11.3 | 1.49× | Sibyl serves the default in 56% of rows. Upstream pipeline is healthy. Next: investigate which lookup-key subset is failing. |
| 45 | `caf_cs_st_p6m_consumer_store_views` | 🟧 high | high_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 24.1% | 55.7% | 7.4% | 0.25 | 486 | 973 | 2.00× | Served-but-skewed: 56% of rows differ >10% relative, rank correlation low (Spearman 0.25). Online inflates ~2.00× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 46 | `daf_cs_b_p168d_global_search_views` | 🟧 high | high_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 23.9% | 56.0% | 7.0% | 0.33 | 21.2 | 35.4 | 1.67× | Served-but-skewed: 56% of rows differ >10% relative, rank correlation low (Spearman 0.33). Online inflates ~1.67× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 47 | `caf_cs_b_p168d_consumer_business_views` | 🟧 high | high_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 19.5% | 61.1% | 7.0% | 0.29 | 132 | 207 | 1.57× | Served-but-skewed: 61% of rows differ >10% relative, rank correlation low (Spearman 0.29). Online inflates ~1.57× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 48 | `baf_o1b_o2b_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_o1b_o2b` | running | 18.8% | 51.6% | 15.9% | — | — | — | — | Drift% 52; rank correlation not measured (object-typed or thin both-real cohort). Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 49 | `caf_cs_b_p84d_consumer_business_views` | 🟧 high | high_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 16.1% | 63.5% | 5.5% | 0.29 | 71 | 113 | 1.59× | Served-but-skewed: 64% of rows differ >10% relative, rank correlation low (Spearman 0.29). Online inflates ~1.59× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 50 | `caf_cs_b_p28d_consumer_business_views` | 🟧 high | high_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 12.4% | 64.3% | 3.9% | 0.28 | 26 | 41.9 | 1.61× | Served-but-skewed: 64% of rows differ >10% relative, rank correlation low (Spearman 0.28). Online inflates ~1.61× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 51 | `baf_b_p7d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 4.4% | 93.4% | 6.3% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 93. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 52 | `baf_b_p7d_doubledash_engagement_num_bundled_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 4.4% | 84.1% | 15.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 84. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 53 | `baf_b_p28d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 4.4% | 93.4% | 6.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 93. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 54 | `baf_b_p28d_doubledash_engagement_num_bundled_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 4.4% | 84.1% | 15.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 84. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 55 | `baf_b_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 4.4% | 93.4% | 6.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 93. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 56 | `baf_b_p84d_doubledash_engagement_num_bundled_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 4.4% | 84.1% | 15.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 84. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 57 | `baf_b_p7d_doubledash_engagement_num_loads_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 92.0% | 7.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 92. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 58 | `baf_b_p7d_doubledash_engagement_num_bundled_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 83.3% | 15.7% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 83. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 59 | `baf_b_p7d_doubledash_engagement_ctr_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.8% | 4.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 96. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 60 | `baf_b_p7d_doubledash_engagement_num_imps_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.7% | 1.9% | 0.12 | 2.77e+05 | 2.23e+05 | 0.80× | Pipeline running, but at source the p7/p28/p84 columns are byte-identical (Comment #8) — zero temporal resolution. Drift% 96; Spearman 0.12 indicates rank correlation broken too. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 61 | `baf_b_p7d_doubledash_engagement_ctr_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.6% | 4.3% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 96. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 62 | `baf_b_p7d_doubledash_engagement_num_loads_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.4% | 4.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 95. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 63 | `baf_b_p7d_doubledash_engagement_ctr_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 93.1% | 6.8% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 93. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 64 | `baf_b_p7d_doubledash_engagement_cvr_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 88.0% | 12.0% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 88. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 65 | `baf_b_p7d_doubledash_engagement_num_bundled_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 87.2% | 12.0% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 87. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 66 | `baf_b_p7d_doubledash_engagement_cvr_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 86.7% | 13.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 87. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 67 | `baf_b_p7d_doubledash_engagement_cvr_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 84.6% | 15.4% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 85. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 68 | `baf_b_p28d_doubledash_engagement_num_loads_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 92.0% | 7.1% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 92. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 69 | `baf_b_p28d_doubledash_engagement_num_bundled_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 83.3% | 15.7% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 83. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 70 | `baf_b_p28d_doubledash_engagement_num_imps_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.7% | 1.9% | 0.12 | 2.77e+05 | 2.23e+05 | 0.80× | Pipeline running, but at source the p7/p28/p84 columns are byte-identical (Comment #8) — zero temporal resolution. Drift% 96; Spearman 0.12 indicates rank correlation broken too. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 71 | `baf_b_p28d_doubledash_engagement_num_loads_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.5% | 4.1% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 95. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 72 | `baf_b_p28d_doubledash_engagement_num_bundled_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 87.2% | 12.0% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 87. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 73 | `baf_b_p28d_doubledash_engagement_cvr_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 84.6% | 15.4% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 85. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 74 | `baf_b_p28d_doubledash_engagement_ctr_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.8% | 4.1% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 96. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 75 | `baf_b_p28d_doubledash_engagement_ctr_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.7% | 4.3% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 96. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 76 | `baf_b_p28d_doubledash_engagement_ctr_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 93.1% | 6.8% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 93. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 77 | `baf_b_p28d_doubledash_engagement_cvr_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 88.0% | 12.0% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 88. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 78 | `baf_b_p28d_doubledash_engagement_cvr_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 86.8% | 13.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 87. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 79 | `baf_b_p84d_doubledash_engagement_num_loads_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 92.0% | 7.1% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 92. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 80 | `baf_b_p84d_doubledash_engagement_num_bundled_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 83.3% | 15.6% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 83. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 81 | `baf_b_p84d_doubledash_engagement_ctr_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.9% | 4.1% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 96. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 82 | `baf_b_p84d_doubledash_engagement_num_imps_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.8% | 1.8% | 0.12 | 2.77e+05 | 2.23e+05 | 0.80× | Pipeline running, but at source the p7/p28/p84 columns are byte-identical (Comment #8) — zero temporal resolution. Drift% 96; Spearman 0.12 indicates rank correlation broken too. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 83 | `baf_b_p84d_doubledash_engagement_ctr_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.7% | 4.2% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 96. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 84 | `baf_b_p84d_doubledash_engagement_num_loads_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 95.5% | 4.1% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 96. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 85 | `baf_b_p84d_doubledash_engagement_ctr_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 93.2% | 6.8% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 93. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 86 | `baf_b_p84d_doubledash_engagement_cvr_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 88.0% | 11.9% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 88. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 87 | `baf_b_p84d_doubledash_engagement_num_bundled_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 87.3% | 11.9% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 87. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 88 | `baf_b_p84d_doubledash_engagement_cvr_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 86.8% | 13.1% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 87. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 89 | `baf_b_p84d_doubledash_engagement_cvr_wo_pin_v2` | 🟧 high | high_drift | `doubledash_store_engagement_b` | running, SOURCE-BUG (p7d/p28d/p84d byte-identical) | 3.6% | 84.7% | 15.3% | — | — | — | — | Pipeline running, but p7/p28/p84 are byte-identical at source (Comment #8). Drift% 85. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 90 | `saf_st_tier_level_v2` | 🟧 high | high_drift | `store_property_features_v2` | running | 2.7% | 63.4% | 0.4% | 0.06 | 3.63 | 3.67 | 1.01× | Served-but-skewed: 63% of rows differ >10% relative, rank correlation low (Spearman 0.06).  Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 91 | `saf_st_p28d_discovery_ctr` | 🟧 high | high_drift | `store_engagement_features` | running | 2.6% | 89.1% | 1.4% | 0.09 | 0.0207 | 0.0235 | 1.13× | Served-but-skewed: 89% of rows differ >10% relative, rank correlation low (Spearman 0.09). Online inflates ~1.13× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 92 | `daf_st_p28d_discovery_cvr` | 🟧 high | high_drift | `store_engagement_features` | running | 2.6% | 79.1% | 5.8% | 0.08 | 0.00635 | 0.00667 | 1.05× | Served-but-skewed: 79% of rows differ >10% relative, rank correlation low (Spearman 0.08).  Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 93 | `saf_st_p84d_discovery_ctr` | 🟧 high | high_drift | `store_engagement_features` | running | 2.6% | 91.0% | 0.4% | 0.10 | 0.0194 | 0.0222 | 1.15× | Served-but-skewed: 91% of rows differ >10% relative, rank correlation low (Spearman 0.10). Online inflates ~1.15× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 94 | `daf_st_p84d_discovery_cvr` | 🟧 high | high_drift | `store_engagement_features` | running | 2.6% | 85.5% | 3.3% | 0.10 | 0.00559 | 0.00606 | 1.08× | Served-but-skewed: 85% of rows differ >10% relative, rank correlation low (Spearman 0.10).  Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 95 | `saf_st_p168d_discovery_ctr` | 🟧 high | high_drift | `store_engagement_features` | running | 2.6% | 90.0% | 0.3% | 0.09 | 0.0229 | 0.0258 | 1.13× | Served-but-skewed: 90% of rows differ >10% relative, rank correlation low (Spearman 0.09). Online inflates ~1.13× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 96 | `daf_st_p168d_discovery_cvr` | 🟧 high | high_drift | `store_engagement_features` | running | 2.6% | 88.3% | 1.7% | 0.10 | 0.00618 | 0.00669 | 1.08× | Served-but-skewed: 88% of rows differ >10% relative, rank correlation low (Spearman 0.10).  Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 97 | `saf_dprg_st_p84d_organic_ctr` | 🟧 high | high_drift | `organic_consumer_daypart_food_vertical_repurchase_features_v2` | running | 1.8% | 83.7% | 0.0% | 0.06 | 0.0568 | 0.0628 | 1.11× | Served-but-skewed: 84% of rows differ >10% relative, rank correlation low (Spearman 0.06). Online inflates ~1.11× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 98 | `saf_dprg_st_p28d_organic_cvr` | 🟧 high | high_drift | `organic_store_daypart_aggregated_features` | running | 1.8% | 88.4% | 0.5% | 0.07 | 0.017 | 0.0188 | 1.11× | Served-but-skewed: 88% of rows differ >10% relative, rank correlation low (Spearman 0.07). Online inflates ~1.11× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 99 | `saf_st_menu_item_photo_coverage` | 🟧 high | high_drift | `store_property_features_v2` | running | 0.9% | 78.5% | 13.1% | — | — | — | — | Drift% 79; rank correlation not measured (object-typed or thin both-real cohort). Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 100 | `daf_st_p4w_store_delivery_market_share_v2` | 🟧 high | high_drift | `store_rx_delivery_metrics` | running | 0.8% | 93.7% | 0.3% | 0.34 | 0.0199 | 0.0276 | 1.39× | Served-but-skewed: 94% of rows differ >10% relative, rank correlation low (Spearman 0.34). Online inflates ~1.39× offline. Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 101 | `baf_o2b_sm_p84d_doubledash_engagement_num_loads_pos_gt1_v2` | 🟧 high | high_drift | `doubledash_store_engagement_o2b_sm` | running | 0.2% | 89.6% | 0.1% | — | — | — | — | Drift% 90; rank correlation not measured (object-typed or thin both-real cohort). Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 102 | `baf_o2b_sm_p84d_doubledash_engagement_num_bundled_v2` | 🟧 high | high_drift | `doubledash_store_engagement_o2b_sm` | running | 0.2% | 73.9% | 1.8% | — | — | — | — | Drift% 74; rank correlation not measured (object-typed or thin both-real cohort). Next: audit served-vs-source value formula (zero-handling, cutoff_date, units). |
| 103 | `daf_cs_b_p56d_global_search_ctr` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 49.6% | 13.4% | 5.8% | 0.21 | 0.359 | 0.358 | 1.00× | Consumer-anchored aggregate: 13% of rows differ, Spearman 0.21. online/offline ratio 1.00×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 104 | `daf_cs_b_p56d_global_search_clicks` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 49.6% | 11.2% | 5.8% | 0.15 | 4.7 | 6.04 | 1.29× | Consumer-anchored aggregate: 11% of rows differ, Spearman 0.15. online/offline ratio 1.29×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 105 | `daf_cs_b_p168d_global_search_ctr` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 44.8% | 23.1% | 8.9% | 0.19 | 0.282 | 0.299 | 1.06× | Consumer-anchored aggregate: 23% of rows differ, Spearman 0.19. online/offline ratio 1.06×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 106 | `daf_cs_b_p168d_global_search_clicks` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 44.8% | 21.2% | 8.9% | 0.18 | 7.5 | 11.8 | 1.57× | Consumer-anchored aggregate: 21% of rows differ, Spearman 0.18. online/offline ratio 1.57×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 107 | `caf_cs_b_p1y_order_recency` | 🟨 medium | moderate_drift | `consumer_business_recency` | running | 36.1% | 15.7% | 14.8% | 0.16 | 78.6 | 56 | 0.71× | Consumer-anchored aggregate: 16% of rows differ, Spearman 0.16. online/offline ratio 0.71×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 108 | `daf_cs_st_p24w_consumer_order_rank_v2` | 🟨 medium | moderate_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 34.2% | 17.1% | 6.7% | 0.00 | 0.617 | 0.635 | 1.03× | Consumer-anchored aggregate: 17% of rows differ, Spearman 0.00. online/offline ratio 1.03×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 109 | `daf_cs_st_p24w_order_count` | 🟨 medium | moderate_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 34.2% | 16.5% | 6.7% | — | — | — | — | Drift% 16. |
| 110 | `daf_cs_st_p24w_order_rank` | 🟨 medium | moderate_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 34.2% | 14.7% | 10.7% | — | — | — | — | Drift% 15. |
| 111 | `daf_cs_st_p24w_consumer_order_count_v2` | 🟨 medium | moderate_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 34.2% | 10.8% | 6.7% | 0.31 | 0.845 | 0.849 | 1.00× | Consumer-anchored aggregate: 11% of rows differ, Spearman 0.31. online/offline ratio 1.00×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 112 | `daf_cs_st_p6m_consumer_store_orders` | 🟨 medium | moderate_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 32.9% | 21.9% | 7.4% | 0.19 | 7.17 | 11.8 | 1.64× | Consumer-anchored aggregate: 22% of rows differ, Spearman 0.19. online/offline ratio 1.64×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 113 | `daf_cs_b_p84d_consumer_business_order_recency` | 🟨 medium | moderate_drift | `consumer_business_recency` | running | 32.9% | 18.7% | 8.9% | 0.10 | 25.5 | 27.1 | 1.06× | Consumer-anchored aggregate: 19% of rows differ, Spearman 0.10. online/offline ratio 1.06×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 114 | `daf_cs_b_p28d_consumer_business_orders` | 🟨 medium | moderate_drift | `consumer_business_recency` | running | 32.5% | 10.9% | 4.6% | 0.15 | 3.05 | 3.63 | 1.19× | Consumer-anchored aggregate: 11% of rows differ, Spearman 0.15. online/offline ratio 1.19×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 115 | `daf_cs_b_p84d_consumer_business_orders` | 🟨 medium | moderate_drift | `consumer_business_recency` | running | 31.0% | 19.7% | 7.0% | 0.18 | 5.38 | 7.8 | 1.45× | Consumer-anchored aggregate: 20% of rows differ, Spearman 0.18. online/offline ratio 1.45×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 116 | `daf_cs_b_p168d_consumer_business_orders` | 🟨 medium | moderate_drift | `consumer_business_recency` | running | 29.4% | 25.9% | 8.4% | 0.19 | 7.71 | 12.5 | 1.62× | Consumer-anchored aggregate: 26% of rows differ, Spearman 0.19. online/offline ratio 1.62×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 117 | `daf_cs_b_p56d_global_search_views` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 26.3% | 41.6% | 5.3% | 0.29 | 10.5 | 16.3 | 1.55× | Consumer-anchored aggregate: 42% of rows differ, Spearman 0.29. online/offline ratio 1.55×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 118 | `caf_cs_st_p28d_consumer_store_clicks` | 🟨 medium | moderate_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 25.4% | 21.5% | 4.0% | 0.16 | 6.49 | 9.61 | 1.48× | Consumer-anchored aggregate: 22% of rows differ, Spearman 0.16. online/offline ratio 1.48×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 119 | `caf_cs_st_p6m_consumer_store_clicks` | 🟨 medium | moderate_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 24.9% | 40.3% | 6.8% | 0.21 | 17.5 | 34.6 | 1.98× | Consumer-anchored aggregate: 40% of rows differ, Spearman 0.21. online/offline ratio 1.98×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 120 | `caf_cs_b_p28d_consumer_business_clicks` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 23.7% | 21.3% | 4.2% | 0.17 | 3.89 | 5.42 | 1.39× | Consumer-anchored aggregate: 21% of rows differ, Spearman 0.17. online/offline ratio 1.39×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 121 | `caf_cs_b_p84d_consumer_business_clicks` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 22.6% | 34.6% | 6.1% | 0.20 | 7.13 | 11.8 | 1.66× | Consumer-anchored aggregate: 35% of rows differ, Spearman 0.20. online/offline ratio 1.66×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 122 | `caf_cs_b_consumer_business_recency_discounted_views` | 🟨 medium | moderate_drift | `consumer_business_recency_discounted_engagement_features` | running | 22.5% | 22.1% | 3.3% | 0.28 | 36.3 | 39.5 | 1.09× | Consumer-anchored aggregate: 22% of rows differ, Spearman 0.28. online/offline ratio 1.09×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 123 | `caf_cs_b_consumer_business_recency_discounted_clicks` | 🟨 medium | moderate_drift | `consumer_business_recency_discounted_engagement_features` | running | 22.5% | 20.7% | 3.8% | 0.17 | 3.03 | 3.93 | 1.30× | Consumer-anchored aggregate: 21% of rows differ, Spearman 0.17. online/offline ratio 1.30×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 124 | `caf_cs_b_consumer_business_recency_discounted_ctr` | 🟨 medium | moderate_drift | `consumer_business_recency_discounted_engagement_features` | running | 22.5% | 20.4% | 3.8% | 0.14 | 0.132 | 0.143 | 1.08× | Consumer-anchored aggregate: 20% of rows differ, Spearman 0.14. online/offline ratio 1.08×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 125 | `caf_cs_b_consumer_business_recency_discounted_orders` | 🟨 medium | moderate_drift | `consumer_business_recency_discounted_engagement_features` | running | 22.5% | 13.1% | 5.9% | 0.15 | 2.1 | 2.66 | 1.27× | Consumer-anchored aggregate: 13% of rows differ, Spearman 0.15. online/offline ratio 1.27×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 126 | `caf_cs_b_consumer_business_recency_discounted_cvr` | 🟨 medium | moderate_drift | `consumer_business_recency_discounted_engagement_features` | running | 22.5% | 12.9% | 5.9% | 0.30 | 0.0635 | 0.0746 | 1.17× | Consumer-anchored aggregate: 13% of rows differ, Spearman 0.30. online/offline ratio 1.17×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 127 | `caf_cs_st_p28d_consumer_store_views` | 🟨 medium | moderate_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 22.1% | 45.4% | 5.1% | 0.21 | 128 | 240 | 1.87× | Consumer-anchored aggregate: 45% of rows differ, Spearman 0.21. online/offline ratio 1.87×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 128 | `caf_cs_b_p84d_consumer_business_click_recency` | 🟨 medium | moderate_drift | `consumer_business_recency` | running | 22.0% | 30.8% | 10.9% | 0.13 | 23.7 | 22.2 | 0.94× | Consumer-anchored aggregate: 31% of rows differ, Spearman 0.13. online/offline ratio 0.94×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 129 | `caf_cs_st_p84d_organic_cvr` | 🟨 medium | moderate_drift | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 21.7% | 13.0% | 8.8% | 0.17 | 0.0568 | 0.077 | 1.36× | Consumer-anchored aggregate: 13% of rows differ, Spearman 0.17. online/offline ratio 1.36×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 130 | `caf_cs_b_p168d_consumer_business_clicks` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 21.6% | 43.1% | 7.1% | 0.21 | 10.7 | 19.1 | 1.79× | Consumer-anchored aggregate: 43% of rows differ, Spearman 0.21. online/offline ratio 1.79×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 131 | `baf_o1b_o2b_p84d_doubledash_engagement_num_bundled_v2` | 🟨 medium | moderate_drift | `doubledash_store_engagement_o1b_o2b` | running | 18.7% | 41.8% | 30.4% | — | — | — | — | Drift% 42. |
| 132 | `caf_cs_b_p84d_organic_cvr` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 17.1% | 15.5% | 9.8% | 0.17 | 0.053 | 0.0719 | 1.36× | Consumer-anchored aggregate: 16% of rows differ, Spearman 0.17. online/offline ratio 1.36×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 133 | `caf_cs_b_p84d_organic_conversions` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 17.1% | 14.5% | 9.8% | 0.15 | 6.02 | 9.8 | 1.63× | Consumer-anchored aggregate: 15% of rows differ, Spearman 0.15. online/offline ratio 1.63×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 134 | `caf_cs_b_p168d_organic_ctr` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 17.0% | 41.4% | 8.9% | 0.16 | 0.0949 | 0.126 | 1.32× | Consumer-anchored aggregate: 41% of rows differ, Spearman 0.16. online/offline ratio 1.32×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 135 | `caf_cs_b_p168d_organic_clicks` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 17.0% | 41.3% | 8.9% | 0.22 | 11.9 | 22 | 1.85× | Consumer-anchored aggregate: 41% of rows differ, Spearman 0.22. online/offline ratio 1.85×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 136 | `caf_cs_b_p168d_organic_cvr` | 🟨 medium | moderate_drift | `organic_consumer_business_aggregated_features_with_cutoffs` | running | 17.0% | 21.6% | 10.4% | 0.19 | 0.044 | 0.0621 | 1.41× | Consumer-anchored aggregate: 22% of rows differ, Spearman 0.19. online/offline ratio 1.41×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 137 | `saf_st_p6m_rx_subtotal_normalized_at_submarket` | 🟨 medium | moderate_drift | `aov_store_aggregated_features` | running | 14.7% | 37.8% | 21.3% | 0.10 | 0.967 | 1.01 | 1.04× | Consumer-anchored aggregate: 38% of rows differ, Spearman 0.10. online/offline ratio 1.04×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 138 | `daf_st_p84d_subtotal_avg` | 🟨 medium | moderate_drift | `store_rx_delivery_metrics` | running | 14.3% | 49.9% | 20.7% | — | — | — | — | Drift% 50. |
| 139 | `saf_st_price_range_v2` | 🟨 medium | moderate_drift | `store_property_features_v2` | running | 7.0% | 39.6% | 4.2% | 0.02 | 1.71 | 1.71 | 1.00× | Consumer-anchored aggregate: 40% of rows differ, Spearman 0.02. online/offline ratio 1.00×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 140 | `caf_cs_bizv_p28d_engagement_ctr` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.8% | 24.5% | 2.3% | — | — | — | — | Drift% 25. |
| 141 | `caf_cs_bizv_p28d_engagement_num_views` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.8% | 25.9% | 0.2% | — | — | — | — | Drift% 26. |
| 142 | `caf_cs_bizv_p28d_engagement_num_clicks` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.8% | 23.0% | 2.3% | — | — | — | — | Drift% 23. |
| 143 | `caf_cs_bizv_p28d_engagement_cvr` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.8% | 22.6% | 4.2% | — | — | — | — | Drift% 23. |
| 144 | `caf_cs_bizv_p28d_engagement_num_orders` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.8% | 20.5% | 4.2% | — | — | — | — | Drift% 21. |
| 145 | `daf_cs_bizv_p84d_consumer_business_vertical_order_recency` | 🟨 medium | moderate_drift | `consumer_business_vertical_engagement_history` | running | 3.7% | 16.4% | 6.2% | 0.14 | 49.4 | 20.2 | 0.41× | Consumer-anchored aggregate: 16% of rows differ, Spearman 0.14. online/offline ratio 0.41×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 146 | `caf_cs_bizv_p84d_consumer_business_vertical_click_recency` | 🟨 medium | moderate_drift | `consumer_business_vertical_engagement_history` | running | 3.7% | 14.3% | 7.7% | 0.15 | 36.8 | 13.4 | 0.36× | Consumer-anchored aggregate: 14% of rows differ, Spearman 0.15. online/offline ratio 0.36×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 147 | `caf_cs_bizv_p56d_engagement_num_views` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.7% | 25.9% | 0.2% | — | — | — | — | Drift% 26. |
| 148 | `caf_cs_bizv_p56d_engagement_ctr` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.7% | 24.5% | 2.3% | — | — | — | — | Drift% 25. |
| 149 | `caf_cs_bizv_p56d_engagement_num_clicks` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.7% | 23.0% | 2.3% | — | — | — | — | Drift% 23. |
| 150 | `caf_cs_bizv_p56d_engagement_cvr` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.7% | 22.6% | 4.2% | — | — | — | — | Drift% 23. |
| 151 | `caf_cs_bizv_p56d_engagement_num_orders` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.7% | 20.5% | 4.2% | — | — | — | — | Drift% 21. |
| 152 | `caf_cs_bizv_p84d_engagement_num_views` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.6% | 25.9% | 0.2% | — | — | — | — | Drift% 26. |
| 153 | `caf_cs_bizv_p84d_engagement_ctr` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.6% | 24.5% | 2.3% | — | — | — | — | Drift% 25. |
| 154 | `caf_cs_bizv_p84d_engagement_num_clicks` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.6% | 23.0% | 2.3% | — | — | — | — | Drift% 23. |
| 155 | `caf_cs_bizv_p84d_engagement_cvr` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.6% | 22.6% | 4.2% | — | — | — | — | Drift% 23. |
| 156 | `caf_cs_bizv_p84d_engagement_num_orders` | 🟨 medium | moderate_drift | `consumer_business_vertical_aggregate_engagement` | running, SOURCE-BUG (p28d/p56d/p84d byte-identical) | 3.6% | 20.5% | 4.2% | — | — | — | — | Drift% 21. |
| 157 | `caf_st_p12w_five_star_rate_lower_bound` | 🟨 medium | moderate_drift | `store_property_features_v2` | running (upper_bound has Sibyl publishing bug per Comment #7) | 3.1% | 40.7% | 0.5% | 0.07 | 0.846 | 0.861 | 1.02× | Consumer-anchored aggregate: 41% of rows differ, Spearman 0.07. online/offline ratio 1.02×. Next: check temporal cutoff used at serving vs training for consumer-side aggregates. |
| 158 | `daf_cs_st_p28d_consumer_store_orders` | 🟨 medium | low_rank_corr | `organic_consumer_store_aggregated_features_with_cutoffs` | running | 33.6% | 9.3% | 3.9% | 0.15 | 2.9 | 3.53 | 1.22× | Drift% 9, rank correlation low (Spearman 0.15). Online and offline disagree both in value and in rank. |
| 159 | `daf_cs_p184d_doubledash_num_orders` | 🟨 medium | stratified_AUC | `doubledash_consumer_engagement_features` | running | 22.1% | 2.1% | 0.5% | 1.00 | 7.5 | 7.45 | 0.99× | Values agree (Spearman 0.996); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.053). Worth a per-feature ablation. |
| 160 | `daf_cs_p184d_doubledash_num_views` | 🟨 medium | stratified_AUC | `doubledash_consumer_engagement_features` | running | 22.1% | 1.6% | 0.3% | 1.00 | 41.3 | 41.1 | 1.00× | Values agree (Spearman 1.000); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.073). Worth a per-feature ablation. |
| 161 | `baf_o1b_o2b_p84d_doubledash_engagement_num_imps_v2` | 🟨 medium | stratified_AUC | `doubledash_store_engagement_o1b_o2b` | running | 18.7% | 4.1% | 0.1% | 1.00 | 3.16e+04 | 3.18e+04 | 1.01× | Values agree (Spearman 1.000); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.092). Worth a per-feature ablation. |
| 162 | `daf_cs_dprg_p84d_organic_food_vertical_total_purchased_business_count` | 🟨 medium | stratified_AUC | `organic_consumer_daypart_food_vertical_repurchase_features_v2` | running | 6.1% | 7.4% | 0.3% | 0.99 | 8.11 | 8.06 | 0.99× | Values agree (Spearman 0.989); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.075). Worth a per-feature ablation. |
| 163 | `daf_cs_dprg_p84d_organic_food_vertical_repurchased_business_count` | 🟨 medium | stratified_AUC | `organic_consumer_daypart_food_vertical_repurchase_features_v2` | running | 6.1% | 5.2% | 0.5% | 0.99 | 5.74 | 5.7 | 0.99× | Values agree (Spearman 0.988); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.056). Worth a per-feature ablation. |
| 164 | `daf_cs_dprg_p365d_organic_food_vertical_total_purchased_business_count` | 🟨 medium | stratified_AUC | `organic_consumer_daypart_food_vertical_repurchase_features_v2` | running | 6.1% | 3.7% | 0.3% | 0.99 | 17.1 | 17 | 1.00× | Values agree (Spearman 0.994); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.066). Worth a per-feature ablation. |
| 165 | `daf_cs_dprg_p365d_organic_food_vertical_repurchased_orders` | 🟨 medium | stratified_AUC | `organic_consumer_daypart_food_vertical_repurchase_features_v2` | running | 6.1% | 3.3% | 0.3% | 0.99 | 52.5 | 51.8 | 0.99× | Values agree (Spearman 0.993); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.052). Worth a per-feature ablation. |
| 166 | `daf_cs_dprg_p365d_organic_food_vertical_repurchased_business_count` | 🟨 medium | stratified_AUC | `organic_consumer_daypart_food_vertical_repurchase_features_v2` | running | 6.1% | 3.1% | 0.3% | 0.99 | 8.85 | 8.8 | 0.99× | Values agree (Spearman 0.991); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.052). Worth a per-feature ablation. |
| 167 | `daf_cs_p84d_organic_food_vertical_repurchased_orders` | 🟨 medium | stratified_AUC | `organic_consumer_food_vertical_repurchase_features_v2` | running | 3.7% | 4.8% | 0.3% | 1.00 | 39 | 38.8 | 1.00× | Values agree (Spearman 0.999); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.068). Worth a per-feature ablation. |
| 168 | `daf_cs_p168d_organic_food_vertical_repurchased_orders` | 🟨 medium | stratified_AUC | `organic_consumer_food_vertical_repurchase_features_v2` | running | 3.7% | 2.6% | 0.2% | 1.00 | 71.7 | 71.5 | 1.00× | Values agree (Spearman 1.000); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.071). Worth a per-feature ablation. |
| 169 | `daf_cs_p10r_has_photos_avg_v2` | 🟨 medium | stratified_AUC | `organic_consumer_aggregated_features` | running | 3.6% | 4.8% | 0.4% | 0.99 | 0.768 | 0.769 | 1.00× | Values agree (Spearman 0.993); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.052). Worth a per-feature ablation. |
| 170 | `saf_cs_p84d_organic_ctr` | 🟨 medium | stratified_AUC | `organic_consumer_aggregated_features` | running | 2.9% | 2.7% | 0.3% | 0.99 | 0.0641 | 0.0641 | 1.00× | Values agree (Spearman 0.991); model behaves differently on real vs null cohorts online vs offline (stratified gap -0.101). Worth a per-feature ablation. |
| 171 | `saf_st_offers_pickup_v2` | 🟨 medium | stratified_AUC | `store_property_features_v2` | running | 0.8% | 0.0% | 24.4% | — | 1 | 1 | 1.00× | Real/null cohort AUC differs between online and offline; needs a per-feature ablation. |
| 172 | `saf_st_is_dashpass_available_v2` | 🟨 medium | stratified_AUC | `store_property_features_v2` | running | 0.8% | 0.0% | 2.4% | — | 1 | 1 | 1.00× | Real/null cohort AUC differs between online and offline; needs a per-feature ablation. |
| 173 | `daf_st_p4w_no_recent_orders_v2` | 🟨 medium | stratified_AUC | `store_rx_delivery_metrics` | running | 0.8% | 0.0% | 0.5% | — | — | — | — | Real/null cohort AUC differs between online and offline; needs a per-feature ablation. |
| 174 | `baf_o2b_sm_p84d_doubledash_engagement_num_imps_v2` | 🟨 medium | stratified_AUC | `doubledash_store_engagement_o2b_sm` | running | 0.2% | 0.4% | 0.0% | 1.00 | 4.77e+04 | 4.8e+04 | 1.00× | Values agree (Spearman 1.000); model behaves differently on real vs null cohorts online vs offline (stratified gap +0.230). Worth a per-feature ablation. |
| 175 | `caf_cs_st_is_saved` | 🟨 medium | aligned | `dimension_consumer_favorite_store_features` | running | 44.7% | 0.0% | 0.0% | — | — | — | — | Both-real cohort agrees, but Sibyl defaults 45% of online rows for this feature — partial serving-availability gap. Next: investigate which lookup-key subset is failing. |
| 176 | `saf_st_composite_score_v2` | 🟨 medium | aligned | `store_property_features_v2` | running | 44.5% | 1.0% | 19.5% | 0.02 | 4.04 | 4.03 | 1.00× | Both-real cohort agrees, but Sibyl defaults 44% of online rows for this feature — partial serving-availability gap. Next: investigate which lookup-key subset is failing. |
| 177 | `saf_st_offers_delivery_v2` | ⬜ none | aligned | `store_property_features_v2` | running | 0.8% | 0.0% | 0.0% | — | 1 | 1 | 1.00× | Both-real cohort agrees, but Sibyl defaults 1% of online rows for this feature — partial serving-availability gap. Next: investigate which lookup-key subset is failing. |
| 178 | `saf_st_has_photos_v2` | ⬜ none | aligned | `store_property_features_v2` | running | 0.8% | 0.0% | 21.3% | — | 1 | 1 | 1.00× | Both-real cohort agrees, but Sibyl defaults 1% of online rows for this feature — partial serving-availability gap. Next: investigate which lookup-key subset is failing. |
| 179 | `saf_st_inflates_price_v2` | ⬜ none | aligned | `store_property_features_v2` | running | 0.8% | 0.0% | 25.2% | — | 1 | 1 | 1.00× | Both-real cohort agrees, but Sibyl defaults 1% of online rows for this feature — partial serving-availability gap. Next: investigate which lookup-key subset is failing. |
| 180 | `saf_st_is_chain_v2` | ⬜ none | aligned | `store_property_features_v2` | running | 0.8% | 0.0% | 18.8% | — | 1 | 1 | 1.00× | Both-real cohort agrees, but Sibyl defaults 1% of online rows for this feature — partial serving-availability gap. Next: investigate which lookup-key subset is failing. |
