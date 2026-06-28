# DbD Store Ranker — Fixes & Latest Knowledge

**Model:** `dbd_ranker_v0_1_metaflow_1781132405` (DoubleDash post-checkout store ranker)
**Companion doc:** `dbd_root_cause_investigation.md` (diagnostics). This doc = the *fixes* + the system knowledge needed to act on them. Everything below is code- or data-verified, with code/doc pointers hyperlinked inline.

---

## 0. TL;DR

The ~0.25 AUC offline/online gap is driven by two serving-side defects in how candidate (O2) features reach the model. **Training data is correct (O2-keyed); the model is fine; the break is entirely in feature serving.**

1. **Bug 3 — non-materialization.** Several candidate features are computed offline (in Iceberg, used in training) but their Fabricator `feature_group` has **no `materialize_spec`**, so they are never published to the online store → Sibyl returns the default (0) on every request.
2. **Bug 4 — O1/O2 entity binding.** Candidate features named generically (`…_b_…`, `…_cs_b_…`, `…_bizv_…`, store-keyed) bind to the **anchor (O1)** entity at serving instead of the **candidate (O2)** entity, so every candidate in a request gets the cart-anchor's value → no candidate discrimination → AUC collapses.

The fix is **additive and requires one retrain**: point the model at O2-bound feature names (use existing `…_o2b_…` twins where present, clone new O2-bound sources where absent), add `materialize_spec` for the genuinely-unmaterialized features, and reuse the existing (already-O2-correct) training parquet with column renames — no training-data regeneration.

---

## 1. How the systems fit together — end to end, for `dbd_ranker_v0_1_metaflow_1781132405`

Six systems collaborate across two phases: an **offline phase** (build features → train → register), and an **online phase** (one ranking request). The key handoffs people get wrong: **Columbus picks *which model* to run; AIMS is the *catalog* that defines the model's feature list + each feature's entity binding; the online store (Redis/Boulder) holds the *values*; and at request time SPS resolves each feature's entity from its NAME and fetches per entity-id.** Walkthrough below, then a diagram, then a per-system reference.

### Offline phase (build → train → register)
1. **Fabricator** runs each source's `.py` Spark job (e.g. [`doubledash_store_engagement_b.py`](https://github.com/doordash/fabricator/blob/master/fabricator/repository/features/doubledash/doubledash_store_engagement_b.py)) → writes an Iceberg table `datalake.ml.<source>`. For every `feature_group` that has a `materialize_spec.sink`, Fabricator also **publishes the feature online** (Redis V2 / Boulder; CRDB being retired). No sink ⇒ offline-only.
2. **Fabricator CD → AIMS.** On merge to `master`, [`sync_repository.py`](https://github.com/doordash/fabricator-core/blob/master/fabricator_core/utils/script/sync_repository.py) calls `aims_stub.CreateFeature`/`CreateSource`. Each AIMS `Feature` records its **entity binding**, derived by parsing the feature *name* via [`entities.yaml`](https://github.com/doordash/fabricator/blob/master/fabricator/repository/entities.yaml) (`b`→`business_id`, `o2b`→`o2_business_id`, `o2b_sm`→`(o2_business_id, submarket_id)`, …). This is the contract that later tells everyone "this feature is keyed on entity X".
3. **Training data.** The v6 job ([`…full_training_instance_v6.py`](https://github.com/doordash/fabricator/blob/master/fabricator/repository/features/doubledash/store_ranking/training_instance/doubledash_store_ranking_full_training_instance_v6.py)) joins all the model's features onto the auction rows (setting `business_id/store_id = o2_*` before the join, so candidate features land at the O2 value) → training table → the parquet `dbd_ranking_data/*`.
4. **dd-models trains + registers.** The Metaflow flow trains the DCNv2+attention model on the parquet, writes `model.pt` + a generated Sibyl `*.config` (the feature list, each flat with `seq_length` defaulting to 1) to S3, then the [`register_model` step](https://github.com/doordash/dd-models/blob/master/models/consumer/ordering/featured_item/dbd_ranker_v0_metaflow/training_flow.py) → `register_sibyl_model_from_path` → [`aims.create_model_version_config(...)`](https://github.com/doordash/dd-models/blob/master/libraries/dd_models_lib/model_management/model_registry_client.py) registers the model in **AIMS** under model id `…1781132405` with its `FeatureDependency` list.
5. **Columbus** (predictor config) maps the predictor `doubledash_store_recommendation_ranking` + a `modelOverrideId` → the model id to actually run (plus shadow %, logging %). This is what a deploy/ramp flips.

### Online phase (one ranking request)
```
 Consumer app ──▶ feed-service (DbD post-checkout ranking stack)
                    │  BundleMLRanker → BundleMLFeatureGenerator.kt
                    │  builds ONE GetPrediction request containing:
                    │   • predictor_name = "doubledash_store_recommendation_ranking"
                    │   • modelOverrideId  (selects NEW path when == postCheckoutModelId)
                    │   • entity IDs (NOT values):
                    │       consumer_id
                    │       business_id  @seq0=O1  @seq1=O2      (generic, seq'd)
                    │       o1_business_id / o2_business_id       (name-distinct)
                    │       o1_store_id / o2_store_id, o1/o2_business_vertical_id …
                    ▼
              ┌──────────────────────────  Sibyl / SPS  ──────────────────────────┐
              │ 1. ROUTE: predictor_name + modelOverrideId ──▶ COLUMBUS            │
              │          └─▶ resolves model id  …1781132405                        │
              │ 2. LOAD model + its config (artifacts from S3, registered in AIMS) │
              │          config = feature list (+ seq_length, default 1)           │
              │ 3. FETCH features (feature-api module):                            │
              │      for each feature in the model config:                         │
              │        name ──parse──▶ entity identifier(s)  (o2b → o2_business_id)│
              │        take the request's id value(s) for that entity name         │
              │        key = "<identifier>_<idValue>"  e.g. o2_business_id=123     │
              │              composite: cs_<consumer>_o2b_<business>               │
              │        GET key ──▶ ONLINE STORE (Redis V2 / Boulder)               │
              │              hit  → value placed at its seqIndex                    │
              │              miss → feature default (e.g. 0)                        │
              │ 4. BUILD input: per feature, read positions 0..seq_length-1        │
              │              (seq_length=1 ⇒ only position 0)                       │
              │ 5. RUN model ──▶ score per candidate                               │
              │ 6. LOG to datalake.sps_prediction_logging.search                   │
              └───────────────────────────────────────────────────────────────────┘
                    │  scores per candidate store
                    ▼
              feed-service ranks the candidate stores → response
```

**So, to answer the "does SPS use Columbus then AIMS?" question directly:**
- **Columbus** = step 1, the routing decision (predictor + override → *which model id*). In the request hot path.
- **AIMS** = the build-time **catalog**: it defines the model's feature dependency list and each feature's entity binding (from the name convention). Those are baked into the model's loaded config + the name→entity resolution map; SPS does **not** make a per-request AIMS call — it uses the loaded config + name parsing ([`FeatureValue.kt` `IDENTIFIER_TO_ENTITY_NAME_MAP`](https://github.com/doordash/sibyl-prediction-service/blob/master/feature-api/src/main/kotlin/com/doordash/sibyl/feature/legacy/FeatureValue.kt)). So AIMS decides *how features are defined/bound*; that definition is consumed at model-load time, not re-queried per request.
- **Online store** (Redis/Boulder) = step 3, the *values*, keyed by `entity_identifier_idValue`.
- **feed-service** supplies the *entity id values* (and which model to override to); it never fetches feature values itself.

The critical chain for the bug: **feature name → (AIMS/name-parse) entity binding → which request entity-id is used → which online-store key is read.** A candidate feature named generically (`baf_b_*`) binds to `business_id` (the seq'd entity, position 0 = O1) → reads the anchor's row; renamed `baf_o2b_*` it binds to `o2_business_id` → reads the candidate's row. Same value in the store, different key resolved.

### Per-system reference
| System | Role for this model |
|---|---|
| **Fabricator** (`doordash/fabricator`) | Computes source tables (one `.py` → one Iceberg table) and, per `feature_group`, registers catalog features + optionally publishes them online via `materialize_spec.sink`. |
| **AIMS** (metadata catalog) | Source of truth for feature/source/model metadata. Feature **entity binding** ([`features.proto` `entities=18`](https://github.com/doordash/services-protobuf/blob/master/protos/aims/features.proto)) derived from the feature name via [`entities.yaml`](https://github.com/doordash/fabricator/blob/master/fabricator/repository/entities.yaml). No `seqIndex` on the Feature. Holds the model's `FeatureDependency` list. |
| **Fabricator → AIMS sync** | [`sync_repository.py`](https://github.com/doordash/fabricator-core/blob/master/fabricator_core/utils/script/sync_repository.py) `CreateFeature`/`CreateSource`, post-merge to `master`. |
| **dd-models** (`doordash/dd-models`) | Trains the model; uploads `model.pt` + Sibyl config to S3; [`register_model`](https://github.com/doordash/dd-models/blob/master/models/consumer/ordering/featured_item/dbd_ranker_v0_metaflow/training_flow.py) → [`aims.create_model_version_config`](https://github.com/doordash/dd-models/blob/master/libraries/dd_models_lib/model_management/model_registry_client.py). |
| **Columbus** | Sibyl predictor config — `(predictor_name, modelOverrideId)` → model id. Routing only; **not** a feature store. |
| **feed-service** (`doordash/feed-service`) | [`BundleMLFeatureGenerator.kt`](https://github.com/doordash/feed-service/blob/master/libraries/domain-util/src/main/kotlin/com/doordash/consumer/feed/domainUtil/bundle/jobs/ranking/BundleMLFeatureGenerator.kt) builds the request: entity IDs (o1/o2 named + generic seq'd) + predictor + override. Sends IDs, not values. |
| **Sibyl / SPS** (`doordash/sibyl-prediction-service`) | Routes via Columbus, loads the model, fetches feature values from the online store keyed by `entity_identifier_idValue` (feature-api module), runs the model, logs to `datalake.sps_prediction_logging.search`. |

**Online store note:** the sinks are **Redis V2 / Boulder** (CockroachDB `feature_store_prod_ug` is being retired) — not "Cassandra" (an earlier mislabel). Sink definitions: [`sinks.yaml`](https://github.com/doordash/fabricator/blob/master/fabricator/repository/sinks.yaml).

---

## 2. The two root causes (verified)

### Bug 3 — non-materialization (offline-only candidate features)
A `feature_group` with **no `materialize_spec`** is computed and written to the offline Iceberg table (and used in training) but **never pushed online**. At serving Sibyl can't find it → returns the default. Confirmed online-serving rates (2026-06-11, n=99,056):

| feature | online nonzero | cause |
|---|---:|---|
| `baf_cs_b_*` | **0%** | `cs_b` feature_group has no `materialize_spec` |
| `baf_b_sm_*` | **0%** | `b_sm` feature_group has no `materialize_spec` |
| `baf_cs_o2b_*` (twin) | 3.4% | materialized (genuine sparsity) |
| `baf_o2b_sm_*` (twin) | **99.9%** | materialized |

A model input that is offline-only is, by definition, a train/serve-skew bug — there is no legitimate reason for a served model to depend on a feature it can't read online.

### Bug 4 — O1/O2 entity binding
Candidate features must be looked up for the **candidate business (O2)**. Generic-named features (`…_b_…`, `…_bizv_…`, store-keyed) bind to the **anchor (O1)** entity, so every candidate in a request receives the *same* anchor value → zero discrimination → AUC collapse offline-vs-online even on identical rows+labels. The §3 "low source coverage" numbers for rows 11-17 were an artifact of probing the source at the O1 key; the sources are fully populated (see §4).

---

## 3. The O1/O2 position machinery (definitive, code-grounded)

This is the part that was most misunderstood. The mechanism is **`seqIndex`-based end-to-end**, and `feat_ind` is **training-only**:

1. **feed-service emits both positions.** [`BundleMLFeatureGenerator.kt:73-84`](https://github.com/doordash/feed-service/blob/master/libraries/domain-util/src/main/kotlin/com/doordash/consumer/feed/domainUtil/bundle/jobs/ranking/BundleMLFeatureGenerator.kt) emits `SIBYL_PREDICTOR_ENTITY_BUSINESS` at `seqIndex=0` (O1) and `seqIndex=1` (O2) (same for store/vertical). It **also** emits name-distinct entities `SIBYL_PREDICTOR_O2_BUSINESS_ID` (no seqIndex) at lines 86-87. So both binding styles are available in one request. Setter: `buildEntityId(...)` lines 577-584. feed-service sends **entity IDs, not feature values**.
2. **The fetch is ID-KEYED (the online-store key contains the entity id value).** This happens inside SPS's `feature-api` module (not a separate Feaser/router). The flow: group request id values by entity name (`FeatureStoreLegacy.kt:153-166`, `business_id → [O1val, O2val]`) → expand each feature into **one lookup request per id value** (`FeatureValue.kt:228-268`; composite keys take the cross-product, `:270-340`) → build the online-store key as `"${identifier}_${idValue}"`, e.g. `b_<businessId>` or composite `cs_<consumerId>_b_<businessId>` (`FeatureStoreClientRedisV2.kt:172-176`) → batch-GET from Redis V2 / Boulder (`FeatureStoreClientRedisV2.kt:20-106`; CRDB is being removed, `FeatureStore.kt:354-358`). So **O1's id and O2's id produce different keys → different fetched rows.** It is *not* one fetch returning a concatenated list.
3. **`seqIndex` only *places* each independently-fetched value into a slot.** After the per-id fetches return, `FeatureStoreClient.kt:952` computes the seqIndex for each (entityName, idValue) and `:990` writes `seqMap[seqIndex] = value`, producing the `populatedNumericalFeatures[name][seqIndex]` map. The consumption code I cited earlier (`Predictor.kt:517-522`, `ModelInput.kt:115-136`) then loops `0 until seqLength` over that map — it indexes, but the values were already fetched per-id upstream.
4. **SPS selects positions by `seq_length`, defaulting to 1.** [`ModelStore.kt:749`](https://github.com/doordash/sibyl-prediction-service/blob/master/prediction-service/src/main/kotlin/com/doordash/sibyl_prediction_service/config/ModelStore.kt): `val seqLength = (numericalFeature["seq_length"] as Int?) ?: 1`. **`seq_length=1` (the default) → only position 0 = O1 is read.** Missing positions → default value. So even though a `business_id`-keyed feature has BOTH O1 and O2 fetched (at seqIndex 0 and 1), a default-`seq_length` model consumes only seqIndex 0 (O1).
5. **`feat_ind` is training-time, not serving.** It selects which positions become model columns at training ([`dbd_store_ranker_development/utils/util.py:116-122`](https://github.com/doordash/dbd_store_ranker_development/blob/master/dbd_store_ranker_development/utils/util.py)): `feat_ind:[1]` → only the `name__1` (O2) column is wired. SPS never reads `feat_ind`.
6. **The old LGBM model worked** by declaring candidate features `seq_length: 2, feat_ind: [1]` ([`dbd_store_ranker_base_model-v02-05-lgbm.py:491,624-671`](https://github.com/doordash/sibyl-models/blob/master/training_scripts/new_verticals/discovery/doubledash/store_ranking/doubledash_store_recommendation_ranking/dbd_store_ranker_base_model-v02-05-lgbm/dbd_store_ranker_base_model-v02-05-lgbm.py)) — emit both positions, train on the O2 column.
7. **The new fireworks/metaflow model has NO position mechanism, and this is the fireworks-wide NORM (not specific to this model).** Its [`feature_config.py`](https://github.com/doordash/dd-models/blob/master/models/consumer/ordering/featured_item/dbd_ranker_v0_metaflow/config/feature_config.py) declares O1/O2 as distinct *named* features (`o1_business_id_int` L35 vs `o2_business_id_int` L231) and carries no `seq_length`/`feat_ind`. Its Sibyl config generator ([`dnn_fireworks/utils/sibyl_model_config_generator.py:66-91`](https://github.com/doordash/dnn_fireworks/blob/master/utils/sibyl_model_config_generator.py)) writes features flat with no `seq_length` (the `override_seq_length` method at L111-121 is **never called** — [`train_common.py:531-538`](https://github.com/doordash/dd-models/blob/master/models/consumer/ordering/featured_item/dbd_ranker_v0_metaflow/train_common.py)). So every feature defaults to `seqLength=1` → SPS reads position 0 (O1). **The model distinguishes O1/O2 only by entity name** — and that is how virtually every fireworks/DNN model works:
   - `override_seq_length` has exactly **one** production caller across `doordash/*` — `dnn_fireworks/model/nvml/route_pick_score/run_train.py` (`seq_length=20` for a genuine NV item-list *sequence*, not an O1/O2 pair). Every other DNN model (sr_v3, ur_v9, store_mtml, sl_search, …) writes flat / no override.
   - `o2_store_id` is a **first-class platform ML entity** (`doordash/pedregal:libraries/python/dash/ml_entity/entity/merchant/ml_o2_store_id.py`, `ENTITY_TYPE_ID=56`) — the serving platform itself defines O2 as a named entity, and feed-service emits `o1_business_id`/`o2_business_id`/`o1_store_id`/`o2_store_id` as named entities.

**Design provenance (no single platform RFC; it's a framework default + the LGBM→DNN migration).** The `seq_length`+`feat_ind` positional idiom is the **deprecated LGBM** mechanism (`sibyl-models`, `dbd_store_ranker_development`). The DNN migration's substantive change was **dropping positions entirely** in favor of distinct entity names — which the fireworks `SibylModelConfigGenerator` makes the default (flat serialization, `seq_length` defaults to 1 in [`aims_model_helpers.py`]). Note the named `o1_*`/`o2_*` store entities actually *predate* the DNN; the old LGBM used BOTH named o1/o2 store entities AND `seq_length`/`feat_ind` for its metric features. There is no platform-wide doc mandating name-based binding (the Sibyl onboarding guide still documents the older `seq_length`/`seq_index` mechanism as generally available); it's documented per-model — see [DbD Post Checkout Rankers (WIP)](https://doordash.atlassian.net/wiki/spaces/DOUB/pages/5623611394) and the DbD Master Doc.

**⚠ The `seq_length`/`feat_ind` position trick does NOT apply to our new model.** It was the *old LGBM* mechanism. The new `dbd_ranker_v0_metaflow` model has no `seq_length`/`feat_ind` anywhere — so we will **not** fix Bug 4 by setting positions. SPS will always read position 0 (O1) for any generic-named feature regardless. The fix is purely O2-naming.

**Consequence + why the fix is clean:** because the new model is name-driven, the correct and idiomatic fix is to **name candidate features `…_o2b_…`** so they bind to the `o2_business_id` entity (which feed-service emits, and which resolves to the candidate directly — no seq logic). This is exactly how `o2_business_id_int` already works in the model. **There is no need to set `seq_length`/`feat_ind` or change Sibyl/feed-service.**

---

## 4. Key verified findings (data)

- **`cs_b` ≡ `cs_o2b` exactly.** [`cs_b.py`](https://github.com/doordash/fabricator/blob/master/fabricator/repository/features/doubledash/doubledash_store_engagement_cs_b.py) and [`cs_o2b.py`](https://github.com/doordash/fabricator/blob/master/fabricator/repository/features/doubledash/doubledash_store_engagement_cs_o2b.py) are identical computations (both `grouping_keys=["consumer_id","o2_business_id"]`); they differ only in `entity_renames` (`o2_business_id → business_id` vs kept). Source-table join: **64,612,490 rows, 100% value-identical**. Same for `b_sm ≡ o2b_sm`.
- **Training parquet is O2-correct.** In `s3://doordash-datalake/prod/ml/training_datasets/dbd_ranking_data/train` (date-partitioned, Jan–Mar 2026, 227 cols): `baf_cs_b == baf_cs_o2b` and `baf_b_sm == baf_o2b_sm` for all 5,646,419 rows. Cross-family spot-check of the symmetric `caf_cs_b` family vs its source `dimension_consumer_business_engagement_aggregated_metrics`: parquet matched **O2-key 81.1%** vs **O1-key 11.5%**. The v6 builder ([`doubledash_store_ranking_full_training_instance_v6.py:1126-1149`](https://github.com/doordash/fabricator/blob/master/fabricator/repository/features/doubledash/store_ranking/training_instance/doubledash_store_ranking_full_training_instance_v6.py)) sets `business_id/store_id = o2_*` before a single `get_features(all_features)` call → uniform O2 for all features.
- **Candidate `_b_` features are O2-context by definition (DbD-engagement) or symmetric (generic).** All `doubledash_store_engagement_*` jobs group on `o2_business_id` over [`base_instance_v2`](https://github.com/doordash/fabricator/blob/master/fabricator/repository/features/doubledash/store_ranking/base_instance/doubledash_store_ranking_base_instances_v2.py) and aggregate candidate events (`is_store_bundled`, `is_store_engaged`) — intrinsically O2. The generic `caf_cs_b_*`/`daf_cs_b_*` (from `dimension_consumer_business_engagement_aggregated_metrics`) are symmetric `(consumer, business)` stats. Either way, candidate-side serving must look up O2.
- **`o1b_o2b` works; `o1b_o2bizv` is broken online.** `baf_o1b_o2b_*`: training 94.5%, **online 80.2%** (paired on two unambiguous entities → correctly served). `baf_o1b_o2bizv_*`: training 32.7%, **online 0%** — **not because the entity is missing** (feed-service *does* emit `o2_business_vertical_id`, see below) but because the Fabricator source `doubledash_store_engagement_o1b_o2bizv` has **no `materialize_spec`** (Bug 3, not published online).
- **feed-service already emits all the O1/O2 named entities the fix needs — NO feed-service change is required.** [`BundleMLFeatureGenerator.kt`](https://github.com/doordash/feed-service/blob/master/libraries/domain-util/src/main/kotlin/com/doordash/consumer/feed/domainUtil/bundle/jobs/ranking/BundleMLFeatureGenerator.kt) (both old and new paths) emits, alongside the seq'd generic entities (`business_id`/`store_id`/`business_vertical_id` at seqIndex 0/1), the name-distinct entities ([`Constants.kt`](https://github.com/doordash/feed-service/blob/master/libraries/domain-util/src/main/kotlin/com/doordash/consumer/feed/domainUtil/bundle/jobs/ranking/Constants.kt)): `o1_business_id`/`o2_business_id` (L231-232/86-87), `o1_business_vertical_id`/`o2_business_vertical_id` (L249/255/104/110), `o1_store_id`/`o2_store_id` (L259-260/116/120). Confirmed live by `baf_o2b_sm_*` serving 99.9%. So o2-named clone features bind to entities feed-service already sends.
- **The model is single-tower DCNv2 + cross-attention** ([`ranker_dnn.py`](https://github.com/doordash/dd-models/blob/master/models/consumer/ordering/featured_item/dbd_ranker_v0_metaflow/modules/ranker_dnn.py)), not two-tower. Query+candidate embeddings + attention output are concatenated → DCNv2 → MLP → one score. A constant (anchor-valued) candidate feature therefore kills within-request ranking.

---

## 5. The fix

### Tier 1 — drop the 6 duplicate engagement features (cheapest, no new source)
`baf_cs_b_*` (3) and `baf_b_sm_*` (3) are exact duplicates of `baf_cs_o2b_*` / `baf_o2b_sm_*`, which are already in the model, already in the parquet, and already materialized online (serve 99.9% / 3.4%). **Remove the 6 generic names from [`feature_config.py`](https://github.com/doordash/dd-models/blob/master/models/consumer/ordering/featured_item/dbd_ranker_v0_metaflow/config/feature_config.py)'s `CANDIDATE_FEATURES`.** Zero signal lost; 6 always-0 noise inputs removed. (Requires the Tier-2 retrain.)

### Tier 2 — migrate the remaining candidate features to O2-bound names + retrain
For the ~90 candidate features bound to generic `business_id` / `business_vertical_id` / `store_id` (the 39 `baf_b_*`, 33 `caf/daf_cs_b_*`, `bizv_cs`/`cs_bizv` families, ~43 store features, and `o1b_o2bizv`):
1. **Create additive O2-bound clone sources** in Fabricator — copy the existing `.py`, change the `entity_renames` to keep `o2_business_id` / `o2_store_id` / `o2_business_vertical_id`, give them `…_o2b_…` / `…_o2_store_…` names, and add `materialize_spec: {override_safe_mode: true, sink: [feature_store_prod_ug]}`. This is the established pattern (`cs_o2b`, `o2b_sm`, `o1b_o2b` already exist this way).
2. **Rename the value-identical columns in the existing training parquet** (`caf_cs_b_X → caf_cs_o2b_X`, etc.). Safe because the parquet already holds the O2 value (§4) and the clone is byte-identical to the generic source — **no training-data regeneration**. ⚠ Before renaming any non-engagement family, spot-check `parquet-column == clone-source@O2` for a sample (one query per family), as done for `caf_cs_b` (81% O2 / 11.5% O1).
3. **Update [`feature_config.py`](https://github.com/doordash/dd-models/blob/master/models/consumer/ordering/featured_item/dbd_ranker_v0_metaflow/config/feature_config.py)** to the new O2-bound names; retrain once.

### Tier 3 — materialize the genuinely-unmaterialized features (true Bug 3)
For the §3 rows 1-5 (e.g. `caf_st_p12w_five_star_rate_upper_bound`, `daf_st_p84d_doubledash_business_position_weighted_imp2conv_over_avg`) — source has the data offline (95-100%) but the `feature_group` lacks `materialize_spec`: **add `materialize_spec.sink`** to the existing feature_group (no new source). For row 18 (`daf_st_offers_store_promo_quality_score_v2_cai`, upstream stopped 2026-05-04): revive the upstream or drop the feature.

---

## 6. Retrain & backfill plan

- **Training data:** reuse the existing April parquet. Tier-1 features need no parquet change (the `o2b` columns already exist). Tier-2 features need an in-place **column rename** (value-preserving — proven O2). No regeneration. v6 itself stopped 2026-03-24, so regeneration isn't even available — reuse is the only path, and it's safe.
- **Backfill (your framing is correct):**
  - *Already-exists-offline, just mis-named / not-online* → **no backfill** (Tier 1; Tier 3 just adds a sink to an existing offline feature).
  - *Genuinely-new named feature* (Tier-2 o2 clones) → **offline first** (backfill the clone source over any dates you need beyond the parquet), **then online** (materialize via the sink — nightly upload, or a triggered backfill for immediate live coverage). The values aren't new (identical to the generic source); you're republishing the same numbers under the O2 name.
- **Retrain:** one retrain covers all `feature_config.py` changes. No Sibyl or feed-service change is required — the fix is purely feature-naming + materialization + model config.

---

## 7. Constraints / do-nots

- **Do NOT mutate or delete `doubledash_store_engagement_cs_b.py`** (or rename its column). It's consumed by many active models — `dd-models` `store_ranking_feature_config_v1`, `lucent-training` `doubledash_store_ranker_v6_3` + item rankers, `dbd_store_ranker_development` v10–v12 + `model_config_v21`, `sibyl-models` LGBM v02-01/02-02, and 6 fabricator training instances. Mutating it breaks their training/serving. The fix is **additive** (new O2 sources / use existing twins); other models migrate on their own cadence.
- **Don't try to fix this via `seq_length`/`feat_ind` on the new model** — the new fireworks path has no position mechanism and relies on entity names. Naming features `…_o2b_…` is the correct lever.
- **Don't "publish under the cs_b name but bind to O2"** — the AIMS entity binding is derived from the feature *name*, so the name must carry `o2b`.

---

## 8. Summary — by feature group

Organized by **feature group** (not by bug). "Intrinsically O2?" = is the feature *value* computed in the candidate (O2) context (Yes), a generic stat that's the same for any business/store role (No / "symmetric"), or unrelated to the O1/O2 axis (n/a). Counts are from `feature_config.py CANDIDATE_FEATURES` (≈172) + 8 QUERY. Full per-feature detail in `dbd_root_cause_investigation.md`.

**"Duplicate" (rows below)** means the model config lists the *same underlying feature values under two different feature names* — a generic-bound name and its O2-bound twin — both in `CANDIDATE_FEATURES`. Verified identical: `baf_cs_b_* ≡ baf_cs_o2b_*` and `baf_b_sm_* ≡ baf_o2b_sm_*` (64.6M source rows / 5.6M parquet rows, 100% value-match). Online, the generic copy serves **0%** (never materialized) while the o2-bound twin serves (3.4% / 99.9%). So the generic copies are dead inputs (always 0) duplicating signal the twins already carry — dropping them loses nothing.

| Feature group (examples) | # | Intrinsically O2? | Current serving | Diagnostic | Fix | Retrain |
|---|---:|---|---|---|---|---|
| **`baf_b_*`** DbD business engagement — ctr/cvr/imps/loads/bundled × p7/28/84d | 39 | **Yes** (grouped on `o2_business_id` over candidate events) | materialized but bound to `business_id` → **O1** | Bug 4 binding | clone `baf_o2b_*` (o2-bound) + `materialize_spec` | yes (T2) |
| **`baf_cs_b_*`** DbD consumer-business engagement | 3 | **Yes** | **0%** (not materialized); twin `baf_cs_o2b_*` already serves | Bug 4 + duplicate | **drop**; keep existing `baf_cs_o2b_*` | yes (T1) |
| **`baf_b_sm_*`** DbD business-submarket engagement | 3 | **Yes** | **0%**; twin `baf_o2b_sm_*` serves 99.9% | Bug 4 + duplicate | **drop**; keep existing `baf_o2b_sm_*` | yes (T1) |
| **`baf_bizv_cs_*`** DbD bizv-consumer engagement | 3 | **Yes** | bound to `business_vertical_id` → **O1** | Bug 4 binding | clone o2bizv-bound + `materialize_spec` | yes (T2) |
| **`baf_o1b_o2bizv_*`** DbD anchor×candidate-bizv | 3 | **Yes** (O1×O2 pair) | **0%** (source not materialized — entity *is* emitted) | Bug 3 (no `materialize_spec`) | add `materialize_spec` to the o1b_o2bizv source | no (serving-only) |
| **`baf_o1b_o2b_*`** DbD anchor×candidate-business | 3 | **Yes** (O1×O2 pair) | **80.2%** — correct | — none ✓ | — | — |
| **`baf_cs_o2b_*`, `baf_o2b_sm_*`** already O2-bound | 6 | **Yes** | serve (3.4% / 99.9%) | — none ✓ | keep (the migration targets) | — |
| **`caf_cs_b_*` / `daf_cs_b_*`** generic consumer-business — clicks/views/orders/recency/organic ctr-cvr/global_search | ~33 | **No** (symmetric `(cs,biz)` stat) | bound to `business_id` → **O1** | Bug 4 binding | clone `…_cs_o2b_…` + `materialize_spec` | yes (T2) |
| **`caf_cs_bizv_*` / `daf_cs_bizv_*`** consumer-business-vertical engagement (+`…cng_vertical_num_orders`) | ~19 | **No** (symmetric) | bound to `business_vertical_id` → **O1** | Bug 4 binding | clone o2bizv-bound + `materialize_spec` | yes (T2) |
| **Store-keyed**: `saf_st_*` (attrs), `caf_st_*` (ratings), `daf_st_*` (discovery/attach), `caf_cs_st_*` (cx-store), `saf_dprg_st_*` | ~43 | **No** (symmetric store stat) | bound to `store_id` → **O1**; subset also unmaterialized | Bug 4 binding **+ Bug 3** for `caf_st_p12w_five_star_rate_upper_bound`, `daf_st_p84d_..._imp2conv_over_avg` (0% online, source has data) | clone `…_o2_store_…` + `materialize_spec`; for the Bug-3 subset add sink to existing group | yes (T2); Bug-3 subset no-retrain |
| **User-history sequence** `caf_cs_p84d_organic_converted_business_id_…_sequence` | 1 | n/a (consumer history; feeds attention) | **0%** (Sibyl-side, not materialized) | Bug 3 | add `materialize_spec` | no |
| **Consumer-only** `daf_cs_p10r/p184d_*`, `cs_dprg_*` | ~11 | n/a (consumer-keyed; `consumer_id` unambiguous) | served | — none | — | — |
| **`cos_sim_*`** (cx_o2 / o1_o2 cosine sims) | ~10 | n/a (computed in-model from embeddings) | in-graph; 5 depend on unmaterialized embeddings | **Bug 2** (5 cos_sim) | materialize the 5 source embeddings | with T2 |
| **QUERY-side** `o1_business_id_int`, `o1_business_vertical_id`, `o1_primary_category_id`, `o1_is_caviar`, `submarket_id`, `hour_of_day`, `day_of_week` | 8 | n/a (O1 by name / request context) | defaulted in NEW path | **Bug 1** | feed-service NEW-path gate + `_int` naming (root-cause doc §1) | — |

**How the groups map to the fix tiers:** **T1** = drop the 2 duplicate groups (`baf_cs_b_*`, `baf_b_sm_*`). **T2** = clone O2-bound sources for the generic-bound groups (`baf_b_*`, `baf_bizv_cs_*`, `caf/daf_cs_b_*`, `caf/daf_cs_bizv_*`, store-keyed) + rename the (already-O2) parquet columns + retrain. **T3** = add `materialize_spec` to the genuinely-unmaterialized groups (history sequence; `baf_o1b_o2bizv_*`; the store Bug-3 subset) — no retrain.

**feed-service requires NO change.** All O1/O2 named entities (`o1_/o2_business_id`, `o1_/o2_store_id`, `o1_/o2_business_vertical_id`) are already emitted by [`BundleMLFeatureGenerator.kt`](https://github.com/doordash/feed-service/blob/master/libraries/domain-util/src/main/kotlin/com/doordash/consumer/feed/domainUtil/bundle/jobs/ranking/BundleMLFeatureGenerator.kt) in both paths (confirmed live by `baf_o2b_sm_*` @ 99.9%). The o2-named clone features bind to entities that are already sent. The seq-length-2 generic emission (`business_id`/`store_id` at seqIndex 0/1) is **unused by the new model** but must be **kept** — legacy LGBM rankers + item rankers on the same predictor still consume it via `seq_length`/`feat_ind`.

**Verified non-issues:** training data is O2-correct across all groups (no relabeling); model architecture is sound (single-tower DCNv2+attention); `cs_b ≡ cs_o2b` exactly; `baf_o1b_o2b_*` serves correctly (80.2%); consumer-only and QUERY-side O1 features are not part of the O1/O2 binding bug.

**One-retrain plan:** T1 + T2 land together in a single retrain on the reused April parquet; T3 `materialize_spec` PRs are independent serving-only changes that need no retrain.
