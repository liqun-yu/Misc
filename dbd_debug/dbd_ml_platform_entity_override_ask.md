# Platform ask: per-feature entity-source override (avoid o2 feature duplication)

**Audience:** ML Platform / Feature Platform (SPS, AIMS Feature V2), Sibyl.
**From:** DbD ranking (store ranker `dbd_ranker_v0_1`).
**TL;DR:** Two-entity ranking models (anchor + candidate) are currently forced to **duplicate every feature** under an `o2`-bound name purely to fetch it at the candidate's id — even though the value already exists in the online store keyed by the feature's natural id. We're asking whether the platform can support a **per-feature entity-source override** at fetch time so the same stored value can be fetched at a different request entity id, eliminating the duplication.

---

## The problem

The DbD post-checkout store ranker scores, for a cart **anchor** store (O1), a set of **candidate** bundle stores (O2). It uses the *same* features for both roles — e.g. `saf_st_composite_score_v2` (store attributes), `caf_cs_b_*` (consumer×business engagement). These are **symmetric**: the value for store/business X is identical regardless of whether X is the anchor or the candidate.

At serving, SPS resolves a feature's entity **from its name** (via AIMS / `entities.yaml`: `st`→`store_id`, `b`→`business_id`, …) and looks it up at the request's value for that entity. With two store entities in the request (`store_id`=O1, `o2_store_id`=O2), a feature named `saf_st_*` is fetched at **O1** — the anchor — so the model never sees the candidate's value. Result: ~0.25 offline/online AUC gap, ~133 candidate features mis-bound or defaulted online.

**The candidate's value already exists in the online store.** `saf_st_composite_score_v2` is materialized for *every* store at key `store_id_<id>`. The candidate store's value is sitting at `store_id_<candidate_id>` right now. Nothing needs to be recomputed or re-stored — the only thing missing is a way to **fetch it using the candidate's id**.

## Current workaround and why it doesn't scale

The only supported route today is to **mint an `o2`-named feature** (`saf_o2st_*`, bound to `o2_store_id`) and materialize it online. This means, for each two-entity model:

- A **duplicate feature definition** per source (recompute or republish).
- A **duplicate online copy** of values that already exist under the natural id.
- **Cross-team work**: the DbD store ranker's ~133 candidate features span cx_discovery and sponsored_listing shared sources. We confirmed those repos have **zero** o2-clone features — duplicating-for-o2 is *not* an established pattern there; it's a DbD-local workaround in the `doubledash/` dir.
- Permanent **maintenance coupling**: every generic feature now needs a parallel o2 twin kept in sync.

This multiplies feature count, storage, and compute for *every* anchor+candidate model (store ranker, item ranker, future bundle models). It does not scale.

## What we confirmed about current platform behavior

(Verified via internal docs — citations below.)

- **SPS** looks up aggregate features using the entity expected by the feature/model (`EntityId`); the lookup key is `<entity_identifier>_<id>`. There is **no generic per-feature request-time override** to use `o2_store_id` instead of `store_id`. [SPS Onboarding Guide]
- **AIMS / Feature V2** ties the lookup key to the feature's declared entity (`entities=[Entity(name=STORE_ID, …)]`); the entity key is prebuilt for the lookup, not overridden by naming another request entity. [Feature naming convention; AIMS Feature V2 thread]
- The closest existing mechanism, **Argo "Sibyl Fields Override"** (`feature_source_override_key`), remaps a Sibyl feature name to a different **Argo source field** (index/computed/context) at runtime — but it is **Argo-Search-specific** and does **not** override the online feature-store **entity** binding. [Argo Broker Sibyl Fields Override]
- `o1_/o2_` store/business/business_vertical are already **first-class entities** in the proto layer (`ml_entity.proto`) and emitted at serving by feed-service — so the candidate id is already present in the request; only the fetch-time binding is missing.

## Important: seqIndex already solves the multi-entity case — and is NOT deprecated

`seq_length` / `seq_index` lets a single feature be fetched at multiple entity ids of the *same* `entity_name` but different `seq_index` (anchor = 0, candidate = 1, … up to N=10) — **one feature definition, one materialized table, fetched at N entity ids**, no duplication. This is exactly the capability we want.

Critically, **this is a current, supported mechanism, not a legacy one:**
- The SPS Onboarding Guide's "How do sequence features work?" section documents it in full (multi-entity `store_id` at `seq_index` 0/1, N=10 cap); the page is **status: current, last updated 2026-02-04**. *(SPS Onboarding Guide)*
- `dnn_fireworks` models actively use it: the ETA team's `nv_dsd` model uses `feature_name@0…@19` (the fireworks seq expansion), and a teammate notes *"the DoubleDash model also uses the seq index to send a list of 2 store ids."* *(ask-ml-platform thread, Apr 2025)*

**I have NOT found any design doc stating FV2/fireworks deliberately deprecated seqIndex.** The accurate statement is narrower: the *specific* new DbD store ranker config does not use `seq_length` for these candidate features and relies on `o1_`/`o2_` entity names + duplicated features instead — an apparent per-model implementation choice, with no documented rationale I can cite, **not** a platform-level removal.

**This materially changes the framing of the ask below.** Before requesting any new platform capability, the first question is: **can the DbD store ranker simply adopt `seq_index`** (send `store_id`/`business_id`/`business_vertical_id` as a length-2 sequence: O1 at index 0, O2 at index 1) for these candidate features — reusing the *existing* generic features with no o2 clones and no platform change? If yes, the duplication problem is solved in-model and the rest of this doc is moot. The open question is whether this fireworks DCN+attention model can consume seq features for dense/embedding candidate inputs the way the item ranker does (TBD — needs model-code verification).

## The ask (only if the seqIndex route above is not viable for this model)

A **per-feature entity-source override** in the model / featureset feature spec:

> Fetch feature `F` (declared entity `E`) using the request's value for entity `E'`.

e.g. declare `saf_st_composite_score_v2` to be fetched at `o2_store_id` for this model, reusing the existing `store_id`-keyed online value. Conceptually this is the online-feature-store analogue of the existing Argo source-field override, extended to entity-key selection.

**Benefits:**
- **Zero** new features, **zero** recompute, **zero** duplicate online storage.
- Solves all ~133 DbD features (and any future anchor+candidate model) with **model config only** — no fabricator changes, no cross-team PRs.
- Removes the maintenance coupling of parallel o2 feature universes.

**Questions for the platform team:**
1. Is a fetch-time entity-source override feasible in SPS / AIMS Feature V2 (or already on the roadmap)?
2. If not, is there a blessed lighter-weight alternative to full feature duplication (e.g. a generic "republish/re-key" utility, or extending the Argo override to online-store entities)?
3. If duplication is the only supported path for now, can we at least standardize a **republish** pattern (re-key an existing materialized table, no recompute) rather than per-team pipeline copies?

## Interim plan (until the above is answered)

- **Hold** the cross-team o2-clone work (cx_discovery / sponsored_listing) — it's the wasteful pattern this ask is trying to avoid.
- DbD-owned fabricator PRs already open: **#30662** (materialize the existing, correctly-o2-named `baf_o1b_o2bizv` — legitimate regardless of this ask) and **#30661** (recompute `baf_o2b`/`baf_o2bizv_cs` — this is itself the workaround pattern; convert to republish or supersede if the override lands).
- Training is unblocked independently (read-time column alias on the existing O2-correct parquet; no serving dependency).

---

### Sources
- Sibyl Prediction Service Onboarding Guide — https://doordash.atlassian.net/wiki/spaces/Eng/pages/1417876560
- Feature Naming Convention (0.1) — https://docs.google.com/document/d/1sRjHPps8QIHdJyKsDIT_RHP3ndkpjDoVNdk4ki2DgfQ
- Argo Broker Sibyl Fields Override — https://doordash.atlassian.net/wiki/spaces/Eng/pages/5397315848
- AIMS / Feature V2 entity registration thread (Hebo, Zhe, Arjeta, Steffen, Artem, Atharva, Vasily, Pranav)
- AI/ML in Pedregal | Strategy (Source of Truth) — https://docs.google.com/document/d/1lphGMXEylBAd9E-s3CZQI68AoFeM-H_F5277xuaKzfk
- [RFC] ML Training Platform — https://docs.google.com/document/d/171EGJowTTxKHwGPSLgqWvPkr3GLNBXeZq-ej_QwDgO0
- Add entity name in deployment config in s3/AIMS — https://doordash.atlassian.net/browse/MSH2-995
