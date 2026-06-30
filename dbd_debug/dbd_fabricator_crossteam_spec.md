# DbD store ranker — cross-team Fabricator o2-clone spec

Companion to `dbd_fixes.md`. The DbD-owned sources are handled in fabricator PR **#30661** (`baf_o2b`, `baf_o2bizv_cs`) and PR **#30662** (`baf_o1b_o2bizv` materialize). The **91 features below live in cx_discovery / sponsored_listing shared sources** — they need o2-bound clones authored + reviewed by the owning team (can't be merged unilaterally).

## The change pattern (same as DbD PR #30661)

For each source: add an **additive** o2-bound clone (copy the `.py`, change `entity_renames` to keep the `o2_*` entity → the prefix becomes `o2b`/`o2bizv`/`o2st`), register a new `sources:` + `feature_groups:` entry with `materialize_spec` (`override_safe_mode: true`, `sink: [feature_store_prod_ug]`). **Do not mutate the generic source** — other models still use it. Reuse the already-registered short-names (`o2b`/`o2bizv`/`o2st`) → no `entities.yaml`/proto change. Per-pair aggregations (consumer×o2-store, consumer×o2-bizv) are genuine recomputes — budget compute for those.

**Total cross-team features: 91** across 10 families.

| Family | # | Owner | Source file(s) | Keep entity | New name |
|---|--:|---|---|---|---|
| `caf_cs_b_*` | 19 | cx_discovery | `consumer_business_engagement_aggregated_metrics.py`, `consumer_business_recency_discounted_engagement_features.py` | `o2_business_id` | `caf_cs_o2b_*` |
| `caf_cs_bizv_*` | 16 | cx_discovery | `discovery_noramlized_features/consumer_business_vertical_*` | `o2_business_vertical_id` | `caf_cs_o2bizv_*` |
| `daf_cs_b_*` | 14 | cx_discovery | `consumer_business_engagement_aggregated_metrics.py` | `o2_business_id` | `daf_cs_o2b_*` |
| `saf_st_*` | 14 | cx_discovery | `store_property_features_v2.py` | `o2_store_id` | `saf_o2st_*` |
| `daf_cs_st_*` | 8 | cx_discovery | `store_engagement_features.py` (consumer-store) | `o2_store_id` | `daf_cs_o2st_*` |
| `daf_st_*` ⚠ subset also Bug 3 (e.g. `daf_st_p84d_..._imp2conv_over_avg`) | 8 | cx_discovery | `store_engagement_features.py` | `o2_store_id` | `daf_o2st_*` |
| `caf_cs_st_*` | 6 | cx_discovery | `store_engagement_features.py` (consumer-store) | `o2_store_id` | `caf_cs_o2st_*` |
| `caf_st_*` ⚠ also Bug 3 (0% online — source has data but no `materialize_spec`) | 2 | cx_discovery | `store_property_features_v2.py` | `o2_store_id` | `caf_o2st_*` |
| `daf_cs_bizv_*` | 2 | cx_discovery | `discovery_noramlized_features/consumer_business_vertical_*` | `o2_business_vertical_id` | `daf_cs_o2bizv_*` |
| `saf_dprg_st_*` | 2 | sponsored_listing | `rx_ads_quality/.../organic_store_daypart_aggregated_features.py` | `o2_store_id` | `saf_dprg_o2st_*` |

## Full feature rename map (generic → o2)

### `caf_cs_b_*` — cx_discovery — `consumer_business_engagement_aggregated_metrics.py`, `consumer_business_recency_discounted_engagement_features.py`
- `caf_cs_b_consumer_business_recency_discounted_clicks` → `caf_cs_o2b_consumer_business_recency_discounted_clicks`
- `caf_cs_b_consumer_business_recency_discounted_ctr` → `caf_cs_o2b_consumer_business_recency_discounted_ctr`
- `caf_cs_b_consumer_business_recency_discounted_cvr` → `caf_cs_o2b_consumer_business_recency_discounted_cvr`
- `caf_cs_b_consumer_business_recency_discounted_orders` → `caf_cs_o2b_consumer_business_recency_discounted_orders`
- `caf_cs_b_consumer_business_recency_discounted_views` → `caf_cs_o2b_consumer_business_recency_discounted_views`
- `caf_cs_b_p168d_consumer_business_clicks` → `caf_cs_o2b_p168d_consumer_business_clicks`
- `caf_cs_b_p168d_consumer_business_views` → `caf_cs_o2b_p168d_consumer_business_views`
- `caf_cs_b_p168d_organic_clicks` → `caf_cs_o2b_p168d_organic_clicks`
- `caf_cs_b_p168d_organic_ctr` → `caf_cs_o2b_p168d_organic_ctr`
- `caf_cs_b_p168d_organic_cvr` → `caf_cs_o2b_p168d_organic_cvr`
- `caf_cs_b_p1y_order_recency` → `caf_cs_o2b_p1y_order_recency`
- `caf_cs_b_p28d_consumer_business_clicks` → `caf_cs_o2b_p28d_consumer_business_clicks`
- `caf_cs_b_p28d_consumer_business_view_recency` → `caf_cs_o2b_p28d_consumer_business_view_recency`
- `caf_cs_b_p28d_consumer_business_views` → `caf_cs_o2b_p28d_consumer_business_views`
- `caf_cs_b_p84d_consumer_business_click_recency` → `caf_cs_o2b_p84d_consumer_business_click_recency`
- `caf_cs_b_p84d_consumer_business_clicks` → `caf_cs_o2b_p84d_consumer_business_clicks`
- `caf_cs_b_p84d_consumer_business_views` → `caf_cs_o2b_p84d_consumer_business_views`
- `caf_cs_b_p84d_organic_conversions` → `caf_cs_o2b_p84d_organic_conversions`
- `caf_cs_b_p84d_organic_cvr` → `caf_cs_o2b_p84d_organic_cvr`

### `caf_cs_bizv_*` — cx_discovery — `discovery_noramlized_features/consumer_business_vertical_*`
- `caf_cs_bizv_p28d_engagement_ctr` → `caf_cs_o2bizv_p28d_engagement_ctr`
- `caf_cs_bizv_p28d_engagement_cvr` → `caf_cs_o2bizv_p28d_engagement_cvr`
- `caf_cs_bizv_p28d_engagement_num_clicks` → `caf_cs_o2bizv_p28d_engagement_num_clicks`
- `caf_cs_bizv_p28d_engagement_num_orders` → `caf_cs_o2bizv_p28d_engagement_num_orders`
- `caf_cs_bizv_p28d_engagement_num_views` → `caf_cs_o2bizv_p28d_engagement_num_views`
- `caf_cs_bizv_p56d_engagement_ctr` → `caf_cs_o2bizv_p56d_engagement_ctr`
- `caf_cs_bizv_p56d_engagement_cvr` → `caf_cs_o2bizv_p56d_engagement_cvr`
- `caf_cs_bizv_p56d_engagement_num_clicks` → `caf_cs_o2bizv_p56d_engagement_num_clicks`
- `caf_cs_bizv_p56d_engagement_num_orders` → `caf_cs_o2bizv_p56d_engagement_num_orders`
- `caf_cs_bizv_p56d_engagement_num_views` → `caf_cs_o2bizv_p56d_engagement_num_views`
- `caf_cs_bizv_p84d_consumer_business_vertical_click_recency` → `caf_cs_o2bizv_p84d_consumer_business_vertical_click_recency`
- `caf_cs_bizv_p84d_engagement_ctr` → `caf_cs_o2bizv_p84d_engagement_ctr`
- `caf_cs_bizv_p84d_engagement_cvr` → `caf_cs_o2bizv_p84d_engagement_cvr`
- `caf_cs_bizv_p84d_engagement_num_clicks` → `caf_cs_o2bizv_p84d_engagement_num_clicks`
- `caf_cs_bizv_p84d_engagement_num_orders` → `caf_cs_o2bizv_p84d_engagement_num_orders`
- `caf_cs_bizv_p84d_engagement_num_views` → `caf_cs_o2bizv_p84d_engagement_num_views`

### `daf_cs_b_*` — cx_discovery — `consumer_business_engagement_aggregated_metrics.py`
- `daf_cs_b_p168d_consumer_business_orders` → `daf_cs_o2b_p168d_consumer_business_orders`
- `daf_cs_b_p168d_global_search_clicks` → `daf_cs_o2b_p168d_global_search_clicks`
- `daf_cs_b_p168d_global_search_conversions` → `daf_cs_o2b_p168d_global_search_conversions`
- `daf_cs_b_p168d_global_search_ctr` → `daf_cs_o2b_p168d_global_search_ctr`
- `daf_cs_b_p168d_global_search_cvr` → `daf_cs_o2b_p168d_global_search_cvr`
- `daf_cs_b_p168d_global_search_views` → `daf_cs_o2b_p168d_global_search_views`
- `daf_cs_b_p28d_consumer_business_orders` → `daf_cs_o2b_p28d_consumer_business_orders`
- `daf_cs_b_p56d_global_search_clicks` → `daf_cs_o2b_p56d_global_search_clicks`
- `daf_cs_b_p56d_global_search_conversions` → `daf_cs_o2b_p56d_global_search_conversions`
- `daf_cs_b_p56d_global_search_ctr` → `daf_cs_o2b_p56d_global_search_ctr`
- `daf_cs_b_p56d_global_search_cvr` → `daf_cs_o2b_p56d_global_search_cvr`
- `daf_cs_b_p56d_global_search_views` → `daf_cs_o2b_p56d_global_search_views`
- `daf_cs_b_p84d_consumer_business_order_recency` → `daf_cs_o2b_p84d_consumer_business_order_recency`
- `daf_cs_b_p84d_consumer_business_orders` → `daf_cs_o2b_p84d_consumer_business_orders`

### `saf_st_*` — cx_discovery — `store_property_features_v2.py`
- `saf_st_composite_score_v2` → `saf_o2st_composite_score_v2`
- `saf_st_has_photos_v2` → `saf_o2st_has_photos_v2`
- `saf_st_inflates_price_v2` → `saf_o2st_inflates_price_v2`
- `saf_st_is_chain_v2` → `saf_o2st_is_chain_v2`
- `saf_st_is_dashpass_available_v2` → `saf_o2st_is_dashpass_available_v2`
- `saf_st_menu_item_photo_coverage` → `saf_o2st_menu_item_photo_coverage`
- `saf_st_offers_delivery_v2` → `saf_o2st_offers_delivery_v2`
- `saf_st_offers_pickup_v2` → `saf_o2st_offers_pickup_v2`
- `saf_st_p168d_discovery_ctr` → `saf_o2st_p168d_discovery_ctr`
- `saf_st_p28d_discovery_ctr` → `saf_o2st_p28d_discovery_ctr`
- `saf_st_p6m_rx_subtotal_normalized_at_submarket` → `saf_o2st_p6m_rx_subtotal_normalized_at_submarket`
- `saf_st_p84d_discovery_ctr` → `saf_o2st_p84d_discovery_ctr`
- `saf_st_price_range_v2` → `saf_o2st_price_range_v2`
- `saf_st_tier_level_v2` → `saf_o2st_tier_level_v2`

### `daf_cs_st_*` — cx_discovery — `store_engagement_features.py` (consumer-store)
- `daf_cs_st_p24w_consumer_order_count_v2` → `daf_cs_o2st_p24w_consumer_order_count_v2`
- `daf_cs_st_p24w_consumer_order_rank_v2` → `daf_cs_o2st_p24w_consumer_order_rank_v2`
- `daf_cs_st_p24w_order_count` → `daf_cs_o2st_p24w_order_count`
- `daf_cs_st_p24w_order_rank` → `daf_cs_o2st_p24w_order_rank`
- `daf_cs_st_p28d_consumer_store_orders` → `daf_cs_o2st_p28d_consumer_store_orders`
- `daf_cs_st_p28d_doubledash_num_orders` → `daf_cs_o2st_p28d_doubledash_num_orders`
- `daf_cs_st_p28d_doubledash_num_views` → `daf_cs_o2st_p28d_doubledash_num_views`
- `daf_cs_st_p6m_consumer_store_orders` → `daf_cs_o2st_p6m_consumer_store_orders`

### `daf_st_*` — cx_discovery — `store_engagement_features.py`
- `daf_st_offers_store_promo_quality_score_v2_cai` → `daf_o2st_offers_store_promo_quality_score_v2_cai`
- `daf_st_p168d_discovery_cvr` → `daf_o2st_p168d_discovery_cvr`
- `daf_st_p28d_discovery_cvr` → `daf_o2st_p28d_discovery_cvr`
- `daf_st_p4w_no_recent_orders_v2` → `daf_o2st_p4w_no_recent_orders_v2`
- `daf_st_p4w_store_delivery_market_share_v2` → `daf_o2st_p4w_store_delivery_market_share_v2`
- `daf_st_p84d_discovery_cvr` → `daf_o2st_p84d_discovery_cvr`
- `daf_st_p84d_doubledash_business_position_weighted_imp2conv_over_avg` → `daf_o2st_p84d_doubledash_business_position_weighted_imp2conv_over_avg`
- `daf_st_p84d_subtotal_avg` → `daf_o2st_p84d_subtotal_avg`

### `caf_cs_st_*` — cx_discovery — `store_engagement_features.py` (consumer-store)
- `caf_cs_st_is_saved` → `caf_cs_o2st_is_saved`
- `caf_cs_st_p28d_consumer_store_clicks` → `caf_cs_o2st_p28d_consumer_store_clicks`
- `caf_cs_st_p28d_consumer_store_views` → `caf_cs_o2st_p28d_consumer_store_views`
- `caf_cs_st_p6m_consumer_store_clicks` → `caf_cs_o2st_p6m_consumer_store_clicks`
- `caf_cs_st_p6m_consumer_store_views` → `caf_cs_o2st_p6m_consumer_store_views`
- `caf_cs_st_p84d_organic_cvr` → `caf_cs_o2st_p84d_organic_cvr`

### `caf_st_*` — cx_discovery — `store_property_features_v2.py`
- `caf_st_p12w_five_star_rate_lower_bound` → `caf_o2st_p12w_five_star_rate_lower_bound`
- `caf_st_p12w_five_star_rate_upper_bound` → `caf_o2st_p12w_five_star_rate_upper_bound`

### `daf_cs_bizv_*` — cx_discovery — `discovery_noramlized_features/consumer_business_vertical_*`
- `daf_cs_bizv_cng_vertical_num_orders` → `daf_cs_o2bizv_cng_vertical_num_orders`
- `daf_cs_bizv_p84d_consumer_business_vertical_order_recency` → `daf_cs_o2bizv_p84d_consumer_business_vertical_order_recency`

### `saf_dprg_st_*` — sponsored_listing — `rx_ads_quality/.../organic_store_daypart_aggregated_features.py`
- `saf_dprg_st_p28d_organic_cvr` → `saf_dprg_o2st_p28d_organic_cvr`
- `saf_dprg_st_p84d_organic_ctr` → `saf_dprg_o2st_p84d_organic_ctr`

## Also cross-team (not in the 91 rename targets)

- **User-history sequence** `caf_cs_p84d_organic_converted_business_id_…_sequence` (1) — cx_discovery `sequence_features/organic_consumer_sequence_*`. Bug 3: add `materialize_spec` (no o2 clone; consumer-keyed).
- **cos_sim store-side embedding operands** (`saf_st_*_emb`) need `o2_store` (and `o1_store` for o1_o2 pairs) bound clones + materialize before the Sibyl `Func.cos_sim` derived feature can serve correctly — see `dbd_fixes.md` §5.1.