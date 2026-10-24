package engine

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/cafe3310/decision-cli/internal/cache"
	"github.com/cafe3310/decision-cli/internal/infra"
	"github.com/cafe3310/decision-cli/internal/types"
)

// Options controls a single or batch decision evaluation run.
type Options struct {
	ModelName      string
	Rubrics        string
	Threshold      float64
	Temperature    float64
	NoCache        bool
	CacheOnly      bool
	FilterPassed   bool
	IncludeRecord  bool
	StreamProgress bool
}

func emitProgressJSON(payload any) {
	if b, err := json.Marshal(payload); err == nil {
		fmt.Fprintln(os.Stderr, string(b))
	}
}

// EvaluateSingle evaluates a single (rubrics, context) pair with SQLite caching and MLX fallback.
func EvaluateSingle(cfg *types.AppConfig, store *cache.Store, opts Options, contextText string) (*types.SingleDecisionResult, error) {
	t0 := time.Now()
	rm, modelErr := infra.ResolveModel(cfg, opts.ModelName)
	canonicalModelID := rm.Spec.HFRepoID
	if canonicalModelID == "" {
		canonicalModelID = opts.ModelName
	}

	cKey := cache.ComputeCacheKey(canonicalModelID, opts.Rubrics, contextText)

	if !opts.NoCache && store != nil {
		hits, err := store.GetBatch([]string{cKey})
		if err == nil {
			if entry, ok := hits[cKey]; ok {
				prob, conf, passed, decision := cache.CalibrateFromLogits(entry.LogitYes, entry.LogitNo, opts.Temperature, opts.Threshold)
				elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
				return &types.SingleDecisionResult{
					Decision:    decision,
					Passed:      passed,
					Probability: cache.Round4(prob),
					Confidence:  cache.Round4(conf),
					Threshold:   opts.Threshold,
					Temperature: opts.Temperature,
					Logits: types.LogitDetails{
						Yes:    cache.Round4(entry.LogitYes),
						No:     cache.Round4(entry.LogitNo),
						Margin: cache.Round4(entry.LogitYes - entry.LogitNo),
					},
					Model:     rm.Alias,
					ModelID:   canonicalModelID,
					Cached:    true,
					LatencyMs: cache.Round4(elapsed),
					CacheKey:  cKey,
				}, nil
			}
		}
	}

	if opts.CacheOnly {
		return nil, fmt.Errorf("cache miss for key %s under --cache-only mode", cKey)
	}
	if modelErr != nil {
		return nil, modelErr
	}

	rt := infra.ResolveRuntime(cfg)
	if !rt.Valid {
		return nil, fmt.Errorf("MLX runtime unavailable: %s", rt.Error)
	}

	resp, err := infra.ScoreBatchViaMLX(cfg, rt, rm, opts.Rubrics, []infra.WorkerScoreItem{
		{ID: "single", Context: contextText},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Results) == 0 {
		return nil, fmt.Errorf("empty result from MLX worker")
	}

	r := resp.Results[0]
	prob, conf, passed, decision := cache.CalibrateFromLogits(r.LogitYes, r.LogitNo, opts.Temperature, opts.Threshold)

	if !opts.NoCache && store != nil {
		defaultProb, _, _, _ := cache.CalibrateFromLogits(r.LogitYes, r.LogitNo, 1.0, 0.5)
		now := time.Now().Format(time.RFC3339)
		_ = store.PutBatch([]types.CacheEntry{
			{
				CacheKey:     cKey,
				ModelID:      canonicalModelID,
				RubricHash:   cache.HashString(cache.NormalizeText(opts.Rubrics)),
				Rubrics:      cache.NormalizeText(opts.Rubrics),
				RecordID:     "single",
				ContextHash:  cache.HashString(cache.NormalizeText(contextText)),
				Context:      cache.NormalizeText(contextText),
				LogitYes:     r.LogitYes,
				LogitNo:      r.LogitNo,
				LogitMargin:  r.Margin,
				Probability:  defaultProb,
				LatencyMs:    r.LatencyMs,
				CreatedAt:    now,
				HitCount:     0,
				LastAccessed: now,
			},
		})
	}

	elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
	return &types.SingleDecisionResult{
		Decision:    decision,
		Passed:      passed,
		Probability: cache.Round4(prob),
		Confidence:  cache.Round4(conf),
		Threshold:   opts.Threshold,
		Temperature: opts.Temperature,
		Logits: types.LogitDetails{
			Yes:    cache.Round4(r.LogitYes),
			No:     cache.Round4(r.LogitNo),
			Margin: cache.Round4(r.Margin),
		},
		Model:     rm.Alias,
		ModelID:   canonicalModelID,
		Cached:    false,
		LatencyMs: cache.Round4(elapsed),
		CacheKey:  cKey,
	}, nil
}

// EvaluateBatch evaluates a slice of BatchItemInput records, consulting SQLite cache first
// and dispatching only cache misses to the MLX worker in chunks.
func EvaluateBatch(cfg *types.AppConfig, store *cache.Store, opts Options, items []types.BatchItemInput) (*types.BatchDecisionReport, error) {
	t0 := time.Now()
	rm, modelErr := infra.ResolveModel(cfg, opts.ModelName)
	canonicalModelID := rm.Spec.HFRepoID
	if canonicalModelID == "" {
		canonicalModelID = opts.ModelName
	}

	keys := make([]string, len(items))
	for i, it := range items {
		keys[i] = cache.ComputeCacheKey(canonicalModelID, opts.Rubrics, it.Context)
	}

	cachedMap := make(map[string]types.CacheEntry)
	if !opts.NoCache && store != nil {
		var err error
		cachedMap, err = store.GetBatch(keys)
		if err != nil {
			return nil, fmt.Errorf("cache lookup failed: %w", err)
		}
	}

	// Identify misses (deduplicated by cache_key in case multiple items have identical context)
	var missWorkerItems []infra.WorkerScoreItem
	missKeySet := make(map[string]types.BatchItemInput)
	keyToOrigItems := make(map[string][]types.BatchItemInput)
	cachedDoneCount := 0
	var initialCachedItems []types.BatchItemResult

	for i, it := range items {
		ck := keys[i]
		keyToOrigItems[ck] = append(keyToOrigItems[ck], it)
		if cEntry, found := cachedMap[ck]; found {
			cachedDoneCount++
			if opts.StreamProgress {
				prob, conf, passed, decision := cache.CalibrateFromLogits(cEntry.LogitYes, cEntry.LogitNo, opts.Temperature, opts.Threshold)
				initialCachedItems = append(initialCachedItems, types.BatchItemResult{
					ID:          it.ID,
					Decision:    decision,
					Passed:      passed,
					Probability: cache.Round4(prob),
					Confidence:  cache.Round4(conf),
					Logits: types.LogitDetails{
						Yes:    cache.Round4(cEntry.LogitYes),
						No:     cache.Round4(cEntry.LogitNo),
						Margin: cache.Round4(cEntry.LogitYes - cEntry.LogitNo),
					},
					Cached:    true,
					LatencyMs: 0.05,
					CacheKey:  ck,
				})
			}
		} else {
			if _, queued := missKeySet[ck]; !queued {
				missKeySet[ck] = it
				missWorkerItems = append(missWorkerItems, infra.WorkerScoreItem{
					ID:      ck, // key by cache_key so duplicates share result
					Context: it.Context,
				})
			}
		}
	}

	if opts.StreamProgress {
		emitProgressJSON(map[string]any{
			"type":         "cache_ready",
			"done":         cachedDoneCount,
			"total":        len(items),
			"cache_hits":   cachedDoneCount,
			"cache_misses": len(items) - cachedDoneCount,
			"results":      initialCachedItems,
		})
	}

	// Evaluate misses via MLX worker in a single model load, streaming & saving to SQLite per item
	freshLogits := make(map[string]infra.WorkerScoreResult, len(missWorkerItems))
	if len(missWorkerItems) > 0 && !opts.CacheOnly {
		if modelErr != nil {
			return nil, modelErr
		}
		rt := infra.ResolveRuntime(cfg)
		if !rt.Valid {
			return nil, fmt.Errorf("MLX runtime unavailable: %s", rt.Error)
		}

		normRubrics := cache.NormalizeText(opts.Rubrics)
		rubricHash := cache.HashString(normRubrics)
		freshDoneItems := 0

		onStatus := func(st string) {
			if opts.StreamProgress {
				emitProgressJSON(map[string]any{
					"type":       "status",
					"status":     st,
					"done":       cachedDoneCount + freshDoneItems,
					"total":      len(items),
					"cache_hits": cachedDoneCount,
					"fresh_done": freshDoneItems,
				})
			}
		}

		onItem := func(wr infra.WorkerScoreResult) {
			freshLogits[wr.ID] = wr
			origItem := missKeySet[wr.ID]
			defProb, _, _, _ := cache.CalibrateFromLogits(wr.LogitYes, wr.LogitNo, 1.0, 0.5)
			normCtx := cache.NormalizeText(origItem.Context)
			now := time.Now().Format(time.RFC3339)

			if !opts.NoCache && store != nil {
				_ = store.PutBatch([]types.CacheEntry{
					{
						CacheKey:     wr.ID,
						ModelID:      canonicalModelID,
						RubricHash:   rubricHash,
						Rubrics:      normRubrics,
						RecordID:     origItem.ID,
						ContextHash:  cache.HashString(normCtx),
						Context:      normCtx,
						LogitYes:     wr.LogitYes,
						LogitNo:      wr.LogitNo,
						LogitMargin:  wr.Margin,
						Probability:  defProb,
						LatencyMs:    wr.LatencyMs,
						CreatedAt:    now,
						HitCount:     0,
						LastAccessed: now,
					},
				})
			}

			origList := keyToOrigItems[wr.ID]
			freshDoneItems += len(origList)

			if opts.StreamProgress {
				var stepResults []types.BatchItemResult
				for _, oit := range origList {
					prob, conf, passed, decision := cache.CalibrateFromLogits(wr.LogitYes, wr.LogitNo, opts.Temperature, opts.Threshold)
					stepResults = append(stepResults, types.BatchItemResult{
						ID:          oit.ID,
						Decision:    decision,
						Passed:      passed,
						Probability: cache.Round4(prob),
						Confidence:  cache.Round4(conf),
						Logits: types.LogitDetails{
							Yes:    cache.Round4(wr.LogitYes),
							No:     cache.Round4(wr.LogitNo),
							Margin: cache.Round4(wr.Margin),
						},
						Cached:    false,
						LatencyMs: cache.Round4(wr.LatencyMs),
						CacheKey:  wr.ID,
					})
				}
				emitProgressJSON(map[string]any{
					"type":       "progress",
					"done":       cachedDoneCount + freshDoneItems,
					"total":      len(items),
					"cache_hits": cachedDoneCount,
					"fresh_done": freshDoneItems,
					"results":    stepResults,
				})
			}
		}

		_, err := infra.ScoreBatchStreamingViaMLX(cfg, rt, rm, opts.Rubrics, missWorkerItems, onStatus, onItem)
		if err != nil {
			return nil, err
		}
	}

	results := make([]types.BatchItemResult, 0, len(items))
	var summary types.BatchSummary
	summary.Total = len(items)

	for i, it := range items {
		ck := keys[i]
		var logitYes, logitNo, margin, latencyMs float64
		var isCached bool

		if cEntry, ok := cachedMap[ck]; ok {
			logitYes = cEntry.LogitYes
			logitNo = cEntry.LogitNo
			margin = cEntry.LogitYes - cEntry.LogitNo
			latencyMs = 0.05
			isCached = true
			summary.CacheHits++
		} else if fEntry, ok := freshLogits[ck]; ok {
			logitYes = fEntry.LogitYes
			logitNo = fEntry.LogitNo
			margin = fEntry.Margin
			latencyMs = fEntry.LatencyMs
			isCached = false
			summary.CacheMisses++
		} else {
			// CacheOnly miss
			summary.CacheMisses++
			continue
		}

		prob, conf, passed, decision := cache.CalibrateFromLogits(logitYes, logitNo, opts.Temperature, opts.Threshold)
		if passed {
			summary.PassedCount++
		} else {
			summary.RejectedCount++
		}

		if opts.FilterPassed && !passed {
			continue
		}

		resItem := types.BatchItemResult{
			ID:          it.ID,
			Decision:    decision,
			Passed:      passed,
			Probability: cache.Round4(prob),
			Confidence:  cache.Round4(conf),
			Logits: types.LogitDetails{
				Yes:    cache.Round4(logitYes),
				No:     cache.Round4(logitNo),
				Margin: cache.Round4(margin),
			},
			Cached:    isCached,
			LatencyMs: cache.Round4(latencyMs),
			CacheKey:  ck,
			Context:   it.Context,
		}
		if opts.IncludeRecord {
			resItem.Record = it.Raw
		}
		results = append(results, resItem)
	}

	summary.TotalTimeMs = cache.Round4(float64(time.Since(t0).Microseconds()) / 1000.0)

	return &types.BatchDecisionReport{
		ExportedAt:  time.Now().Format(time.RFC3339),
		Model:       rm.Alias,
		ModelID:     canonicalModelID,
		Rubrics:     opts.Rubrics,
		Threshold:   opts.Threshold,
		Temperature: opts.Temperature,
		Summary:     summary,
		Results:     results,
	}, nil
}
