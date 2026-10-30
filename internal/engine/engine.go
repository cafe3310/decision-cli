package engine

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/cafe3310/decision-cli/internal/cache"
	"github.com/cafe3310/decision-cli/internal/config"
	"github.com/cafe3310/decision-cli/internal/infra"
	"github.com/cafe3310/decision-cli/internal/types"
)

// Options controls a single or batch decision evaluation run.
type Options struct {
	ModelName      string
	Primitive      string // "noul" (default) | "choice" | "score"
	Rubrics        string
	Choices        []types.ChoiceOption
	ScoreLevels    []string
	Images         []string
	ImageTokens    int
	DebiasOrder    bool
	Threshold      float64
	Temperature    float64
	CalibA         *float64
	CalibB         *float64
	NoCache        bool
	CacheOnly      bool
	FilterPassed   bool
	IncludeRecord  bool
	StreamProgress bool
	SlicedHead     string // "on" | "off" | "" (default "on")
}

func validateVisionModelSupport(rm infra.ResolvedModel, images []string) error {
	if len(images) == 0 {
		return nil
	}
	if !rm.Spec.SupportsVision && rm.Spec.PromptStyle == "qwen3-reranker" {
		return fmt.Errorf("模型 %s (%s) 为纯文本模型，不支持 --image 多模态视觉输入；请改用 --model gemma-4-26b-a4b（默认推荐）或 --model gemma-4-e4b", rm.Alias, rm.Spec.HFRepoID)
	}
	return nil
}

// ParseChoicesFlag parses a --choices string where options are separated by ',' and key/description are separated by ':'.
// Example: "bug:系统报错或崩溃,feature:新功能建议,sales:商务询价"
// Also supports JSON array/map if raw starts with '[' or '{'.
func ParseChoicesFlag(raw string) ([]types.ChoiceOption, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return nil, nil
	}
	if strings.HasPrefix(s, "[") {
		var opts []types.ChoiceOption
		if err := json.Unmarshal([]byte(s), &opts); err == nil && len(opts) >= 2 {
			return opts, nil
		}
		var strList []string
		if err := json.Unmarshal([]byte(s), &strList); err == nil && len(strList) >= 2 {
			out := make([]types.ChoiceOption, 0, len(strList))
			for _, item := range strList {
				item = strings.TrimSpace(item)
				if item != "" {
					out = append(out, types.ChoiceOption{Key: item, Description: item})
				}
			}
			if len(out) >= 2 {
				return out, nil
			}
		}
	}
	if strings.HasPrefix(s, "{") {
		var m map[string]string
		if err := json.Unmarshal([]byte(s), &m); err == nil && len(m) >= 2 {
			keys := make([]string, 0, len(m))
			for k := range m {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			out := make([]types.ChoiceOption, 0, len(keys))
			for _, k := range keys {
				out = append(out, types.ChoiceOption{Key: k, Description: m[k]})
			}
			return out, nil
		}
	}

	parts := strings.Split(s, ",")
	var out []types.ChoiceOption
	seen := make(map[string]bool)
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var key, desc string
		if colonIdx := strings.Index(p, ":"); colonIdx >= 0 {
			key = strings.TrimSpace(p[:colonIdx])
			desc = strings.TrimSpace(p[colonIdx+1:])
		} else {
			key = p
			desc = p
		}
		if key == "" {
			continue
		}
		if seen[key] {
			return nil, fmt.Errorf("duplicate choice key %q in --choices (note: ',' separates options and ':' separates key:description)", key)
		}
		seen[key] = true
		out = append(out, types.ChoiceOption{Key: key, Description: desc})
	}
	if len(out) < 2 {
		return nil, fmt.Errorf("--choices requires at least 2 options separated by ',' in 'key:description' format (got %d)", len(out))
	}
	return out, nil
}

// ParseScoreFlag parses a --score string where ordinal rubric levels (from 0 to M-1) are separated by ','.
// Example: "平静常规反馈,轻微不满或催促,严重受阻或情绪愤怒"
func ParseScoreFlag(raw string) ([]string, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return nil, nil
	}
	if strings.HasPrefix(s, "[") {
		var levels []string
		if err := json.Unmarshal([]byte(s), &levels); err == nil && len(levels) >= 2 {
			return levels, nil
		}
	}
	parts := strings.Split(s, ",")
	var out []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	if len(out) < 2 || len(out) > 10 {
		return nil, fmt.Errorf("--score requires 2 to 10 ordinal levels separated by ',' (got %d)", len(out))
	}
	return out, nil
}

func emitProgressJSON(payload any) {
	if b, err := json.Marshal(payload); err == nil {
		fmt.Fprintln(os.Stderr, string(b))
	}
}

func roundLogitsMap(m map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(m))
	for k, v := range m {
		out[k] = cache.Round4(v)
	}
	return out
}

// EvaluateSingleChoice evaluates a single multi-class choice decision with SQLite caching and MLX fallback.
func EvaluateSingleChoice(cfg *types.AppConfig, store *cache.Store, opts Options, contextText string) (*types.SingleChoiceResult, error) {
	t0 := time.Now()
	if _, err := infra.ParseSlicedHeadMode(opts.SlicedHead); err != nil {
		return nil, err
	}
	rm, modelErr := infra.ResolveModel(cfg, opts.ModelName)
	if err := validateVisionModelSupport(rm, opts.Images); err != nil {
		return nil, err
	}
	resolvedImages, imageSig, normImgTokens, imgErr := cache.ResolveAndHashImages(opts.Images, opts.ImageTokens)
	if imgErr != nil {
		return nil, imgErr
	}
	canonicalModelID := rm.Spec.HFRepoID
	if canonicalModelID == "" {
		canonicalModelID = opts.ModelName
	}

	critBytes, _ := json.Marshal(opts.Choices)
	critJSON := string(critBytes)
	cKey := cache.ComputeMultiCacheKeyWithImages(canonicalModelID, "choice", opts.Rubrics, critJSON, contextText, opts.DebiasOrder, imageSig)

	if !opts.NoCache && store != nil {
		if hits, err := store.GetBatch([]string{cKey}); err == nil {
			if entry, ok := hits[cKey]; ok && entry.RawLogitsJSON != "" {
				var rawLogits map[string]float64
				if json.Unmarshal([]byte(entry.RawLogitsJSON), &rawLogits) == nil {
					winner, conf, probs := cache.CalibrateChoiceFromLogits(rawLogits, opts.Choices, opts.Temperature)
					elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
					return &types.SingleChoiceResult{
						Primitive:     "choice",
						Choice:        winner,
						Confidence:    conf,
						Probabilities: probs,
						Logits:        roundLogitsMap(rawLogits),
						DebiasedOrder: opts.DebiasOrder,
						Temperature:   opts.Temperature,
						Model:         rm.Alias,
						ModelID:       canonicalModelID,
						Images:        opts.Images,
						ImageTokens:   normImgTokens,
						Cached:        true,
						LatencyMs:     cache.Round4(elapsed),
						CacheKey:      cKey,
					}, nil
				}
			}
		}
	}

	if opts.CacheOnly {
		return nil, fmt.Errorf("cache miss for key %s under --cache-only mode", cKey)
	}
	if modelErr != nil {
		return nil, modelErr
	}

	resp, err := infra.ScoreMultiViaMLX(cfg, nil, rm, infra.WorkerMultiParams{
		Primitive:   "choice",
		Rubrics:     opts.Rubrics,
		Choices:     opts.Choices,
		DebiasOrder: opts.DebiasOrder,
		ImageTokens: normImgTokens,
		SlicedHead:  opts.SlicedHead,
	}, []infra.WorkerScoreItem{
		{ID: "single", Context: contextText, Images: resolvedImages, ImageTokens: normImgTokens},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Results) == 0 {
		return nil, fmt.Errorf("empty result from MLX worker")
	}

	r := resp.Results[0]
	winner, conf, probs := cache.CalibrateChoiceFromLogits(r.OptionLogits, opts.Choices, opts.Temperature)

	if !opts.NoCache && store != nil {
		rawBytes, _ := json.Marshal(r.OptionLogits)
		now := time.Now().Format(time.RFC3339)
		_ = store.PutBatch([]types.CacheEntry{
			{
				CacheKey:      cKey,
				ModelID:       canonicalModelID,
				Primitive:     "choice",
				CriteriaJSON:  critJSON,
				RawLogitsJSON: string(rawBytes),
				RubricHash:    cache.HashString(cache.NormalizeText(opts.Rubrics)),
				Rubrics:       cache.NormalizeText(opts.Rubrics),
				RecordID:      "single",
				ContextHash:   cache.HashString(cache.NormalizeText(contextText) + "|" + imageSig),
				Context:       cache.NormalizeText(contextText),
				Probability:   conf,
				LatencyMs:     r.LatencyMs,
				CreatedAt:     now,
				HitCount:      0,
				LastAccessed:  now,
			},
		})
	}

	elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
	return &types.SingleChoiceResult{
		Primitive:     "choice",
		Choice:        winner,
		Confidence:    conf,
		Probabilities: probs,
		Logits:        roundLogitsMap(r.OptionLogits),
		DebiasedOrder: opts.DebiasOrder,
		Temperature:   opts.Temperature,
		Model:         rm.Alias,
		ModelID:       canonicalModelID,
		Images:        opts.Images,
		ImageTokens:   normImgTokens,
		Cached:        false,
		LatencyMs:     cache.Round4(elapsed),
		CacheKey:      cKey,
	}, nil
}

// EvaluateSingleScore evaluates a single ordinal score decision with SQLite caching and MLX fallback.
func EvaluateSingleScore(cfg *types.AppConfig, store *cache.Store, opts Options, contextText string) (*types.SingleScoreResult, error) {
	t0 := time.Now()
	if _, err := infra.ParseSlicedHeadMode(opts.SlicedHead); err != nil {
		return nil, err
	}
	rm, modelErr := infra.ResolveModel(cfg, opts.ModelName)
	if err := validateVisionModelSupport(rm, opts.Images); err != nil {
		return nil, err
	}
	resolvedImages, imageSig, normImgTokens, imgErr := cache.ResolveAndHashImages(opts.Images, opts.ImageTokens)
	if imgErr != nil {
		return nil, imgErr
	}
	canonicalModelID := rm.Spec.HFRepoID
	if canonicalModelID == "" {
		canonicalModelID = opts.ModelName
	}

	critBytes, _ := json.Marshal(opts.ScoreLevels)
	critJSON := string(critBytes)
	cKey := cache.ComputeMultiCacheKeyWithImages(canonicalModelID, "score", opts.Rubrics, critJSON, contextText, opts.DebiasOrder, imageSig)

	if !opts.NoCache && store != nil {
		if hits, err := store.GetBatch([]string{cKey}); err == nil {
			if entry, ok := hits[cKey]; ok && entry.RawLogitsJSON != "" {
				var rawLogits map[string]float64
				if json.Unmarshal([]byte(entry.RawLogitsJSON), &rawLogits) == nil {
					expScore, normScore, topLv, topLbl, conf, probs, legend := cache.CalibrateScoreFromLogits(rawLogits, opts.ScoreLevels, opts.Temperature)
					elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
					return &types.SingleScoreResult{
						Primitive:       "score",
						Score:           expScore,
						NormalizedScore: normScore,
						TopLevel:        topLv,
						TopLabel:        topLbl,
						Confidence:      conf,
						Probabilities:   probs,
						Legend:          legend,
						Logits:          roundLogitsMap(rawLogits),
						DebiasedOrder:   opts.DebiasOrder,
						Temperature:     opts.Temperature,
						Model:           rm.Alias,
						ModelID:         canonicalModelID,
						Images:          opts.Images,
						ImageTokens:     normImgTokens,
						Cached:          true,
						LatencyMs:       cache.Round4(elapsed),
						CacheKey:        cKey,
					}, nil
				}
			}
		}
	}

	if opts.CacheOnly {
		return nil, fmt.Errorf("cache miss for key %s under --cache-only mode", cKey)
	}
	if modelErr != nil {
		return nil, modelErr
	}

	resp, err := infra.ScoreMultiViaMLX(cfg, nil, rm, infra.WorkerMultiParams{
		Primitive:   "score",
		Rubrics:     opts.Rubrics,
		ScoreLevels: opts.ScoreLevels,
		DebiasOrder: opts.DebiasOrder,
		ImageTokens: normImgTokens,
		SlicedHead:  opts.SlicedHead,
	}, []infra.WorkerScoreItem{
		{ID: "single", Context: contextText, Images: resolvedImages, ImageTokens: normImgTokens},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Results) == 0 {
		return nil, fmt.Errorf("empty result from MLX worker")
	}

	r := resp.Results[0]
	expScore, normScore, topLv, topLbl, conf, probs, legend := cache.CalibrateScoreFromLogits(r.OptionLogits, opts.ScoreLevels, opts.Temperature)

	if !opts.NoCache && store != nil {
		rawBytes, _ := json.Marshal(r.OptionLogits)
		now := time.Now().Format(time.RFC3339)
		_ = store.PutBatch([]types.CacheEntry{
			{
				CacheKey:      cKey,
				ModelID:       canonicalModelID,
				Primitive:     "score",
				CriteriaJSON:  critJSON,
				RawLogitsJSON: string(rawBytes),
				RubricHash:    cache.HashString(cache.NormalizeText(opts.Rubrics)),
				Rubrics:       cache.NormalizeText(opts.Rubrics),
				RecordID:      "single",
				ContextHash:   cache.HashString(cache.NormalizeText(contextText) + "|" + imageSig),
				Context:       cache.NormalizeText(contextText),
				Probability:   expScore,
				LatencyMs:     r.LatencyMs,
				CreatedAt:     now,
				HitCount:      0,
				LastAccessed:  now,
			},
		})
	}

	elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
	return &types.SingleScoreResult{
		Primitive:       "score",
		Score:           expScore,
		NormalizedScore: normScore,
		TopLevel:        topLv,
		TopLabel:        topLbl,
		Confidence:      conf,
		Probabilities:   probs,
		Legend:          legend,
		Logits:          roundLogitsMap(r.OptionLogits),
		DebiasedOrder:   opts.DebiasOrder,
		Temperature:     opts.Temperature,
		Model:           rm.Alias,
		ModelID:         canonicalModelID,
		Images:          opts.Images,
		ImageTokens:     normImgTokens,
		Cached:          false,
		LatencyMs:       cache.Round4(elapsed),
		CacheKey:        cKey,
	}, nil
}

// CalibrateDecision resolves either temperature-based or affine-based Platt calibration.
func CalibrateDecision(logitYes, logitNo float64, opts Options) (prob, conf float64, passed bool, decision string) {
	if opts.CalibA != nil || opts.CalibB != nil {
		a := 1.0
		if opts.CalibA != nil {
			a = *opts.CalibA
		} else if opts.Temperature > 0 {
			a = 1.0 / opts.Temperature
		}
		b := 0.0
		if opts.CalibB != nil {
			b = *opts.CalibB
		}
		return cache.CalibrateFromLogitsWithAffine(logitYes, logitNo, a, b, opts.Threshold)
	}
	return cache.CalibrateFromLogits(logitYes, logitNo, opts.Temperature, opts.Threshold)
}

// EvaluateSingle evaluates a single (rubrics, context) pair with SQLite caching and MLX fallback.
func EvaluateSingle(cfg *types.AppConfig, store *cache.Store, opts Options, contextText string) (*types.SingleDecisionResult, error) {
	t0 := time.Now()
	if _, err := infra.ParseSlicedHeadMode(opts.SlicedHead); err != nil {
		return nil, err
	}
	rm, modelErr := infra.ResolveModel(cfg, opts.ModelName)
	if err := validateVisionModelSupport(rm, opts.Images); err != nil {
		return nil, err
	}
	resolvedImages, imageSig, normImgTokens, imgErr := cache.ResolveAndHashImages(opts.Images, opts.ImageTokens)
	if imgErr != nil {
		return nil, imgErr
	}
	canonicalModelID := rm.Spec.HFRepoID
	if canonicalModelID == "" {
		canonicalModelID = opts.ModelName
	}

	cKey := cache.ComputeCacheKeyWithImages(canonicalModelID, opts.Rubrics, contextText, imageSig)

	if !opts.NoCache && store != nil {
		hits, err := store.GetBatch([]string{cKey})
		if err == nil {
			if entry, ok := hits[cKey]; ok {
				prob, conf, passed, decision := CalibrateDecision(entry.LogitYes, entry.LogitNo, opts)
				elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
				return &types.SingleDecisionResult{
					Decision:    decision,
					Passed:      passed,
					Probability: cache.Round4(prob),
					Confidence:  cache.Round4(conf),
					Threshold:   opts.Threshold,
					Temperature: opts.Temperature,
					CalibA:      opts.CalibA,
					CalibB:      opts.CalibB,
					Logits: types.LogitDetails{
						Yes:    cache.Round4(entry.LogitYes),
						No:     cache.Round4(entry.LogitNo),
						Margin: cache.Round4(entry.LogitYes - entry.LogitNo),
					},
					Model:       rm.Alias,
					ModelID:     canonicalModelID,
					Images:      opts.Images,
					ImageTokens: normImgTokens,
					Cached:      true,
					LatencyMs:   cache.Round4(elapsed),
					CacheKey:    cKey,
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

	resp, err := infra.ScoreMultiViaMLX(cfg, nil, rm, infra.WorkerMultiParams{
		Primitive:   "noul",
		Rubrics:     opts.Rubrics,
		ImageTokens: normImgTokens,
		SlicedHead:  opts.SlicedHead,
	}, []infra.WorkerScoreItem{
		{ID: "single", Context: contextText, Images: resolvedImages, ImageTokens: normImgTokens},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Results) == 0 {
		return nil, fmt.Errorf("empty result from MLX worker")
	}

	r := resp.Results[0]
	prob, conf, passed, decision := CalibrateDecision(r.LogitYes, r.LogitNo, opts)

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
				ContextHash:  cache.HashString(cache.NormalizeText(contextText) + "|" + imageSig),
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
		CalibA:      opts.CalibA,
		CalibB:      opts.CalibB,
		Logits: types.LogitDetails{
			Yes:    cache.Round4(r.LogitYes),
			No:     cache.Round4(r.LogitNo),
			Margin: cache.Round4(r.Margin),
		},
		Model:       rm.Alias,
		ModelID:     canonicalModelID,
		Images:      opts.Images,
		ImageTokens: normImgTokens,
		Cached:      false,
		LatencyMs:   cache.Round4(elapsed),
		CacheKey:    cKey,
	}, nil
}

// EvaluateBatch evaluates a slice of BatchItemInput records across "noul", "choice", or "score" primitives,
// consulting SQLite cache first and dispatching only cache misses to the MLX worker in a single model load.
func EvaluateBatch(cfg *types.AppConfig, store *cache.Store, opts Options, items []types.BatchItemInput) (*types.BatchDecisionReport, error) {
	t0 := time.Now()
	if _, err := infra.ParseSlicedHeadMode(opts.SlicedHead); err != nil {
		return nil, err
	}
	infra.EnsureStatusBarRunning()
	prim := opts.Primitive
	if prim == "" {
		prim = "noul"
	}
	rm, modelErr := infra.ResolveModel(cfg, opts.ModelName)
	canonicalModelID := rm.Spec.HFRepoID
	if canonicalModelID == "" {
		canonicalModelID = opts.ModelName
	}

	var critJSON string
	if prim == "choice" {
		b, _ := json.Marshal(opts.Choices)
		critJSON = string(b)
	} else if prim == "score" {
		b, _ := json.Marshal(opts.ScoreLevels)
		critJSON = string(b)
	}

	keys := make([]string, len(items))
	resolvedImgsPerItem := make([][]string, len(items))
	normImgTokens := cache.NormalizeImageTokens(opts.ImageTokens)
	for i, it := range items {
		itemImgs := it.Images
		if len(itemImgs) == 0 && len(opts.Images) > 0 {
			itemImgs = opts.Images
		}
		if len(itemImgs) > 0 {
			if err := validateVisionModelSupport(rm, itemImgs); err != nil {
				return nil, err
			}
		}
		resImgs, imgSig, _, imgErr := cache.ResolveAndHashImages(itemImgs, opts.ImageTokens)
		if imgErr != nil {
			return nil, fmt.Errorf("record %q: %w", it.ID, imgErr)
		}
		resolvedImgsPerItem[i] = resImgs
		if prim == "noul" {
			keys[i] = cache.ComputeCacheKeyWithImages(canonicalModelID, opts.Rubrics, it.Context, imgSig)
		} else {
			keys[i] = cache.ComputeMultiCacheKeyWithImages(canonicalModelID, prim, opts.Rubrics, critJSON, it.Context, opts.DebiasOrder, imgSig)
		}
	}

	cachedMap := make(map[string]types.CacheEntry)
	if !opts.NoCache && store != nil {
		var err error
		cachedMap, err = store.GetBatch(keys)
		if err != nil {
			return nil, fmt.Errorf("cache lookup failed: %w", err)
		}
	}

	buildBatchItem := func(it types.BatchItemInput, ck string, isCached bool, latencyMs float64, logitYes, logitNo float64, optLogits map[string]float64) types.BatchItemResult {
		itemImgs := it.Images
		if len(itemImgs) == 0 && len(opts.Images) > 0 {
			itemImgs = opts.Images
		}
		resItem := types.BatchItemResult{
			ID:        it.ID,
			Primitive: prim,
			Images:    itemImgs,
			Cached:    isCached,
			LatencyMs: cache.Round4(latencyMs),
			CacheKey:  ck,
			Context:   it.Context,
		}
		if opts.IncludeRecord {
			resItem.Record = it.Raw
		}
		switch prim {
		case "choice":
			winner, conf, probs := cache.CalibrateChoiceFromLogits(optLogits, opts.Choices, opts.Temperature)
			resItem.Decision = winner
			resItem.Choice = winner
			resItem.Confidence = conf
			resItem.Probabilities = probs
			resItem.OptionLogits = roundLogitsMap(optLogits)
		case "score":
			expScore, normScore, _, topLbl, conf, probs, _ := cache.CalibrateScoreFromLogits(optLogits, opts.ScoreLevels, opts.Temperature)
			resItem.Decision = topLbl
			resItem.Score = &expScore
			resItem.NormalizedScore = &normScore
			resItem.Confidence = conf
			resItem.Probabilities = probs
			resItem.OptionLogits = roundLogitsMap(optLogits)
		default:
			prob, conf, passed, decision := CalibrateDecision(logitYes, logitNo, opts)
			resItem.Decision = decision
			resItem.Passed = passed
			resItem.Probability = cache.Round4(prob)
			resItem.Confidence = cache.Round4(conf)
			resItem.Logits = &types.LogitDetails{
				Yes:    cache.Round4(logitYes),
				No:     cache.Round4(logitNo),
				Margin: cache.Round4(logitYes - logitNo),
			}
		}
		return resItem
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
		cEntry, found := cachedMap[ck]
		if found && (prim == "noul" || cEntry.RawLogitsJSON != "") {
			cachedDoneCount++
			if opts.StreamProgress {
				var optLogits map[string]float64
				if prim != "noul" {
					_ = json.Unmarshal([]byte(cEntry.RawLogitsJSON), &optLogits)
				}
				initialCachedItems = append(initialCachedItems, buildBatchItem(it, ck, true, 0.05, cEntry.LogitYes, cEntry.LogitNo, optLogits))
			}
		} else {
			if _, queued := missKeySet[ck]; !queued {
				missKeySet[ck] = it
				missWorkerItems = append(missWorkerItems, infra.WorkerScoreItem{
					ID:          ck, // key by cache_key so duplicates share result
					Context:     it.Context,
					Images:      resolvedImgsPerItem[i],
					ImageTokens: normImgTokens,
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
			normCtx := cache.NormalizeText(origItem.Context)
			now := time.Now().Format(time.RFC3339)

			if !opts.NoCache && store != nil {
				entry := types.CacheEntry{
					CacheKey:     wr.ID,
					ModelID:      canonicalModelID,
					Primitive:    prim,
					CriteriaJSON: critJSON,
					RubricHash:   rubricHash,
					Rubrics:      normRubrics,
					RecordID:     origItem.ID,
					ContextHash:  cache.HashString(normCtx),
					Context:      normCtx,
					LatencyMs:    wr.LatencyMs,
					CreatedAt:    now,
					HitCount:     0,
					LastAccessed: now,
				}
				if prim == "noul" {
					defProb, _, _, _ := cache.CalibrateFromLogits(wr.LogitYes, wr.LogitNo, 1.0, 0.5)
					entry.LogitYes = wr.LogitYes
					entry.LogitNo = wr.LogitNo
					entry.LogitMargin = wr.Margin
					entry.Probability = defProb
				} else {
					rawBytes, _ := json.Marshal(wr.OptionLogits)
					entry.RawLogitsJSON = string(rawBytes)
				}
				_ = store.PutBatch([]types.CacheEntry{entry})
			}

			origList := keyToOrigItems[wr.ID]
			freshDoneItems += len(origList)

			if opts.StreamProgress {
				var stepResults []types.BatchItemResult
				for _, oit := range origList {
					stepResults = append(stepResults, buildBatchItem(oit, wr.ID, false, wr.LatencyMs, wr.LogitYes, wr.LogitNo, wr.OptionLogits))
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

		_, err := infra.ScoreMultiStreamingViaMLX(cfg, nil, rm, infra.WorkerMultiParams{
			Primitive:   prim,
			Rubrics:     opts.Rubrics,
			Choices:     opts.Choices,
			ScoreLevels: opts.ScoreLevels,
			DebiasOrder: opts.DebiasOrder,
			ImageTokens: normImgTokens,
			SlicedHead:  opts.SlicedHead,
		}, missWorkerItems, onStatus, onItem)
		if err != nil {
			return nil, err
		}
	}

	results := make([]types.BatchItemResult, 0, len(items))
	var summary types.BatchSummary
	summary.Total = len(items)
	if prim == "choice" {
		summary.ChoiceCounts = make(map[string]int)
	}
	var scoreSum float64
	var scoreCount int

	for i, it := range items {
		ck := keys[i]
		var resItem types.BatchItemResult

		if cEntry, ok := cachedMap[ck]; ok && (prim == "noul" || cEntry.RawLogitsJSON != "") {
			var optLogits map[string]float64
			if prim != "noul" {
				_ = json.Unmarshal([]byte(cEntry.RawLogitsJSON), &optLogits)
			}
			resItem = buildBatchItem(it, ck, true, 0.05, cEntry.LogitYes, cEntry.LogitNo, optLogits)
			summary.CacheHits++
		} else if fEntry, ok := freshLogits[ck]; ok {
			resItem = buildBatchItem(it, ck, false, fEntry.LatencyMs, fEntry.LogitYes, fEntry.LogitNo, fEntry.OptionLogits)
			summary.CacheMisses++
		} else {
			summary.CacheMisses++
			continue
		}

		switch prim {
		case "choice":
			summary.ChoiceCounts[resItem.Choice]++
		case "score":
			if resItem.Score != nil {
				scoreSum += *resItem.Score
				scoreCount++
			}
		default:
			if resItem.Passed {
				summary.PassedCount++
			} else {
				summary.RejectedCount++
			}
			if opts.FilterPassed && !resItem.Passed {
				continue
			}
		}
		results = append(results, resItem)
	}

	if prim == "score" && scoreCount > 0 {
		mean := cache.Round4(scoreSum / float64(scoreCount))
		summary.MeanScore = &mean
	}

	summary.TotalTimeMs = cache.Round4(float64(time.Since(t0).Microseconds()) / 1000.0)

	return &types.BatchDecisionReport{
		ExportedAt:    time.Now().Format(time.RFC3339),
		Primitive:     prim,
		Model:         rm.Alias,
		ModelID:       canonicalModelID,
		Rubrics:       opts.Rubrics,
		Choices:       opts.Choices,
		ScoreLevels:   opts.ScoreLevels,
		DebiasedOrder: opts.DebiasOrder,
		Threshold:     opts.Threshold,
		Temperature:   opts.Temperature,
		CalibA:        opts.CalibA,
		CalibB:        opts.CalibB,
		Summary:       summary,
		Results:       results,
	}, nil
}

// EvaluateSystemOne evaluates a TypeSafe Jev /v1/systemone request containing multiple typed questions (noul, choice, score)
// against a shared state context, batching all cache-miss questions into a single MLX model pass.
func EvaluateSystemOne(cfg *types.AppConfig, store *cache.Store, baseOpts Options, req *types.SystemOneRequest, flagTemp float64, flagTempChanged bool) (*types.SystemOneResponse, error) {
	t0 := time.Now()
	if _, err := infra.ParseSlicedHeadMode(baseOpts.SlicedHead); err != nil {
		return nil, err
	}
	modelName := baseOpts.ModelName
	if modelName == "" && req.Model != "" && !strings.HasPrefix(strings.ToLower(req.Model), "jev") {
		modelName = req.Model
	}
	if modelName == "" {
		modelName = cfg.DefaultModel
	}
	rm, modelErr := infra.ResolveModel(cfg, modelName)
	canonicalModelID := rm.Spec.HFRepoID
	if canonicalModelID == "" {
		canonicalModelID = modelName
	}

	var stateStr string
	switch v := req.State.(type) {
	case nil:
		stateStr = ""
	case string:
		stateStr = v
	default:
		b, _ := json.Marshal(v)
		stateStr = string(b)
	}

	// Collect multimodal images from baseOpts.Images, req.Images, and req.Assets
	var combinedImages []string
	combinedImages = append(combinedImages, baseOpts.Images...)
	combinedImages = append(combinedImages, req.Images...)
	assetKeyToIndex := make(map[string]int)
	if len(req.Assets) > 0 {
		assetKeys := make([]string, 0, len(req.Assets))
		for k := range req.Assets {
			assetKeys = append(assetKeys, k)
		}
		sort.Strings(assetKeys)
		for _, k := range assetKeys {
			combinedImages = append(combinedImages, req.Assets[k])
			assetKeyToIndex[k] = len(combinedImages)
		}
	}
	effImgTokens := baseOpts.ImageTokens
	if req.ImageTokens > 0 {
		effImgTokens = req.ImageTokens
	}
	if len(combinedImages) > 0 {
		if err := validateVisionModelSupport(rm, combinedImages); err != nil {
			return nil, err
		}
	}
	resolvedSysImages, sysImageSig, normSysImgTokens, sysImgErr := cache.ResolveAndHashImages(combinedImages, effImgTokens)
	if sysImgErr != nil {
		return nil, sysImgErr
	}
	replaceAssetRefs := func(s string) string {
		for k, idx := range assetKeyToIndex {
			s = strings.ReplaceAll(s, fmt.Sprintf("{{asset:%s}}", k), fmt.Sprintf("{{image:%d}}", idx))
			s = strings.ReplaceAll(s, fmt.Sprintf("{{assets.%s}}", k), fmt.Sprintf("{{image:%d}}", idx))
		}
		return s
	}
	stateStr = replaceAssetRefs(stateStr)

	// Deterministic ordering of question IDs
	qIDs := make([]string, 0, len(req.Questions))
	for k := range req.Questions {
		qIDs = append(qIDs, k)
	}
	sort.Strings(qIDs)

	type preparedQuestion struct {
		qID         string
		qType       string
		rubrics     string
		choices     []types.ChoiceOption
		scoreLevels []string
		critJSON    string
		cacheKey    string
		temp        float64
	}

	prepared := make([]preparedQuestion, 0, len(qIDs))
	cacheKeys := make([]string, 0, len(qIDs))

	for _, qID := range qIDs {
		q := req.Questions[qID]
		qType := strings.ToLower(strings.TrimSpace(q.Type))
		if qType == "" {
			qType = "noul"
		}
		instr := replaceAssetRefs(q.Instructions)
		pq := preparedQuestion{
			qID:     qID,
			qType:   qType,
			rubrics: instr,
			temp:    config.EffectivePrimitiveTemperature(cfg, rm.Spec, qType, flagTemp, flagTempChanged),
		}
		switch qType {
		case "noul":
			if cmap, ok := q.Criteria.(map[string]any); ok && len(cmap) > 0 {
				var cParts []string
				for _, k := range []string{"true", "yes", "false", "no"} {
					if v, exists := cmap[k]; exists && v != nil {
						s := strings.TrimSpace(fmt.Sprintf("%v", v))
						if s != "" {
							label := "YES (true)"
							if k == "false" || k == "no" {
								label = "NO (false)"
							}
							cParts = append(cParts, fmt.Sprintf("- %s: %s", label, s))
						}
					}
				}
				if len(cParts) > 0 {
					pq.rubrics = strings.TrimSpace(instr) + "\n" + strings.Join(cParts, "\n")
				}
			}
			pq.cacheKey = cache.ComputeCacheKeyWithImages(canonicalModelID, pq.rubrics, stateStr, sysImageSig)
		case "choice":
			choices, err := parseCriteriaToChoices(q.Criteria)
			if err != nil {
				return nil, fmt.Errorf("question %q (choice) invalid criteria: %w", qID, err)
			}
			pq.choices = choices
			cb, _ := json.Marshal(choices)
			pq.critJSON = string(cb)
			pq.cacheKey = cache.ComputeMultiCacheKeyWithImages(canonicalModelID, "choice", instr, pq.critJSON, stateStr, baseOpts.DebiasOrder, sysImageSig)
		case "score":
			levels, err := parseCriteriaToScoreLevels(q.Criteria)
			if err != nil {
				return nil, fmt.Errorf("question %q (score) invalid criteria: %w", qID, err)
			}
			pq.scoreLevels = levels
			cb, _ := json.Marshal(levels)
			pq.critJSON = string(cb)
			pq.cacheKey = cache.ComputeMultiCacheKeyWithImages(canonicalModelID, "score", instr, pq.critJSON, stateStr, baseOpts.DebiasOrder, sysImageSig)
		default:
			return nil, fmt.Errorf("unsupported question type %q in question %q (must be noul, choice, or score)", q.Type, qID)
		}
		prepared = append(prepared, pq)
		cacheKeys = append(cacheKeys, pq.cacheKey)
	}

	cachedMap := make(map[string]types.CacheEntry)
	if !baseOpts.NoCache && store != nil {
		if hits, err := store.GetBatch(cacheKeys); err == nil {
			cachedMap = hits
		}
	}

	var missWorkerItems []infra.WorkerScoreItem
	debias := baseOpts.DebiasOrder
	for _, pq := range prepared {
		cEntry, ok := cachedMap[pq.cacheKey]
		if ok && (pq.qType == "noul" || cEntry.RawLogitsJSON != "") {
			continue
		}
		missWorkerItems = append(missWorkerItems, infra.WorkerScoreItem{
			ID:          pq.qID,
			Context:     stateStr,
			Images:      resolvedSysImages,
			ImageTokens: normSysImgTokens,
			Primitive:   pq.qType,
			Rubrics:     pq.rubrics,
			Choices:     pq.choices,
			ScoreLevels: pq.scoreLevels,
			DebiasOrder: &debias,
		})
	}

	freshByQID := make(map[string]infra.WorkerScoreResult, len(missWorkerItems))
	if len(missWorkerItems) > 0 {
		if baseOpts.CacheOnly {
			return nil, fmt.Errorf("cache miss for %d systemone questions under --cache-only mode", len(missWorkerItems))
		}
		if modelErr != nil {
			return nil, modelErr
		}
		resp, err := infra.ScoreMultiViaMLX(cfg, nil, rm, infra.WorkerMultiParams{
			Primitive:   "noul",
			DebiasOrder: baseOpts.DebiasOrder,
			ImageTokens: normSysImgTokens,
			SlicedHead:  baseOpts.SlicedHead,
		}, missWorkerItems)
		if err != nil {
			return nil, err
		}
		for _, r := range resp.Results {
			freshByQID[r.ID] = r
		}
	}

	answers := make(map[string]types.SystemOneAnswer, len(prepared))
	var toCache []types.CacheEntry
	now := time.Now().Format(time.RFC3339)
	normState := cache.NormalizeText(stateStr)
	stateHash := cache.HashString(normState)

	for _, pq := range prepared {
		cEntry, isCached := cachedMap[pq.cacheKey]
		if isCached && (pq.qType == "noul" || cEntry.RawLogitsJSON != "") {
			switch pq.qType {
			case "noul":
				prob, _, _, _ := CalibrateDecision(cEntry.LogitYes, cEntry.LogitNo, Options{
					Threshold:   baseOpts.Threshold,
					Temperature: pq.temp,
					CalibA:      baseOpts.CalibA,
					CalibB:      baseOpts.CalibB,
				})
				p := cache.Round4(prob)
				answers[pq.qID] = types.SystemOneAnswer{
					Type:      "noul",
					Noul:      &p,
					Cached:    true,
					LatencyMs: 0.05,
				}
			case "choice":
				var rawLogits map[string]float64
				_ = json.Unmarshal([]byte(cEntry.RawLogitsJSON), &rawLogits)
				winner, conf, probs := cache.CalibrateChoiceFromLogits(rawLogits, pq.choices, pq.temp)
				answers[pq.qID] = types.SystemOneAnswer{
					Type:          "choice",
					Choice:        winner,
					Confidence:    &conf,
					Probabilities: probs,
					Cached:        true,
					LatencyMs:     0.05,
				}
			case "score":
				var rawLogits map[string]float64
				_ = json.Unmarshal([]byte(cEntry.RawLogitsJSON), &rawLogits)
				expScore, _, _, _, conf, probs, legend := cache.CalibrateScoreFromLogits(rawLogits, pq.scoreLevels, pq.temp)
				answers[pq.qID] = types.SystemOneAnswer{
					Type:          "score",
					Score:         &expScore,
					Confidence:    &conf,
					Probabilities: probs,
					Legend:        legend,
					Cached:        true,
					LatencyMs:     0.05,
				}
			}
			continue
		}

		fr, ok := freshByQID[pq.qID]
		if !ok {
			return nil, fmt.Errorf("missing result from MLX worker for question %q", pq.qID)
		}
		normRubric := cache.NormalizeText(pq.rubrics)

		switch pq.qType {
		case "noul":
			prob, _, _, _ := CalibrateDecision(fr.LogitYes, fr.LogitNo, Options{
				Threshold:   baseOpts.Threshold,
				Temperature: pq.temp,
				CalibA:      baseOpts.CalibA,
				CalibB:      baseOpts.CalibB,
			})
			defProb, _, _, _ := cache.CalibrateFromLogits(fr.LogitYes, fr.LogitNo, 1.0, 0.5)
			p := cache.Round4(prob)
			answers[pq.qID] = types.SystemOneAnswer{
				Type:      "noul",
				Noul:      &p,
				Cached:    false,
				LatencyMs: cache.Round4(fr.LatencyMs),
			}
			toCache = append(toCache, types.CacheEntry{
				CacheKey:     pq.cacheKey,
				ModelID:      canonicalModelID,
				Primitive:    "noul",
				RubricHash:   cache.HashString(normRubric),
				Rubrics:      normRubric,
				RecordID:     pq.qID,
				ContextHash:  stateHash,
				Context:      normState,
				LogitYes:     fr.LogitYes,
				LogitNo:      fr.LogitNo,
				LogitMargin:  fr.Margin,
				Probability:  defProb,
				LatencyMs:    fr.LatencyMs,
				CreatedAt:    now,
				LastAccessed: now,
			})
		case "choice":
			winner, conf, probs := cache.CalibrateChoiceFromLogits(fr.OptionLogits, pq.choices, pq.temp)
			answers[pq.qID] = types.SystemOneAnswer{
				Type:          "choice",
				Choice:        winner,
				Confidence:    &conf,
				Probabilities: probs,
				Cached:        false,
				LatencyMs:     cache.Round4(fr.LatencyMs),
			}
			rawBytes, _ := json.Marshal(fr.OptionLogits)
			toCache = append(toCache, types.CacheEntry{
				CacheKey:      pq.cacheKey,
				ModelID:       canonicalModelID,
				Primitive:     "choice",
				CriteriaJSON:  pq.critJSON,
				RawLogitsJSON: string(rawBytes),
				RubricHash:    cache.HashString(normRubric),
				Rubrics:       normRubric,
				RecordID:      pq.qID,
				ContextHash:   stateHash,
				Context:       normState,
				Probability:   conf,
				LatencyMs:     fr.LatencyMs,
				CreatedAt:     now,
				LastAccessed:  now,
			})
		case "score":
			expScore, _, _, _, conf, probs, legend := cache.CalibrateScoreFromLogits(fr.OptionLogits, pq.scoreLevels, pq.temp)
			answers[pq.qID] = types.SystemOneAnswer{
				Type:          "score",
				Score:         &expScore,
				Confidence:    &conf,
				Probabilities: probs,
				Legend:        legend,
				Cached:        false,
				LatencyMs:     cache.Round4(fr.LatencyMs),
			}
			rawBytes, _ := json.Marshal(fr.OptionLogits)
			toCache = append(toCache, types.CacheEntry{
				CacheKey:      pq.cacheKey,
				ModelID:       canonicalModelID,
				Primitive:     "score",
				CriteriaJSON:  pq.critJSON,
				RawLogitsJSON: string(rawBytes),
				RubricHash:    cache.HashString(normRubric),
				Rubrics:       normRubric,
				RecordID:      pq.qID,
				ContextHash:   stateHash,
				Context:       normState,
				Probability:   expScore,
				LatencyMs:     fr.LatencyMs,
				CreatedAt:     now,
				LastAccessed:  now,
			})
		}
	}

	if !baseOpts.NoCache && store != nil && len(toCache) > 0 {
		_ = store.PutBatch(toCache)
	}

	return &types.SystemOneResponse{
		Model:       rm.Alias,
		ModelID:     canonicalModelID,
		Answers:     answers,
		TotalTimeMs: cache.Round4(float64(time.Since(t0).Microseconds()) / 1000.0),
	}, nil
}

func parseCriteriaToChoices(criteria any) ([]types.ChoiceOption, error) {
	switch v := criteria.(type) {
	case string:
		return ParseChoicesFlag(v)
	case []any:
		out := make([]types.ChoiceOption, 0, len(v))
		for _, elem := range v {
			switch e := elem.(type) {
			case string:
				out = append(out, types.ChoiceOption{Key: e, Description: e})
			case map[string]any:
				k, _ := e["key"].(string)
				d, _ := e["description"].(string)
				if k != "" {
					out = append(out, types.ChoiceOption{Key: k, Description: d})
				}
			}
		}
		if len(out) >= 2 {
			return out, nil
		}
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := make([]types.ChoiceOption, 0, len(keys))
		for _, k := range keys {
			out = append(out, types.ChoiceOption{
				Key:         k,
				Description: fmt.Sprintf("%v", v[k]),
			})
		}
		if len(out) >= 2 {
			return out, nil
		}
	}
	return nil, fmt.Errorf("choice criteria must be a map of {key: description}, array of strings, or 'key:desc,key2:desc2' string with >= 2 options")
}

func parseCriteriaToScoreLevels(criteria any) ([]string, error) {
	switch v := criteria.(type) {
	case string:
		return ParseScoreFlag(v)
	case []any:
		out := make([]string, 0, len(v))
		for _, elem := range v {
			s := strings.TrimSpace(fmt.Sprintf("%v", elem))
			if s != "" {
				out = append(out, s)
			}
		}
		if len(out) >= 2 && len(out) <= 10 {
			return out, nil
		}
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := make([]string, 0, len(keys))
		for _, k := range keys {
			out = append(out, fmt.Sprintf("%v", v[k]))
		}
		if len(out) >= 2 && len(out) <= 10 {
			return out, nil
		}
	}
	return nil, fmt.Errorf("score criteria must be an array of 2..10 level descriptions or comma-separated string")
}
