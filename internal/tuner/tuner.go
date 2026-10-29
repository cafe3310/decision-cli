package tuner

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// TuneSample represents a single data point for threshold and calibration tuning.
type TuneSample struct {
	ID       string  `json:"id"`
	Margin   float64 `json:"margin"` // logitYes - logitNo
	Label    float64 `json:"label"`  // 1.0 (positive/pass/failure depending on task) or 0.0
	LogitYes float64 `json:"logit_yes"`
	LogitNo  float64 `json:"logit_no"`
}

// Metrics captures performance and calibration quality.
type Metrics struct {
	Accuracy  float64 `json:"accuracy"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	Brier     float64 `json:"brier"` // (1/N) * sum((p - y)^2) [0.0 is perfect]
	ECE       float64 `json:"ece"`   // Expected Calibration Error (10 bins)
}

// FoldResult captures evaluation metrics on a single validation fold.
type FoldResult struct {
	FoldIndex       int     `json:"fold_index"`
	DefaultMetrics  Metrics `json:"default_metrics"`
	TunedMetrics    Metrics `json:"tuned_metrics"`
	OptimalThresh   float64 `json:"optimal_thresh"`
	PlattA          float64 `json:"platt_a"`
	PlattB          float64 `json:"platt_b"`
	MetRequirements bool    `json:"met_requirements"`
}

// TuneReport is the overall output of the tuning process.
type TuneReport struct {
	TotalSamples     int          `json:"total_samples"`
	PositiveSamples  int          `json:"positive_samples"`
	NegativeSamples  int          `json:"negative_samples"`
	BaseRate         float64      `json:"base_rate"`
	Folds            []FoldResult `json:"folds"`
	MeanDefault      Metrics      `json:"mean_default"`
	MeanTuned        Metrics      `json:"mean_tuned"`
	RecommendedThresh float64     `json:"recommended_threshold"`
	RecommendedCalibA float64     `json:"recommended_calib_a"`
	RecommendedCalibB float64     `json:"recommended_calib_b"`
	AllFoldsQualified bool        `json:"all_folds_qualified"`
}

// TuneOptions controls the tuning constraints.
type TuneOptions struct {
	MinPrecision float64
	MinRecall    float64
	Beta         float64 // 1.0 for F1, 0.5 for F0.5 (favor precision), 2.0 for F2 (favor recall)
	DefaultTemp  float64 // baseline temperature, e.g. 4.5 or 1.0
	NumFolds     int     // default 5
}

func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1.0 / (1.0 + math.Exp(-z))
	}
	ez := math.Exp(z)
	return ez / (1.0 + ez)
}

// FitPlattScaling fits (A, B) such that sigmoid(A * margin + B) matches label via regularized logistic regression.
func FitPlattScaling(margins []float64, labels []float64, initialA, initialB float64) (float64, float64) {
	n := len(margins)
	if n == 0 {
		return initialA, initialB
	}
	a := initialA
	b := initialB
	lr := 0.05
	lambda := 1e-4 // small L2 regularization

	for iter := 0; iter < 200; iter++ {
		var gradA, gradB float64
		for i := 0; i < n; i++ {
			p := sigmoid(a*margins[i] + b)
			diff := p - labels[i]
			gradA += diff * margins[i]
			gradB += diff
		}
		gradA = (gradA / float64(n)) + lambda*a
		gradB = (gradB / float64(n))
		a -= lr * gradA
		b -= lr * gradB
	}
	return a, b
}

// EvaluateMetrics computes accuracy, precision, recall, F1, Brier, and ECE.
func EvaluateMetrics(probs []float64, labels []float64, threshold float64) Metrics {
	n := len(probs)
	if n == 0 {
		return Metrics{}
	}
	var tp, fp, tn, fn float64
	var brierSum float64

	for i := 0; i < n; i++ {
		p := probs[i]
		y := labels[i]
		brierSum += (p - y) * (p - y)
		pred := p >= threshold
		if pred && y > 0.5 {
			tp++
		} else if pred && y <= 0.5 {
			fp++
		} else if !pred && y <= 0.5 {
			tn++
		} else {
			fn++
		}
	}

	acc := (tp + tn) / float64(n)
	var prec, rec, f1 float64
	if tp+fp > 0 {
		prec = tp / (tp + fp)
	}
	if tp+fn > 0 {
		rec = tp / (tp + fn)
	}
	if prec+rec > 0 {
		f1 = 2 * (prec * rec) / (prec + rec)
	}

	// 10-bin ECE
	numBins := 10
	binCounts := make([]int, numBins)
	binAcc := make([]float64, numBins)
	binConf := make([]float64, numBins)

	for i := 0; i < n; i++ {
		p := probs[i]
		y := labels[i]
		binIdx := int(p * float64(numBins))
		if binIdx >= numBins {
			binIdx = numBins - 1
		}
		binCounts[binIdx]++
		binAcc[binIdx] += y
		binConf[binIdx] += p
	}

	var ece float64
	for b := 0; b < numBins; b++ {
		if binCounts[b] > 0 {
			accB := binAcc[b] / float64(binCounts[b])
			confB := binConf[b] / float64(binCounts[b])
			ece += (float64(binCounts[b]) / float64(n)) * math.Abs(accB-confB)
		}
	}

	return Metrics{
		Accuracy:  math.Round(acc*10000) / 10000,
		Precision: math.Round(prec*10000) / 10000,
		Recall:    math.Round(rec*10000) / 10000,
		F1:        math.Round(f1*10000) / 10000,
		Brier:     math.Round((brierSum/float64(n))*10000) / 10000,
		ECE:       math.Round(ece*10000) / 10000,
	}
}

// TuneOptimalThreshold searches for the optimal threshold on training data subject to constraints.
func TuneOptimalThreshold(margins []float64, labels []float64, a, b float64, opts TuneOptions) (float64, bool) {
	n := len(margins)
	probs := make([]float64, n)
	for i := 0; i < n; i++ {
		probs[i] = sigmoid(a*margins[i] + b)
	}

	beta2 := opts.Beta * opts.Beta
	if beta2 <= 0 {
		beta2 = 1.0
	}

	bestThresh := 0.5
	bestScore := -1.0
	qualified := false

	// Grid search over fine thresholds [0.02, 0.98]
	for t := 0.02; t <= 0.98; t += 0.01 {
		m := EvaluateMetrics(probs, labels, t)
		meetsReq := (m.Precision >= opts.MinPrecision) && (m.Recall >= opts.MinRecall)

		fBeta := 0.0
		if (beta2*m.Precision + m.Recall) > 0 {
			fBeta = (1 + beta2) * (m.Precision * m.Recall) / (beta2*m.Precision + m.Recall)
		}

		if meetsReq {
			if !qualified || fBeta > bestScore {
				qualified = true
				bestScore = fBeta
				bestThresh = t
			}
		} else if !qualified && fBeta > bestScore {
			// Fallback unconstrained best
			bestScore = fBeta
			bestThresh = t
		}
	}

	return math.Round(bestThresh*100) / 100, qualified
}

// RunCrossValidation performs stratified 5-fold cross-validation tuning.
func RunCrossValidation(samples []TuneSample, opts TuneOptions) (*TuneReport, error) {
	if len(samples) < 5 {
		return nil, fmt.Errorf("insufficient samples for 5-fold CV (need at least 5, got %d)", len(samples))
	}
	if opts.NumFolds <= 0 {
		opts.NumFolds = 5
	}
	if opts.Beta <= 0 {
		opts.Beta = 1.0
	}
	if opts.DefaultTemp <= 0 {
		opts.DefaultTemp = 1.0
	}

	var pos, neg []TuneSample
	for _, s := range samples {
		if s.Label > 0.5 {
			pos = append(pos, s)
		} else {
			neg = append(neg, s)
		}
	}

	r := rand.New(rand.NewSource(42))
	r.Shuffle(len(pos), func(i, j int) { pos[i], pos[j] = pos[j], pos[i] })
	r.Shuffle(len(neg), func(i, j int) { neg[i], neg[j] = neg[j], neg[i] })

	k := opts.NumFolds
	folds := make([]FoldResult, k)
	allQualified := true

	var sumDef, sumTuned Metrics
	var sumThresh, sumA, sumB float64

	for f := 0; f < k; f++ {
		var train, val []TuneSample

		for i, s := range pos {
			if i%k == f {
				val = append(val, s)
			} else {
				train = append(train, s)
			}
		}
		for i, s := range neg {
			if i%k == f {
				val = append(val, s)
			} else {
				train = append(train, s)
			}
		}

		trainMargins := make([]float64, len(train))
		trainLabels := make([]float64, len(train))
		for i, s := range train {
			trainMargins[i] = s.Margin
			trainLabels[i] = s.Label
		}

		valMargins := make([]float64, len(val))
		valLabels := make([]float64, len(val))
		for i, s := range val {
			valMargins[i] = s.Margin
			valLabels[i] = s.Label
		}

		// 1. Evaluate baseline (default threshold 0.50, A = 1/DefaultTemp, B = 0)
		baseA := 1.0 / opts.DefaultTemp
		baseProbs := make([]float64, len(val))
		for i, m := range valMargins {
			baseProbs[i] = sigmoid(baseA * m)
		}
		defMet := EvaluateMetrics(baseProbs, valLabels, 0.50)

		// 2. Fit Platt Scaling on Train
		fitA, fitB := FitPlattScaling(trainMargins, trainLabels, baseA, 0.0)

		// 3. Search Optimal Threshold on Train with Platt probabilities
		optThresh, qualified := TuneOptimalThreshold(trainMargins, trainLabels, fitA, fitB, opts)
		if !qualified {
			allQualified = false
		}

		// 4. Evaluate Tuned on Validation Fold
		tunedProbs := make([]float64, len(val))
		for i, m := range valMargins {
			tunedProbs[i] = sigmoid(fitA*m + fitB)
		}
		tunedMet := EvaluateMetrics(tunedProbs, valLabels, optThresh)

		folds[f] = FoldResult{
			FoldIndex:       f + 1,
			DefaultMetrics:  defMet,
			TunedMetrics:    tunedMet,
			OptimalThresh:   optThresh,
			PlattA:          math.Round(fitA*10000) / 10000,
			PlattB:          math.Round(fitB*10000) / 10000,
			MetRequirements: qualified,
		}

		sumDef.Accuracy += defMet.Accuracy
		sumDef.Precision += defMet.Precision
		sumDef.Recall += defMet.Recall
		sumDef.F1 += defMet.F1
		sumDef.Brier += defMet.Brier
		sumDef.ECE += defMet.ECE

		sumTuned.Accuracy += tunedMet.Accuracy
		sumTuned.Precision += tunedMet.Precision
		sumTuned.Recall += tunedMet.Recall
		sumTuned.F1 += tunedMet.F1
		sumTuned.Brier += tunedMet.Brier
		sumTuned.ECE += tunedMet.ECE

		sumThresh += optThresh
		sumA += fitA
		sumB += fitB
	}

	fk := float64(k)
	meanDef := Metrics{
		Accuracy:  math.Round((sumDef.Accuracy/fk)*10000) / 10000,
		Precision: math.Round((sumDef.Precision/fk)*10000) / 10000,
		Recall:    math.Round((sumDef.Recall/fk)*10000) / 10000,
		F1:        math.Round((sumDef.F1/fk)*10000) / 10000,
		Brier:     math.Round((sumDef.Brier/fk)*10000) / 10000,
		ECE:       math.Round((sumDef.ECE/fk)*10000) / 10000,
	}
	meanTuned := Metrics{
		Accuracy:  math.Round((sumTuned.Accuracy/fk)*10000) / 10000,
		Precision: math.Round((sumTuned.Precision/fk)*10000) / 10000,
		Recall:    math.Round((sumTuned.Recall/fk)*10000) / 10000,
		F1:        math.Round((sumTuned.F1/fk)*10000) / 10000,
		Brier:     math.Round((sumTuned.Brier/fk)*10000) / 10000,
		ECE:       math.Round((sumTuned.ECE/fk)*10000) / 10000,
	}

	total := len(samples)
	baseRate := float64(len(pos)) / float64(total)

	return &TuneReport{
		TotalSamples:      total,
		PositiveSamples:   len(pos),
		NegativeSamples:   len(neg),
		BaseRate:          math.Round(baseRate*10000) / 10000,
		Folds:             folds,
		MeanDefault:       meanDef,
		MeanTuned:         meanTuned,
		RecommendedThresh: math.Round((sumThresh/fk)*100) / 100,
		RecommendedCalibA: math.Round((sumA/fk)*10000) / 10000,
		RecommendedCalibB: math.Round((sumB/fk)*10000) / 10000,
		AllFoldsQualified: allQualified,
	}, nil
}

// LoadSamplesFromFile parses JSON, JSONL, or YAML batch/eval files containing logits and labels.
func LoadSamplesFromFile(path string, labelField string) ([]TuneSample, error) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	ext := strings.ToLower(filepath.Ext(path))

	var rawList []map[string]any

	if ext == ".jsonl" {
		lines := strings.Split(string(bytes), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			var m map[string]any
			if err := json.Unmarshal([]byte(line), &m); err == nil {
				rawList = append(rawList, m)
			}
		}
	} else if ext == ".json" {
		var singleMap map[string]any
		if err := json.Unmarshal(bytes, &singleMap); err == nil {
			if res, ok := singleMap["results"].([]any); ok {
				for _, r := range res {
					if rm, ok := r.(map[string]any); ok {
						rawList = append(rawList, rm)
					}
				}
			} else if modelsMap, ok := singleMap["models"].(map[string]any); ok {
				for _, mData := range modelsMap {
					if mObj, ok := mData.(map[string]any); ok {
						if res, ok := mObj["results"].([]any); ok {
							for _, r := range res {
								if rm, ok := r.(map[string]any); ok {
									rawList = append(rawList, rm)
								}
							}
						}
					}
				}
			}
		}
		if len(rawList) == 0 {
			_ = json.Unmarshal(bytes, &rawList)
		}
	} else if ext == ".yaml" || ext == ".yml" {
		var singleMap map[string]any
		if err := yaml.Unmarshal(bytes, &singleMap); err == nil {
			if res, ok := singleMap["results"].([]any); ok {
				for _, r := range res {
					if rm, ok := r.(map[string]any); ok {
						rawList = append(rawList, rm)
					}
				}
			}
		}
		if len(rawList) == 0 {
			_ = yaml.Unmarshal(bytes, &rawList)
		}
	}

	if len(rawList) == 0 {
		return nil, fmt.Errorf("no usable items found in %s", path)
	}

	var samples []TuneSample
	for i, item := range rawList {
		id := fmt.Sprintf("%d", i+1)
		if v, ok := item["id"].(string); ok && v != "" {
			id = v
		}

		// Extract margin / logits
		var margin float64
		var hasMargin bool
		var lYes, lNo float64

		if lMap, ok := item["logits"].(map[string]any); ok {
			if mVal, ok := lMap["margin"].(float64); ok {
				margin = mVal
				hasMargin = true
			}
			if yVal, ok := lMap["yes"].(float64); ok {
				lYes = yVal
			}
			if nVal, ok := lMap["no"].(float64); ok {
				lNo = nVal
			}
			if !hasMargin {
				margin = lYes - lNo
				hasMargin = true
			}
		} else if mVal, ok := item["logit_margin"].(float64); ok {
			margin = mVal
			hasMargin = true
		} else if mVal, ok := item["margin"].(float64); ok {
			margin = mVal
			hasMargin = true
		}

		if !hasMargin {
			continue
		}

		// Extract label
		var label float64
		var hasLabel bool

		candidateFields := []string{"label", "target", "gold", "ground_truth", "expected", "expected_decision"}
		if labelField != "" {
			candidateFields = append([]string{labelField}, candidateFields...)
		}

		for _, k := range candidateFields {
			if val, ok := item[k]; ok {
				switch v := val.(type) {
				case float64:
					label = v
					hasLabel = true
				case int:
					label = float64(v)
					hasLabel = true
				case bool:
					if v {
						label = 1.0
					} else {
						label = 0.0
					}
					hasLabel = true
				case string:
					s := strings.ToLower(strings.TrimSpace(v))
					if s == "1" || s == "true" || s == "是" || s == "yes" || s == "pass" || s == "fail_found" {
						label = 1.0
						hasLabel = true
					} else if s == "0" || s == "false" || s == "否" || s == "no" || s == "fail" {
						label = 0.0
						hasLabel = true
					}
				}
				if hasLabel {
					break
				}
			}
		}

		if hasLabel {
			samples = append(samples, TuneSample{
				ID:       id,
				Margin:   margin,
				Label:    label,
				LogitYes: lYes,
				LogitNo:  lNo,
			})
		}
	}

	if len(samples) == 0 {
		return nil, fmt.Errorf("found items with logits, but no ground-truth label fields (e.g. 'label', 'target', 'expected') matched")
	}

	return samples, nil
}
