package tuner

import (
	"math"
	"testing"
)

func TestFitPlattScalingAndCrossValidation(t *testing.T) {
	// Synthesize 50 samples: 25 positive with positive margins, 25 negative with negative margins
	var samples []TuneSample
	for i := 0; i < 25; i++ {
		samples = append(samples, TuneSample{
			ID:     "pos",
			Margin: 5.0 + float64(i)*0.2,
			Label:  1.0,
		})
		samples = append(samples, TuneSample{
			ID:     "neg",
			Margin: -5.0 - float64(i)*0.2,
			Label:  0.0,
		})
	}

	report, err := RunCrossValidation(samples, TuneOptions{
		MinPrecision: 0.8,
		MinRecall:    0.8,
		Beta:         1.0,
		DefaultTemp:  4.5,
		NumFolds:     5,
	})
	if err != nil {
		t.Fatalf("RunCrossValidation failed: %v", err)
	}

	if report.TotalSamples != 50 {
		t.Errorf("expected 50 samples, got %d", report.TotalSamples)
	}
	if report.MeanTuned.Accuracy < 0.95 {
		t.Errorf("expected high accuracy, got %f", report.MeanTuned.Accuracy)
	}
	if report.MeanTuned.Brier > 0.05 {
		t.Errorf("expected low brier score, got %f", report.MeanTuned.Brier)
	}
	if !report.AllFoldsQualified {
		t.Errorf("expected all folds to meet 80/80 requirements")
	}
}

func TestEvaluateMetrics(t *testing.T) {
	probs := []float64{0.9, 0.8, 0.2, 0.1}
	labels := []float64{1.0, 1.0, 0.0, 0.0}
	m := EvaluateMetrics(probs, labels, 0.5)

	if m.Accuracy != 1.0 {
		t.Errorf("expected accuracy 1.0, got %f", m.Accuracy)
	}
	if m.Precision != 1.0 || m.Recall != 1.0 || m.F1 != 1.0 {
		t.Errorf("expected perfect P/R/F1, got P=%f, R=%f, F1=%f", m.Precision, m.Recall, m.F1)
	}
	if math.Abs(m.Brier-0.025) > 1e-3 {
		t.Errorf("expected brier 0.025, got %f", m.Brier)
	}
}
