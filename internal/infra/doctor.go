package infra

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/cafe3310/decision-cli/internal/cache"
	"github.com/cafe3310/decision-cli/internal/config"
	"github.com/cafe3310/decision-cli/internal/types"
)

// CanaryCase defines a known-outcome inference test for 'decision-cli doctor'.
type CanaryCase struct {
	Name        string `json:"name" yaml:"name"`
	Rubrics     string `json:"rubrics" yaml:"rubrics"`
	Context     string `json:"context" yaml:"context"`
	ExpectPass  bool   `json:"expect_pass" yaml:"expect_pass"`
	ExpectedStr string `json:"expected_str" yaml:"expected_str"`
}

// CanaryResult holds the outcome of a single canary test.
type CanaryResult struct {
	Name        string  `json:"name" yaml:"name"`
	Rubrics     string  `json:"rubrics" yaml:"rubrics"`
	Context     string  `json:"context" yaml:"context"`
	Expected    string  `json:"expected" yaml:"expected"`
	Actual      string  `json:"actual" yaml:"actual"`
	Probability float64 `json:"probability" yaml:"probability"`
	LogitYes    float64 `json:"logit_yes" yaml:"logit_yes"`
	LogitNo     float64 `json:"logit_no" yaml:"logit_no"`
	Margin      float64 `json:"margin" yaml:"margin"`
	LatencyMs   float64 `json:"latency_ms" yaml:"latency_ms"`
	PassedCheck bool    `json:"passed_check" yaml:"passed_check"`
}

// DefaultCanaryCases returns the 3 unambiguous canary inference test cases.
func DefaultCanaryCases() []CanaryCase {
	return []CanaryCase{
		{
			Name:        "Canary #1 (Positive Match — 文档/API教程诉求)",
			Rubrics:     "判断这些文本是否代表文档开发相关的需求",
			Context:     "希望能提供一份完整的 API 接口参数说明、SDK 快速上手指南和部署教程文档，现在的 README 太简略了。",
			ExpectPass:  true,
			ExpectedStr: "是",
		},
		{
			Name:        "Canary #2 (Negative Mismatch — 纯算力硬件吐槽)",
			Rubrics:     "判断这些文本是否代表文档开发相关的需求",
			Context:     "今天中午食堂的红烧肉太咸了，而且下午天气很热。",
			ExpectPass:  false,
			ExpectedStr: "否",
		},
		{
			Name:        "Canary #3 (Instruction Sensitivity Flip — 商务采购线索)",
			Rubrics:     "判断这些文本是否包含商业采购或企业合作意向",
			Context:     "我们是某股份制银行智能客服团队，计划采购百灵大模型企业版 API，请问商务怎么联系？",
			ExpectPass:  true,
			ExpectedStr: "是",
		},
	}
}

func getSysctlString(key string) string {
	out, err := exec.Command("sysctl", "-n", key).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// RunDoctor executes the 4-stage diagnostic check and prints a clear report.
func RunDoctor(cfg *types.AppConfig, configPath string, configCreated bool, targetModel string) error {
	fmt.Println("================================================================================")
	fmt.Println("decision-cli doctor — Environment, Config & Canary Inference Diagnostics")
	fmt.Println("================================================================================")

	// [1/4] Config & System
	fmt.Println("\n[1/4] Configuration & Apple Silicon Hardware")
	chip := getSysctlString("machdep.cpu.brand_string")
	if chip == "" {
		chip = runtime.GOARCH
	}
	memBytesStr := getSysctlString("hw.memsize")
	fmt.Printf("  - OS / Arch         : %s/%s (%s, mem=%s bytes)\n", runtime.GOOS, runtime.GOARCH, chip, memBytesStr)
	if configCreated {
		fmt.Printf("  - Config File       : %s (auto-created with defaults)\n", configPath)
	} else {
		fmt.Printf("  - Config File       : %s (loaded)\n", configPath)
	}
	fmt.Printf("  - Default Model     : %s (threshold=%.2f, temperature=%.2f)\n", cfg.DefaultModel, cfg.DefaultThreshold, cfg.DefaultTemperature)
	fmt.Printf("  - Cache DB Path     : %s\n", cfg.Cache.DBPath)
	fmt.Printf("  - Configured Scan   : %v\n", cfg.Models.ScanDirs)

	store, err := cache.Open(cfg.Cache.DBPath)
	if err != nil {
		fmt.Printf("  - Cache DB Status   : [FAIL] %v\n", err)
		return err
	}
	stats, err := store.Stats()
	_ = store.Close()
	if err == nil {
		fmt.Printf("  - Cache DB Status   : [OK] %s (%d cached entries, %d total hits)\n", stats.DBPath, stats.TotalEntries, stats.TotalHits)
	}

	// [2/4] Runtime & MLX
	fmt.Println("\n[2/4] MLX & Python Runtime Resolution")
	rt := ResolveRuntime(cfg)
	if !rt.Valid {
		fmt.Printf("  - Runtime Status    : [FAIL] %s\n", rt.Error)
		fmt.Println("  - Remediation       : Set runtime.python_bin or runtime.venv_dir in ~/.config/decision-cli/config.yaml")
		return fmt.Errorf("runtime check failed: %s", rt.Error)
	}
	fmt.Printf("  - Resolved Command  : %s\n", strings.Join(rt.Command, " "))
	fmt.Printf("  - Resolution Source : %s\n", rt.Source)
	fmt.Printf("  - MLX / MLX-LM      : mlx=%s, mlx_lm=%s, device=%s [OK]\n", rt.MLXVersion, rt.MLXLMVer, rt.DeviceInfo)

	// [3/4] Model Resolution
	fmt.Println("\n[3/4] Model Path & Scan Directory Resolution")
	var firstReadyModel *ResolvedModel
	aliases := []string{"gemma-4-26b-a4b", "gemma-4-e4b", "qwen3-reranker-4b"}
	for _, alias := range aliases {
		rm, mErr := ResolveModel(cfg, alias)
		effT := config.EffectiveTemperature(cfg, rm.Spec, 0, false)
		if mErr == nil && rm.CachedOnDisk {
			fmt.Printf("  - %-18s: [READY]   %s (via %s, calibration_temp=%.2f)\n", alias, rm.LocalPath, rm.Source, effT)
			if firstReadyModel == nil || alias == targetModel || (targetModel == "" && alias == cfg.DefaultModel) {
				cp := rm
				if firstReadyModel == nil || alias == targetModel || alias == cfg.DefaultModel {
					firstReadyModel = &cp
				}
			}
		} else {
			fmt.Printf("  - %-18s: [MISSING] (%s, calibration_temp=%.2f) — not found in paths or scan_dirs\n", alias, rm.Spec.HFRepoID, effT)
		}
	}

	if targetModel != "" {
		rm, mErr := ResolveModel(cfg, targetModel)
		if mErr != nil {
			return fmt.Errorf("requested model %q is not available on disk: %w", targetModel, mErr)
		}
		firstReadyModel = &rm
	}

	if firstReadyModel == nil {
		fmt.Println("\n[4/4] Canary Inference Check")
		fmt.Println("  - [SKIPPED] No local model weights found on disk yet.")
		return fmt.Errorf("no local models available for canary inference")
	}

	// [4/4] Canary Inference
	effTemp := config.EffectiveTemperature(cfg, firstReadyModel.Spec, 0, false)
	effThresh := config.EffectiveThreshold(cfg, firstReadyModel.Spec, 0, false)
	fmt.Printf("\n[4/4] Canary Inference Verification (Model: %s -> %s, calibration_temp=%.2f, threshold=%.2f)\n",
		firstReadyModel.Alias, firstReadyModel.Spec.HFRepoID, effTemp, effThresh)
	canaries := DefaultCanaryCases()
	allPassed := true

	for idx, c := range canaries {
		t0 := time.Now()
		resp, sErr := ScoreBatchViaMLX(cfg, rt, *firstReadyModel, c.Rubrics, []WorkerScoreItem{
			{ID: fmt.Sprintf("canary-%d", idx+1), Context: c.Context},
		})
		elapsed := float64(time.Since(t0).Milliseconds())
		if sErr != nil {
			fmt.Printf("  - %s: [ERROR] %v\n", c.Name, sErr)
			allPassed = false
			continue
		}
		if len(resp.Results) == 0 {
			fmt.Printf("  - %s: [ERROR] empty result from worker\n", c.Name)
			allPassed = false
			continue
		}
		r := resp.Results[0]
		prob, _, passed, decision := cache.CalibrateFromLogits(r.LogitYes, r.LogitNo, effTemp, effThresh)
		checkOK := (passed == c.ExpectPass)
		statusTag := "PASS"
		if !checkOK {
			statusTag = "FAIL"
			allPassed = false
		}
		fmt.Printf("  - [%s] %s\n", statusTag, c.Name)
		fmt.Printf("      Rubric   : %s\n", c.Rubrics)
		fmt.Printf("      Context  : %s\n", c.Context)
		fmt.Printf("      Expected : %s | Actual: %s (P(yes)=%.4f, logit_yes=%.3f, logit_no=%.3f, margin=%+.3f, worker=%.1fms, total=%.0fms)\n",
			c.ExpectedStr, decision, prob, r.LogitYes, r.LogitNo, r.Margin, r.LatencyMs, elapsed)
	}

	fmt.Println("\n--------------------------------------------------------------------------------")
	if allPassed {
		fmt.Println("DOCTOR RESULT: HEALTHY — All environment, config, and canary inference checks passed.")
		return nil
	}
	fmt.Println("DOCTOR RESULT: WARNING — One or more canary checks did not match expected polarity.")
	return fmt.Errorf("canary verification failed")
}
