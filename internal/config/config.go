package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/cafe3310/decision-cli/internal/types"
	"gopkg.in/yaml.v3"
)

const DefaultConfigPath = "~/.config/decision-cli/config.yaml"

// ExpandPath expands leading "~/" to the user's home directory.
func ExpandPath(p string) string {
	if p == "" {
		return ""
	}
	if p == "~" || strings.HasPrefix(p, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			if p == "~" {
				return home
			}
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

// DefaultRegistry returns the built-in out-of-the-box supported models,
// including per-model empirical calibration temperatures (T_noul) derived from jev-bench.
func DefaultRegistry() map[string]types.ModelSpec {
	return map[string]types.ModelSpec{
		"qwen3-reranker-4b": {
			Alias:                  "qwen3-reranker-4b",
			HFRepoID:               "mlx-community/Qwen3-Reranker-4B-4bit",
			PromptStyle:            "qwen3-reranker",
			PositiveTokens:         []string{"yes", "Yes", "YES", "是"},
			NegativeTokens:         []string{"no", "No", "NO", "否"},
			CalibrationTemperature: 1.0,
			DefaultThreshold:       0.5,
			ApproxMemoryGB:         2.5,
			Description:            "Qwen3-Reranker-4B (4-bit MLX) — Cross-Encoder Pointwise Rubric Scorer",
		},
		"gemma-4-26b-a4b": {
			Alias:                  "gemma-4-26b-a4b",
			HFRepoID:               "mlx-community/gemma-4-26b-a4b-it-4bit",
			PromptStyle:            "gemma4-binary",
			PositiveTokens:         []string{"是", "Yes", "yes", "YES"},
			NegativeTokens:         []string{"否", "No", "no", "NO"},
			CalibrationTemperature: 4.5,
			DefaultThreshold:       0.5,
			ApproxMemoryGB:         14.8,
			Description:            "Gemma-4-26B-A4B-it (4-bit MLX MoE, 3.8B active) — High-Precision Rubric Judge",
		},
		"gemma-4-e4b": {
			Alias:                  "gemma-4-e4b",
			HFRepoID:               "mlx-community/gemma-4-e4b-it-4bit",
			PromptStyle:            "gemma4-binary",
			PositiveTokens:         []string{"是", "Yes", "yes", "YES"},
			NegativeTokens:         []string{"否", "No", "no", "NO"},
			CalibrationTemperature: 3.5,
			DefaultThreshold:       0.5,
			ApproxMemoryGB:         3.2,
			Description:            "Gemma-4-E4B-it (4-bit MLX Dense) — Ultra-Fast Lightweight Judge",
		},
	}
}

// EffectiveTemperature resolves the temperature to use for a given model:
//  1. Explicit CLI --temperature flag (if flagChanged && flagVal > 0)
//  2. Per-model models.registry[alias].calibration_temperature in config.yaml (if > 0)
//  3. Global default_temperature in config.yaml (if > 0)
//  4. Fallback 1.0
func EffectiveTemperature(cfg *types.AppConfig, spec types.ModelSpec, flagVal float64, flagChanged bool) float64 {
	if flagChanged && flagVal > 0 {
		return flagVal
	}
	if spec.CalibrationTemperature > 0 {
		return spec.CalibrationTemperature
	}
	if cfg != nil && cfg.DefaultTemperature > 0 {
		return cfg.DefaultTemperature
	}
	return 1.0
}

// EffectiveThreshold resolves the threshold to use for a given model:
//  1. Explicit CLI --threshold flag (if flagChanged && flagVal > 0)
//  2. Per-model models.registry[alias].default_threshold in config.yaml (if > 0)
//  3. Global default_threshold in config.yaml (if > 0)
//  4. Fallback 0.5
func EffectiveThreshold(cfg *types.AppConfig, spec types.ModelSpec, flagVal float64, flagChanged bool) float64 {
	if flagChanged && flagVal > 0 {
		return flagVal
	}
	if spec.DefaultThreshold > 0 {
		return spec.DefaultThreshold
	}
	if cfg != nil && cfg.DefaultThreshold > 0 {
		return cfg.DefaultThreshold
	}
	return 0.5
}

// DefaultAppConfig returns the default AppConfig structure.
func DefaultAppConfig() types.AppConfig {
	return types.AppConfig{
		DefaultModel:       "gemma-4-26b-a4b",
		DefaultThreshold:   0.5,
		DefaultTemperature: 1.0,
		Runtime: types.RuntimeConfig{
			VenvDir:      "~/.cache/decision-cli/venv",
			PythonBin:    "",
			MLXCommand:   []string{},
			UVBin:        "uv",
			WorkerSocket: "~/.cache/decision-cli/worker.sock",
			IdleTimeout:  "5m",
		},
		Models: types.ModelsConfig{
			Paths: map[string]string{
				"qwen3-reranker-4b": "",
				"gemma-4-26b-a4b":   "",
				"gemma-4-e4b":       "",
			},
			ScanDirs: []string{
				"~/.cache/huggingface/hub",
				"~/models",
				"~/.lmstudio/models",
			},
			Registry: DefaultRegistry(),
		},
		Cache: types.CacheConfig{
			DBPath: "~/.cache/decision-cli/cache.db",
		},
	}
}

// DefaultConfigYAMLTemplate returns a richly commented ~/.config/decision-cli/config.yaml template.
func DefaultConfigYAMLTemplate() string {
	return `# decision-cli Global Configuration File
# Auto-generated on first run. Modify as needed; CLI flags take precedence over this file.

# Default model alias ("gemma-4-26b-a4b", "gemma-4-e4b", or "qwen3-reranker-4b")
default_model: "gemma-4-26b-a4b"
default_threshold: 0.5

# Fallback global temperature if a model does not specify its own calibration_temperature below.
default_temperature: 1.0

# 1. Runtime & MLX Environment Configuration
runtime:
  # Custom Python virtual environment directory containing mlx and mlx-lm.
  # If ~/.cache/decision-cli/venv does not exist, decision-cli also checks system/workspace MLX venvs.
  venv_dir: "~/.cache/decision-cli/venv"

  # Optional: explicit path to a python3 executable that has mlx and mlx-lm installed.
  # Takes precedence over venv_dir when non-empty (e.g. "/Users/you/my-mlx-venv/bin/python3").
  python_bin: ""

  # Optional: custom MLX worker invocation prefix (e.g. ["conda", "run", "-n", "mlx", "python3"]).
  # Takes highest precedence over python_bin and venv_dir when non-empty.
  mlx_command: []

  # uv binary used by 'decision-cli setup' when creating a managed venv
  uv_bin: "uv"

  # Unix Domain Socket path for persistent resident MLX worker daemon
  worker_socket: "~/.cache/decision-cli/worker.sock"
  idle_timeout: "5m"

# 2. Model Path Mapping, Scan Directories & Per-Model Calibration Temperatures
# Note on calibration_temperature:
#   - Instruction-tuned LLMs (like Gemma-4) have sharpened post-RLHF logit margins (|margin| ~ 12-20),
#     causing uncalibrated T=1.0 probabilities to saturate at 0.9999 / 0.0001.
#   - Setting per-model calibration_temperature (e.g. 4.5 for gemma-4-26b-a4b, 3.5 for gemma-4-e4b,
#     1.0 for qwen3-reranker-4b) scales P(yes) = sigmoid((logit_yes - logit_no) / T) into well-calibrated
#     probabilities without changing the P=0.5 decision boundary.
#   - Because SQLite caches raw logits, changing calibration_temperature takes effect in <1ms without re-running inference.
models:
  # Explicit mapping: model-name -> absolute directory path containing model.safetensors & config.json.
  # Takes highest priority over scan_dirs when non-empty.
  paths:
    qwen3-reranker-4b: ""
    gemma-4-26b-a4b: ""
    gemma-4-e4b: ""

  # Automatic scan directories searched in order when models.paths[model-name] is empty.
  # Supports both HuggingFace Hub cache structure (models--org--repo/snapshots/<hash>)
  # and flat directory layouts (<dir>/<org>/<repo> or <dir>/<repo>).
  scan_dirs:
    - "~/.cache/huggingface/hub"
    - "~/models"
    - "~/.lmstudio/models"

  # Model registry metadata, positive/negative tokens, and per-model calibration_temperature
  registry:
    qwen3-reranker-4b:
      alias: "qwen3-reranker-4b"
      hf_repo_id: "mlx-community/Qwen3-Reranker-4B-4bit"
      prompt_style: "qwen3-reranker"
      positive_tokens: ["yes", "Yes", "YES", "是"]
      negative_tokens: ["no", "No", "NO", "否"]
      calibration_temperature: 1.0
      default_threshold: 0.5
      approx_memory_gb: 2.5
      description: "Qwen3-Reranker-4B (4-bit MLX) — Cross-Encoder Pointwise Rubric Scorer"
    gemma-4-26b-a4b:
      alias: "gemma-4-26b-a4b"
      hf_repo_id: "mlx-community/gemma-4-26b-a4b-it-4bit"
      prompt_style: "gemma4-binary"
      positive_tokens: ["是", "Yes", "yes", "YES"]
      negative_tokens: ["否", "No", "no", "NO"]
      calibration_temperature: 4.5
      default_threshold: 0.5
      approx_memory_gb: 14.8
      description: "Gemma-4-26B-A4B-it (4-bit MLX MoE, 3.8B active) — High-Precision Rubric Judge"
    gemma-4-e4b:
      alias: "gemma-4-e4b"
      hf_repo_id: "mlx-community/gemma-4-e4b-it-4bit"
      prompt_style: "gemma4-binary"
      positive_tokens: ["是", "Yes", "yes", "YES"]
      negative_tokens: ["否", "No", "no", "NO"]
      calibration_temperature: 3.5
      default_threshold: 0.5
      approx_memory_gb: 3.2
      description: "Gemma-4-E4B-it (4-bit MLX Dense) — Ultra-Fast Lightweight Judge"

# 3. Deterministic SQLite Content-Addressable Cache
cache:
  db_path: "~/.cache/decision-cli/cache.db"
`
}

// LoadOrInit loads the configuration from configPath (default ~/.config/decision-cli/config.yaml).
// If the file does not exist, it creates parent directories and writes DefaultConfigYAMLTemplate().
func LoadOrInit(configPath string) (*types.AppConfig, string, bool, error) {
	if configPath == "" {
		if envPath := os.Getenv("DECISION_CLI_CONFIG"); envPath != "" {
			configPath = envPath
		} else {
			configPath = DefaultConfigPath
		}
	}
	resolvedPath := ExpandPath(configPath)

	created := false
	if _, err := os.Stat(resolvedPath); os.IsNotExist(err) {
		if mkErr := os.MkdirAll(filepath.Dir(resolvedPath), 0755); mkErr != nil {
			return nil, resolvedPath, false, fmt.Errorf("failed to create config dir: %w", mkErr)
		}
		if wErr := os.WriteFile(resolvedPath, []byte(DefaultConfigYAMLTemplate()), 0644); wErr != nil {
			return nil, resolvedPath, false, fmt.Errorf("failed to write default config: %w", wErr)
		}
		created = true
	}

	cfg := DefaultAppConfig()
	rawBytes, err := os.ReadFile(resolvedPath)
	if err != nil {
		return nil, resolvedPath, created, fmt.Errorf("failed to read config file %s: %w", resolvedPath, err)
	}
	if err := yaml.Unmarshal(rawBytes, &cfg); err != nil {
		return nil, resolvedPath, created, fmt.Errorf("failed to parse config file %s: %w", resolvedPath, err)
	}

	// Merge any missing built-in registry entries or missing calibration_temperature defaults
	defReg := DefaultRegistry()
	if cfg.Models.Registry == nil {
		cfg.Models.Registry = defReg
	} else {
		for k, defSpec := range defReg {
			if existing, ok := cfg.Models.Registry[k]; !ok {
				cfg.Models.Registry[k] = defSpec
			} else {
				if existing.CalibrationTemperature <= 0 {
					existing.CalibrationTemperature = defSpec.CalibrationTemperature
				}
				if existing.DefaultThreshold <= 0 {
					existing.DefaultThreshold = defSpec.DefaultThreshold
				}
				cfg.Models.Registry[k] = existing
			}
		}
	}
	if cfg.Models.Paths == nil {
		cfg.Models.Paths = map[string]string{}
	}
	if len(cfg.Models.ScanDirs) == 0 {
		cfg.Models.ScanDirs = []string{"~/.cache/huggingface/hub", "~/models", "~/.lmstudio/models"}
	}
	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "gemma-4-26b-a4b"
	}
	if cfg.DefaultThreshold <= 0 {
		cfg.DefaultThreshold = 0.5
	}
	if cfg.DefaultTemperature <= 0 {
		cfg.DefaultTemperature = 1.0
	}
	if cfg.Cache.DBPath == "" {
		cfg.Cache.DBPath = "~/.cache/decision-cli/cache.db"
	}
	if cfg.Runtime.WorkerSocket == "" {
		cfg.Runtime.WorkerSocket = "~/.cache/decision-cli/worker.sock"
	}

	return &cfg, resolvedPath, created, nil
}
