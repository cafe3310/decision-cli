package types

// LogitDetails holds raw yes/no log-probabilities and their margin (yes - no).
type LogitDetails struct {
	Yes    float64 `json:"yes" yaml:"yes"`
	No     float64 `json:"no" yaml:"no"`
	Margin float64 `json:"margin" yaml:"margin"`
}

// SingleDecisionResult is the standard output for a single --rubrics + --context evaluation.
type SingleDecisionResult struct {
	Decision    string       `json:"decision" yaml:"decision"`       // "是" or "否"
	Passed      bool         `json:"passed" yaml:"passed"`           // probability >= threshold
	Probability float64      `json:"probability" yaml:"probability"` // [0.0, 1.0] calibrated P(yes)
	Confidence  float64      `json:"confidence" yaml:"confidence"`   // max(p, 1-p)
	Threshold   float64      `json:"threshold" yaml:"threshold"`     // active threshold (default 0.5)
	Temperature float64      `json:"temperature" yaml:"temperature"` // active temperature (default 1.0)
	Logits      LogitDetails `json:"logits" yaml:"logits"`
	Model       string       `json:"model" yaml:"model"`
	ModelID     string       `json:"model_id" yaml:"model_id"`
	Cached      bool         `json:"cached" yaml:"cached"`
	LatencyMs   float64      `json:"latency_ms" yaml:"latency_ms"`
	CacheKey    string       `json:"cache_key" yaml:"cache_key"`
}

// BatchItemInput represents a single item extracted from a batch input file.
type BatchItemInput struct {
	ID      string         `json:"id" yaml:"id"`
	Context string         `json:"context" yaml:"context"`
	Raw     map[string]any `json:"raw,omitempty" yaml:"raw,omitempty"`
}

// BatchItemResult is the per-record output in batch mode.
type BatchItemResult struct {
	ID          string         `json:"id" yaml:"id"`
	Decision    string         `json:"decision" yaml:"decision"` // "是" or "否"
	Passed      bool           `json:"passed" yaml:"passed"`
	Probability float64        `json:"probability" yaml:"probability"`
	Confidence  float64        `json:"confidence" yaml:"confidence"`
	Logits      LogitDetails   `json:"logits" yaml:"logits"`
	Cached      bool           `json:"cached" yaml:"cached"`
	LatencyMs   float64        `json:"latency_ms" yaml:"latency_ms"`
	CacheKey    string         `json:"cache_key" yaml:"cache_key"`
	Context     string         `json:"context,omitempty" yaml:"context,omitempty"`
	Record      map[string]any `json:"record,omitempty" yaml:"record,omitempty"`
}

// BatchSummary summarizes a batch evaluation run.
type BatchSummary struct {
	Total         int     `json:"total" yaml:"total"`
	PassedCount   int     `json:"passed_count" yaml:"passed_count"`
	RejectedCount int     `json:"rejected_count" yaml:"rejected_count"`
	CacheHits     int     `json:"cache_hits" yaml:"cache_hits"`
	CacheMisses   int     `json:"cache_misses" yaml:"cache_misses"`
	TotalTimeMs   float64 `json:"total_time_ms" yaml:"total_time_ms"`
}

// BatchDecisionReport is the full output structure for batch mode or cache export.
type BatchDecisionReport struct {
	ExportedAt  string            `json:"exported_at" yaml:"exported_at"`
	Model       string            `json:"model" yaml:"model"`
	ModelID     string            `json:"model_id" yaml:"model_id"`
	Rubrics     string            `json:"rubrics" yaml:"rubrics"`
	Threshold   float64           `json:"threshold" yaml:"threshold"`
	Temperature float64           `json:"temperature" yaml:"temperature"`
	Summary     BatchSummary      `json:"summary" yaml:"summary"`
	Results     []BatchItemResult `json:"results" yaml:"results"`
}

// ModelSpec defines metadata, token configuration, and per-model calibration temperature.
type ModelSpec struct {
	Alias                  string   `json:"alias" yaml:"alias"`
	HFRepoID               string   `json:"hf_repo_id" yaml:"hf_repo_id"`
	PromptStyle            string   `json:"prompt_style" yaml:"prompt_style"` // "qwen3-reranker" | "gemma4-binary"
	PositiveTokens         []string `json:"positive_tokens" yaml:"positive_tokens"`
	NegativeTokens         []string `json:"negative_tokens" yaml:"negative_tokens"`
	CalibrationTemperature float64  `json:"calibration_temperature" yaml:"calibration_temperature"`
	DefaultThreshold       float64  `json:"default_threshold,omitempty" yaml:"default_threshold,omitempty"`
	ApproxMemoryGB         float64  `json:"approx_memory_gb" yaml:"approx_memory_gb"`
	Description            string   `json:"description" yaml:"description"`
}

// RuntimeConfig configures how the MLX Python worker is invoked.
type RuntimeConfig struct {
	VenvDir      string   `json:"venv_dir" yaml:"venv_dir"`
	PythonBin    string   `json:"python_bin" yaml:"python_bin"`
	MLXCommand   []string `json:"mlx_command" yaml:"mlx_command"`
	UVBin        string   `json:"uv_bin" yaml:"uv_bin"`
	WorkerSocket string   `json:"worker_socket" yaml:"worker_socket"`
	IdleTimeout  string   `json:"idle_timeout" yaml:"idle_timeout"`
}

// ModelsConfig configures explicit model paths, scan directories, and registry.
type ModelsConfig struct {
	Paths    map[string]string    `json:"paths" yaml:"paths"`
	ScanDirs []string             `json:"scan_dirs" yaml:"scan_dirs"`
	Registry map[string]ModelSpec `json:"registry" yaml:"registry"`
}

// CacheConfig configures the SQLite cache file path.
type CacheConfig struct {
	DBPath string `json:"db_path" yaml:"db_path"`
}

// AppConfig represents ~/.config/decision-cli/config.yaml.
type AppConfig struct {
	DefaultModel       string        `json:"default_model" yaml:"default_model"`
	DefaultThreshold   float64       `json:"default_threshold" yaml:"default_threshold"`
	DefaultTemperature float64       `json:"default_temperature" yaml:"default_temperature"`
	Runtime            RuntimeConfig `json:"runtime" yaml:"runtime"`
	Models             ModelsConfig  `json:"models" yaml:"models"`
	Cache              CacheConfig   `json:"cache" yaml:"cache"`
}

// CacheEntry represents a row in the decision_cache SQLite table.
type CacheEntry struct {
	CacheKey     string  `json:"cache_key" yaml:"cache_key"`
	ModelID      string  `json:"model_id" yaml:"model_id"`
	RubricHash   string  `json:"rubric_hash" yaml:"rubric_hash"`
	Rubrics      string  `json:"rubrics" yaml:"rubrics"`
	RecordID     string  `json:"record_id" yaml:"record_id"`
	ContextHash  string  `json:"context_hash" yaml:"context_hash"`
	Context      string  `json:"context" yaml:"context"`
	LogitYes     float64 `json:"logit_yes" yaml:"logit_yes"`
	LogitNo      float64 `json:"logit_no" yaml:"logit_no"`
	LogitMargin  float64 `json:"logit_margin" yaml:"logit_margin"`
	Probability  float64 `json:"probability" yaml:"probability"`
	LatencyMs    float64 `json:"latency_ms" yaml:"latency_ms"`
	CreatedAt    string  `json:"created_at" yaml:"created_at"`
	HitCount     int     `json:"hit_count" yaml:"hit_count"`
	LastAccessed string  `json:"last_accessed" yaml:"last_accessed"`
}
