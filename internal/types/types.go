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
	CalibA      *float64     `json:"calib_a,omitempty" yaml:"calib_a,omitempty"` // Platt scaling slope A
	CalibB      *float64     `json:"calib_b,omitempty" yaml:"calib_b,omitempty"` // Platt scaling intercept B
	Logits      LogitDetails `json:"logits" yaml:"logits"`
	Model       string       `json:"model" yaml:"model"`
	ModelID     string       `json:"model_id" yaml:"model_id"`
	Images      []string     `json:"images,omitempty" yaml:"images,omitempty"`
	ImageTokens int          `json:"image_tokens,omitempty" yaml:"image_tokens,omitempty"`
	Cached      bool         `json:"cached" yaml:"cached"`
	LatencyMs   float64      `json:"latency_ms" yaml:"latency_ms"`
	CacheKey    string       `json:"cache_key" yaml:"cache_key"`
}

// ChoiceOption defines a single candidate option for the "choice" primitive.
type ChoiceOption struct {
	Key         string `json:"key" yaml:"key"`
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

// SingleChoiceResult is the output for a single --choices evaluation.
type SingleChoiceResult struct {
	Primitive     string             `json:"primitive" yaml:"primitive"`         // "choice"
	Choice        string             `json:"choice" yaml:"choice"`               // winning option key
	Confidence    float64            `json:"confidence" yaml:"confidence"`       // max_i P(k_i)
	Probabilities map[string]float64 `json:"probabilities" yaml:"probabilities"` // key -> calibrated probability
	Logits        map[string]float64 `json:"logits" yaml:"logits"`               // key -> raw log-probability
	DebiasedOrder bool               `json:"debiased_order,omitempty" yaml:"debiased_order,omitempty"`
	Temperature   float64            `json:"temperature" yaml:"temperature"`
	Model         string             `json:"model" yaml:"model"`
	ModelID       string             `json:"model_id" yaml:"model_id"`
	Images        []string           `json:"images,omitempty" yaml:"images,omitempty"`
	ImageTokens   int                `json:"image_tokens,omitempty" yaml:"image_tokens,omitempty"`
	Cached        bool               `json:"cached" yaml:"cached"`
	LatencyMs     float64            `json:"latency_ms" yaml:"latency_ms"`
	CacheKey      string             `json:"cache_key" yaml:"cache_key"`
}

// SingleScoreResult is the output for a single --score ordinal rubric evaluation.
type SingleScoreResult struct {
	Primitive       string             `json:"primitive" yaml:"primitive"`               // "score"
	Score           float64            `json:"score" yaml:"score"`                       // expected value sum(m * P(m)) in [0, M-1]
	NormalizedScore float64            `json:"normalized_score" yaml:"normalized_score"` // score / (M - 1) in [0.0, 1.0]
	TopLevel        int                `json:"top_level" yaml:"top_level"`               // argmax_m P(m)
	TopLabel        string             `json:"top_label" yaml:"top_label"`               // rubric description for TopLevel
	Confidence      float64            `json:"confidence" yaml:"confidence"`             // max_m P(m)
	Probabilities   map[string]float64 `json:"probabilities" yaml:"probabilities"`       // "0": p0, "1": p1, ...
	Legend          map[string]string  `json:"legend" yaml:"legend"`                     // "0": level0, "1": level1, ...
	Logits          map[string]float64 `json:"logits" yaml:"logits"`                     // "0": l0, "1": l1, ...
	DebiasedOrder   bool               `json:"debiased_order,omitempty" yaml:"debiased_order,omitempty"`
	Temperature     float64            `json:"temperature" yaml:"temperature"`
	Model           string             `json:"model" yaml:"model"`
	ModelID         string             `json:"model_id" yaml:"model_id"`
	Images          []string           `json:"images,omitempty" yaml:"images,omitempty"`
	ImageTokens     int                `json:"image_tokens,omitempty" yaml:"image_tokens,omitempty"`
	Cached          bool               `json:"cached" yaml:"cached"`
	LatencyMs       float64            `json:"latency_ms" yaml:"latency_ms"`
	CacheKey        string             `json:"cache_key" yaml:"cache_key"`
}

// BatchItemInput represents a single item extracted from a batch input file.
type BatchItemInput struct {
	ID      string         `json:"id" yaml:"id"`
	Context string         `json:"context" yaml:"context"`
	Images  []string       `json:"images,omitempty" yaml:"images,omitempty"`
	Raw     map[string]any `json:"raw,omitempty" yaml:"raw,omitempty"`
}

// BatchItemResult is the per-record output in batch mode.
type BatchItemResult struct {
	ID              string             `json:"id" yaml:"id"`
	Primitive       string             `json:"primitive,omitempty" yaml:"primitive,omitempty"`
	Decision        string             `json:"decision,omitempty" yaml:"decision,omitempty"` // "是" or "否" (noul), winning key (choice), or top level (score)
	Choice          string             `json:"choice,omitempty" yaml:"choice,omitempty"`
	Score           *float64           `json:"score,omitempty" yaml:"score,omitempty"`
	NormalizedScore *float64           `json:"normalized_score,omitempty" yaml:"normalized_score,omitempty"`
	Passed          bool               `json:"passed,omitempty" yaml:"passed,omitempty"`
	Probability     float64            `json:"probability,omitempty" yaml:"probability,omitempty"`
	Probabilities   map[string]float64 `json:"probabilities,omitempty" yaml:"probabilities,omitempty"`
	Confidence      float64            `json:"confidence" yaml:"confidence"`
	Logits          *LogitDetails      `json:"logits,omitempty" yaml:"logits,omitempty"`
	OptionLogits    map[string]float64 `json:"option_logits,omitempty" yaml:"option_logits,omitempty"`
	Images          []string           `json:"images,omitempty" yaml:"images,omitempty"`
	Cached          bool               `json:"cached" yaml:"cached"`
	LatencyMs       float64            `json:"latency_ms" yaml:"latency_ms"`
	CacheKey        string             `json:"cache_key" yaml:"cache_key"`
	Context         string             `json:"context,omitempty" yaml:"context,omitempty"`
	Record          map[string]any     `json:"record,omitempty" yaml:"record,omitempty"`
}

// BatchSummary summarizes a batch evaluation run.
type BatchSummary struct {
	Total         int            `json:"total" yaml:"total"`
	PassedCount   int            `json:"passed_count,omitempty" yaml:"passed_count,omitempty"`
	RejectedCount int            `json:"rejected_count,omitempty" yaml:"rejected_count,omitempty"`
	ChoiceCounts  map[string]int `json:"choice_counts,omitempty" yaml:"choice_counts,omitempty"`
	MeanScore     *float64       `json:"mean_score,omitempty" yaml:"mean_score,omitempty"`
	CacheHits     int            `json:"cache_hits" yaml:"cache_hits"`
	CacheMisses   int            `json:"cache_misses" yaml:"cache_misses"`
	TotalTimeMs   float64        `json:"total_time_ms" yaml:"total_time_ms"`
}

// BatchDecisionReport is the full output structure for batch mode or cache export.
type BatchDecisionReport struct {
	ExportedAt    string            `json:"exported_at" yaml:"exported_at"`
	Primitive     string            `json:"primitive,omitempty" yaml:"primitive,omitempty"`
	Model         string            `json:"model" yaml:"model"`
	ModelID       string            `json:"model_id" yaml:"model_id"`
	Rubrics       string            `json:"rubrics" yaml:"rubrics"`
	Choices       []ChoiceOption    `json:"choices,omitempty" yaml:"choices,omitempty"`
	ScoreLevels   []string          `json:"score_levels,omitempty" yaml:"score_levels,omitempty"`
	DebiasedOrder bool              `json:"debiased_order,omitempty" yaml:"debiased_order,omitempty"`
	Threshold     float64           `json:"threshold,omitempty" yaml:"threshold,omitempty"`
	Temperature   float64           `json:"temperature" yaml:"temperature"`
	CalibA        *float64          `json:"calib_a,omitempty" yaml:"calib_a,omitempty"`
	CalibB        *float64          `json:"calib_b,omitempty" yaml:"calib_b,omitempty"`
	Summary       BatchSummary      `json:"summary" yaml:"summary"`
	Results       []BatchItemResult `json:"results" yaml:"results"`
}

// SystemOneQuestion represents a single question in a TypeSafe Jev /v1/systemone request.
type SystemOneQuestion struct {
	Type         string `json:"type" yaml:"type"`                 // "noul" | "choice" | "score"
	Instructions string `json:"instructions" yaml:"instructions"` // natural-language prompt/question
	Criteria     any    `json:"criteria,omitempty" yaml:"criteria,omitempty"`
}

// SystemOneRequest represents a TypeSafe Jev /v1/systemone request payload.
type SystemOneRequest struct {
	Model       string                       `json:"model,omitempty" yaml:"model,omitempty"`
	ImageTokens int                          `json:"image_tokens,omitempty" yaml:"image_tokens,omitempty"`
	Images      []string                     `json:"images,omitempty" yaml:"images,omitempty"`
	Assets      map[string]string            `json:"assets,omitempty" yaml:"assets,omitempty"`
	State       any                          `json:"state" yaml:"state"`
	Questions   map[string]SystemOneQuestion `json:"questions" yaml:"questions"`
}

// SystemOneAnswer represents a single evaluated question in a /v1/systemone response.
type SystemOneAnswer struct {
	Type          string             `json:"type" yaml:"type"`
	Noul          *float64           `json:"noul,omitempty" yaml:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty" yaml:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty" yaml:"score,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty" yaml:"confidence,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty" yaml:"probabilities,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty" yaml:"legend,omitempty"`
	Cached        bool               `json:"cached" yaml:"cached"`
	LatencyMs     float64            `json:"latency_ms" yaml:"latency_ms"`
}

// SystemOneResponse represents a TypeSafe Jev /v1/systemone response payload.
type SystemOneResponse struct {
	Model       string                     `json:"model" yaml:"model"`
	ModelID     string                     `json:"model_id" yaml:"model_id"`
	Answers     map[string]SystemOneAnswer `json:"answers" yaml:"answers"`
	TotalTimeMs float64                    `json:"total_time_ms" yaml:"total_time_ms"`
}

// ModelSpec defines metadata, token configuration, and per-model calibration temperature.
type ModelSpec struct {
	Alias                  string   `json:"alias" yaml:"alias"`
	HFRepoID               string   `json:"hf_repo_id" yaml:"hf_repo_id"`
	PromptStyle            string   `json:"prompt_style" yaml:"prompt_style"` // "qwen3-reranker" | "gemma4-binary"
	SupportsVision         bool     `json:"supports_vision,omitempty" yaml:"supports_vision,omitempty"`
	PositiveTokens         []string `json:"positive_tokens" yaml:"positive_tokens"`
	NegativeTokens         []string `json:"negative_tokens" yaml:"negative_tokens"`
	CalibrationTemperature float64  `json:"calibration_temperature" yaml:"calibration_temperature"`
	CalibA                 *float64 `json:"calib_a,omitempty" yaml:"calib_a,omitempty"`
	CalibB                 *float64 `json:"calib_b,omitempty" yaml:"calib_b,omitempty"`
	ChoiceTemperature      float64  `json:"choice_temperature,omitempty" yaml:"choice_temperature,omitempty"`
	ScoreTemperature       float64  `json:"score_temperature,omitempty" yaml:"score_temperature,omitempty"`
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
	CacheKey      string  `json:"cache_key" yaml:"cache_key"`
	ModelID       string  `json:"model_id" yaml:"model_id"`
	Primitive     string  `json:"primitive,omitempty" yaml:"primitive,omitempty"`
	CriteriaJSON  string  `json:"criteria_json,omitempty" yaml:"criteria_json,omitempty"`
	RawLogitsJSON string  `json:"raw_logits_json,omitempty" yaml:"raw_logits_json,omitempty"`
	RubricHash    string  `json:"rubric_hash" yaml:"rubric_hash"`
	Rubrics       string  `json:"rubrics" yaml:"rubrics"`
	RecordID      string  `json:"record_id" yaml:"record_id"`
	ContextHash   string  `json:"context_hash" yaml:"context_hash"`
	Context       string  `json:"context" yaml:"context"`
	LogitYes      float64 `json:"logit_yes" yaml:"logit_yes"`
	LogitNo       float64 `json:"logit_no" yaml:"logit_no"`
	LogitMargin   float64 `json:"logit_margin" yaml:"logit_margin"`
	Probability   float64 `json:"probability" yaml:"probability"`
	LatencyMs     float64 `json:"latency_ms" yaml:"latency_ms"`
	CreatedAt     string  `json:"created_at" yaml:"created_at"`
	HitCount      int     `json:"hit_count" yaml:"hit_count"`
	LastAccessed  string  `json:"last_accessed" yaml:"last_accessed"`
}
