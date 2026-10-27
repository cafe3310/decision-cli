package infra

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/cafe3310/decision-cli/internal/config"
	"github.com/cafe3310/decision-cli/internal/types"
)

//go:embed mlx_worker.py
var embeddedWorkerPy []byte

// ResolvedRuntime holds the discovered MLX execution command and environment metadata.
type ResolvedRuntime struct {
	Command    []string `json:"command" yaml:"command"`
	Source     string   `json:"source" yaml:"source"`
	MLXVersion string   `json:"mlx_version" yaml:"mlx_version"`
	MLXLMVer   string   `json:"mlx_lm_version" yaml:"mlx_lm_version"`
	DeviceInfo string   `json:"device_info" yaml:"device_info"`
	Valid      bool     `json:"valid" yaml:"valid"`
	Error      string   `json:"error,omitempty" yaml:"error,omitempty"`
}

// ResolvedModel holds the discovered local path and metadata for a model alias.
type ResolvedModel struct {
	Alias       string          `json:"alias" yaml:"alias"`
	Spec        types.ModelSpec `json:"spec" yaml:"spec"`
	LocalPath   string          `json:"local_path" yaml:"local_path"`
	Source      string          `json:"source" yaml:"source"`
	CachedOnDisk bool           `json:"cached_on_disk" yaml:"cached_on_disk"`
}

// ProbeMLXCommand checks whether cmdPrefix (e.g. ["/path/to/python3"]) has mlx, mlx_lm, and PIL installed.
func ProbeMLXCommand(cmdPrefix []string) (mlxVer, mlxLMVer, device string, ok bool) {
	if len(cmdPrefix) == 0 {
		return "", "", "", false
	}
	probeScript := `import json, mlx.core as mx, mlx_lm, PIL
print(json.dumps({
  "mlx": getattr(mx, "__version__", "unknown"),
  "mlx_lm": getattr(mlx_lm, "__version__", "unknown"),
  "device": str(mx.default_device())
}))`
	args := append(append([]string{}, cmdPrefix[1:]...), "-c", probeScript)
	cmd := exec.Command(cmdPrefix[0], args...)
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return "", "", "", false
	}
	var parsed struct {
		MLX    string `json:"mlx"`
		MLXLM  string `json:"mlx_lm"`
		Device string `json:"device"`
	}
	if err := json.Unmarshal(bytes.TrimSpace(out.Bytes()), &parsed); err != nil {
		return "", "", "", false
	}
	return parsed.MLX, parsed.MLXLM, parsed.Device, true
}

// EnsureManagedVenv creates or repairs the self-contained Python virtual environment
// at cfg.Runtime.VenvDir (default ~/.cache/decision-cli/venv) using uv.
func EnsureManagedVenv(cfg *types.AppConfig) (string, error) {
	venvDir := cfg.Runtime.VenvDir
	if strings.TrimSpace(venvDir) == "" {
		venvDir = "~/.cache/decision-cli/venv"
	}
	expandedVenv := config.ExpandPath(venvDir)
	venvPy := filepath.Join(expandedVenv, "bin", "python3")

	if _, err := os.Stat(venvPy); err == nil {
		if _, _, _, ok := ProbeMLXCommand([]string{venvPy}); ok {
			return venvPy, nil
		}
	}

	uvBin := strings.TrimSpace(cfg.Runtime.UVBin)
	if uvBin == "" {
		uvBin = "uv"
	}
	foundUV := false
	if _, err := exec.LookPath(uvBin); err == nil {
		foundUV = true
	} else {
		// Check common uv locations on macOS
		for _, cand := range []string{"/opt/homebrew/bin/uv", filepath.Join(config.ExpandPath("~"), ".local/bin/uv"), filepath.Join(config.ExpandPath("~"), ".cargo/bin/uv")} {
			if _, sErr := os.Stat(cand); sErr == nil {
				uvBin = cand
				foundUV = true
				break
			}
		}
	}
	if !foundUV {
		return "", fmt.Errorf("未检测到 uv 工具（decision-cli 运行时强依赖 uv 管理自包含 Python 虚拟环境）。\n建议安装 uv 后重试：\n  brew install uv  # 或运行: curl -LsSf https://astral.sh/uv/install.sh | sh\n安装完成后运行: decision-cli setup")
	}

	if err := os.MkdirAll(filepath.Dir(expandedVenv), 0755); err != nil {
		return "", fmt.Errorf("failed to create parent directory for venv %s: %w", expandedVenv, err)
	}

	if _, err := os.Stat(venvPy); os.IsNotExist(err) {
		venvCmd := exec.Command(uvBin, "venv", expandedVenv)
		if out, err := venvCmd.CombinedOutput(); err != nil {
			return "", fmt.Errorf("failed to create self-contained venv at %s via %s: %v (%s)", expandedVenv, uvBin, err, strings.TrimSpace(string(out)))
		}
	}

	pipCmd := exec.Command(uvBin, "pip", "install", "--python", venvPy, "mlx>=0.32.0", "mlx-lm>=0.32.0", "pillow")
	if out, err := pipCmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("failed to install mlx, mlx-lm, pillow into %s: %v (%s)", expandedVenv, err, strings.TrimSpace(string(out)))
	}

	return venvPy, nil
}

// ResolveRuntime resolves the self-contained Python + MLX command according to config.yaml priority:
// 1. runtime.mlx_command
// 2. runtime.python_bin
// 3. runtime.venv_dir (default ~/.cache/decision-cli/venv, auto-provisioned via uv if missing)
// 4. System python3 fallback if uv provisioning is unavailable
func ResolveRuntime(cfg *types.AppConfig) ResolvedRuntime {
	// 1. Explicit mlx_command
	if len(cfg.Runtime.MLXCommand) > 0 {
		cmd := make([]string, len(cfg.Runtime.MLXCommand))
		for i, c := range cfg.Runtime.MLXCommand {
			cmd[i] = config.ExpandPath(c)
		}
		if mlxV, lmV, dev, ok := ProbeMLXCommand(cmd); ok {
			return ResolvedRuntime{
				Command:    cmd,
				Source:     "config:runtime.mlx_command",
				MLXVersion: mlxV,
				MLXLMVer:   lmV,
				DeviceInfo: dev,
				Valid:      true,
			}
		}
		return ResolvedRuntime{
			Command: cmd,
			Source:  "config:runtime.mlx_command",
			Valid:   false,
			Error:   fmt.Sprintf("configured runtime.mlx_command %v failed 'import mlx.core, mlx_lm, PIL'", cmd),
		}
	}

	// 2. Explicit python_bin
	if cfg.Runtime.PythonBin != "" {
		py := config.ExpandPath(cfg.Runtime.PythonBin)
		cmd := []string{py}
		if mlxV, lmV, dev, ok := ProbeMLXCommand(cmd); ok {
			return ResolvedRuntime{
				Command:    cmd,
				Source:     "config:runtime.python_bin",
				MLXVersion: mlxV,
				MLXLMVer:   lmV,
				DeviceInfo: dev,
				Valid:      true,
			}
		}
		return ResolvedRuntime{
			Command: cmd,
			Source:  "config:runtime.python_bin",
			Valid:   false,
			Error:   fmt.Sprintf("configured runtime.python_bin (%s) failed 'import mlx.core, mlx_lm, PIL'", py),
		}
	}

	// 3. Self-contained venv_dir (default ~/.cache/decision-cli/venv)
	venvDir := cfg.Runtime.VenvDir
	if strings.TrimSpace(venvDir) == "" {
		venvDir = "~/.cache/decision-cli/venv"
	}
	venvPy := filepath.Join(config.ExpandPath(venvDir), "bin", "python3")
	if _, err := os.Stat(venvPy); err == nil {
		cmd := []string{venvPy}
		if mlxV, lmV, dev, ok := ProbeMLXCommand(cmd); ok {
			return ResolvedRuntime{
				Command:    cmd,
				Source:     fmt.Sprintf("self-contained venv (%s)", venvDir),
				MLXVersion: mlxV,
				MLXLMVer:   lmV,
				DeviceInfo: dev,
				Valid:      true,
			}
		}
	}

	// Auto-provision the self-contained venv via uv if missing or incomplete
	if provisionedPy, err := EnsureManagedVenv(cfg); err == nil {
		cmd := []string{provisionedPy}
		if mlxV, lmV, dev, ok := ProbeMLXCommand(cmd); ok {
			return ResolvedRuntime{
				Command:    cmd,
				Source:     fmt.Sprintf("self-contained venv (auto-provisioned: %s)", venvDir),
				MLXVersion: mlxV,
				MLXLMVer:   lmV,
				DeviceInfo: dev,
				Valid:      true,
			}
		}
	}

	// 4. System python3 fallback (no external workspace venv reliance)
	candidates := []struct {
		bin    string
		source string
	}{
		{"/opt/homebrew/bin/python3", "system: /opt/homebrew/bin/python3"},
		{"python3", "system: PATH python3"},
	}
	for _, cand := range candidates {
		cmd := []string{cand.bin}
		if mlxV, lmV, dev, ok := ProbeMLXCommand(cmd); ok {
			return ResolvedRuntime{
				Command:    cmd,
				Source:     cand.source,
				MLXVersion: mlxV,
				MLXLMVer:   lmV,
				DeviceInfo: dev,
				Valid:      true,
			}
		}
	}

	return ResolvedRuntime{
		Valid: false,
		Error: "no self-contained Python environment with 'mlx', 'mlx_lm', and 'pillow' found at ~/.cache/decision-cli/venv; ensure 'uv' is installed or configure runtime.python_bin in ~/.config/decision-cli/config.yaml",
	}
}

// isValidModelDir checks if dir contains config.json and at least one .safetensors or weights file.
func isValidModelDir(dir string) bool {
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		return false
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		name := e.Name()
		if strings.HasSuffix(name, ".safetensors") || strings.HasSuffix(name, ".npz") || strings.HasSuffix(name, ".gguf") {
			return true
		}
	}
	return false
}

// ResolveModel resolves a model alias or HF repo ID to a local directory on disk.
// Priority:
//  1. Direct local directory path passed as modelName
//  2. models.paths[alias] in config.yaml
//  3. Scan models.scan_dirs in order (supports HF Hub models--org--repo/snapshots/<rev> and flat dirs)
func ResolveModel(cfg *types.AppConfig, modelName string) (ResolvedModel, error) {
	if modelName == "" {
		modelName = cfg.DefaultModel
	}

	// Lookup spec in registry (by alias or by hf_repo_id)
	spec, ok := cfg.Models.Registry[modelName]
	if !ok {
		for _, s := range cfg.Models.Registry {
			if strings.EqualFold(s.HFRepoID, modelName) || strings.EqualFold(s.Alias, modelName) {
				spec = s
				ok = true
				break
			}
		}
	}
	if !ok {
		// Allow ad-hoc local model directory or custom HF repo ID
		spec = types.ModelSpec{
			Alias:          modelName,
			HFRepoID:       modelName,
			PromptStyle:    "gemma4-binary",
			PositiveTokens: []string{"是", "Yes", "yes", "YES"},
			NegativeTokens: []string{"否", "No", "no", "NO"},
			Description:    "Custom model",
		}
	}

	// 1. Check if modelName itself is a valid local directory path
	expandedArg := config.ExpandPath(modelName)
	if isValidModelDir(expandedArg) {
		return ResolvedModel{
			Alias:        spec.Alias,
			Spec:         spec,
			LocalPath:    expandedArg,
			Source:       "cli:direct_path",
			CachedOnDisk: true,
		}, nil
	}

	// 2. Check explicit models.paths[alias] in config.yaml
	if cfgPath, exists := cfg.Models.Paths[spec.Alias]; exists && strings.TrimSpace(cfgPath) != "" {
		exp := config.ExpandPath(strings.TrimSpace(cfgPath))
		if isValidModelDir(exp) {
			return ResolvedModel{
				Alias:        spec.Alias,
				Spec:         spec,
				LocalPath:    exp,
				Source:       fmt.Sprintf("config:models.paths[%s]", spec.Alias),
				CachedOnDisk: true,
			}, nil
		}
		// Also check if exp is a HuggingFace models--org--repo root containing snapshots/
		if snap := findLatestSnapshot(exp); snap != "" {
			return ResolvedModel{
				Alias:        spec.Alias,
				Spec:         spec,
				LocalPath:    snap,
				Source:       fmt.Sprintf("config:models.paths[%s]", spec.Alias),
				CachedOnDisk: true,
			}, nil
		}
		return ResolvedModel{
			Alias:        spec.Alias,
			Spec:         spec,
			LocalPath:    exp,
			Source:       fmt.Sprintf("config:models.paths[%s] (INVALID/MISSING)", spec.Alias),
			CachedOnDisk: false,
		}, fmt.Errorf("configured model path %s for %s does not contain config.json and weights", exp, spec.Alias)
	}

	// 3. Scan models.scan_dirs in order
	hfFolder := "models--" + strings.ReplaceAll(spec.HFRepoID, "/", "--")
	repoParts := strings.Split(spec.HFRepoID, "/")
	repoBase := repoParts[len(repoParts)-1]

	for _, rawScanDir := range cfg.Models.ScanDirs {
		scanDir := config.ExpandPath(strings.TrimSpace(rawScanDir))
		if scanDir == "" {
			continue
		}
		// 3a. HF Hub cache layout: <scan_dir>/models--<org>--<repo>/snapshots/<hash>
		hfRepoRoot := filepath.Join(scanDir, hfFolder)
		if snap := findLatestSnapshot(hfRepoRoot); snap != "" {
			return ResolvedModel{
				Alias:        spec.Alias,
				Spec:         spec,
				LocalPath:    snap,
				Source:       fmt.Sprintf("scan_dir:%s (hf-hub)", rawScanDir),
				CachedOnDisk: true,
			}, nil
		}

		// 3b. Flat layout: <scan_dir>/<org>/<repo> or <scan_dir>/<repo> or <scan_dir>/<alias>
		flatCandidates := []string{
			filepath.Join(scanDir, spec.HFRepoID),
			filepath.Join(scanDir, repoBase),
			filepath.Join(scanDir, strings.TrimSuffix(repoBase, "-MLX")),
			filepath.Join(scanDir, spec.Alias),
		}
		for _, fc := range flatCandidates {
			if isValidModelDir(fc) {
				return ResolvedModel{
					Alias:        spec.Alias,
					Spec:         spec,
					LocalPath:    fc,
					Source:       fmt.Sprintf("scan_dir:%s (flat)", rawScanDir),
					CachedOnDisk: true,
				}, nil
			}
		}
	}

	return ResolvedModel{
		Alias:        spec.Alias,
		Spec:         spec,
		LocalPath:    "",
		Source:       "not_found",
		CachedOnDisk: false,
	}, fmt.Errorf("未在本地找到模型 %q (%s)。\n当前已扫描路径:\n  - models.paths: %v\n  - models.scan_dirs: %v\n建议操作:\n  1. 运行 `decision-cli models` 查看当前已就绪的模型列表;\n  2. 运行 `decision-cli models download %s` (或 `huggingface-cli download %s`) 下载权重;\n  3. 或在 ~/.config/decision-cli/config.yaml 中的 models.paths[%q] 显式指定本地物理路径。",
		spec.Alias, spec.HFRepoID, cfg.Models.Paths, cfg.Models.ScanDirs, spec.Alias, spec.HFRepoID, spec.Alias)
}

// DownloadModel 利用 huggingface-cli 下载指定模型权重到本地扫描目录，并返回解析后的本地物理路径
func DownloadModel(cfg *types.AppConfig, modelArg string) (string, string, error) {
	var repoID, alias string
	if spec, ok := cfg.Models.Registry[modelArg]; ok {
		alias = spec.Alias
		repoID = spec.HFRepoID
	} else {
		for _, s := range cfg.Models.Registry {
			if strings.EqualFold(s.HFRepoID, modelArg) || strings.EqualFold(s.Alias, modelArg) {
				alias = s.Alias
				repoID = s.HFRepoID
				break
			}
		}
	}
	if repoID == "" {
		alias = modelArg
		repoID = modelArg
	}

	// 检查是否已经就绪
	if rm, err := ResolveModel(cfg, modelArg); err == nil && rm.CachedOnDisk && rm.LocalPath != "" {
		return alias, rm.LocalPath, nil
	}

	// 查找 huggingface-cli
	hfCli := "huggingface-cli"
	found := false
	if _, err := exec.LookPath(hfCli); err == nil {
		found = true
	} else {
		for _, cand := range []string{
			"/opt/homebrew/bin/huggingface-cli",
			filepath.Join(config.ExpandPath("~"), ".local/bin/huggingface-cli"),
			filepath.Join(config.ExpandPath("~/.cache/decision-cli/venv/bin"), "huggingface-cli"),
		} {
			if _, err := os.Stat(cand); err == nil {
				hfCli = cand
				found = true
				break
			}
		}
	}

	if !found {
		return alias, "", fmt.Errorf("未检测到 huggingface-cli 工具。\n建议安装方案：\n  brew install huggingface-cli  # 或运行: uv tool install huggingface_hub[cli]\n安装完成后重新执行: decision-cli models download %s", alias)
	}

	fmt.Fprintf(os.Stderr, "📥 正在通过 %s 下载模型权重: %s (别名: %s)...\n", hfCli, repoID, alias)
	cmd := exec.Command(hfCli, "download", repoID)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	if err := cmd.Run(); err != nil {
		return alias, "", fmt.Errorf("模型权重下载失败: %w", err)
	}

	// 重新解析模型路径
	rm, err := ResolveModel(cfg, modelArg)
	if err != nil || !rm.CachedOnDisk {
		return alias, "", fmt.Errorf("模型已下载，但未在 scan_dirs 中扫描到有效结构，请检查 ~/.cache/huggingface/hub 是否在 scan_dirs 配置中")
	}

	return alias, rm.LocalPath, nil
}

func findLatestSnapshot(hfRepoRoot string) string {
	snapDir := filepath.Join(hfRepoRoot, "snapshots")
	entries, err := os.ReadDir(snapDir)
	if err != nil || len(entries) == 0 {
		return ""
	}
	// Prefer refs/main if present
	if refBytes, err := os.ReadFile(filepath.Join(hfRepoRoot, "refs", "main")); err == nil {
		commit := strings.TrimSpace(string(refBytes))
		candidate := filepath.Join(snapDir, commit)
		if isValidModelDir(candidate) {
			return candidate
		}
	}
	// Otherwise sort snapshots by modification time descending
	type snapEntry struct {
		path    string
		modTime time.Time
	}
	var validSnaps []snapEntry
	for _, e := range entries {
		p := filepath.Join(snapDir, e.Name())
		if isValidModelDir(p) {
			info, _ := e.Info()
			var mt time.Time
			if info != nil {
				mt = info.ModTime()
			}
			validSnaps = append(validSnaps, snapEntry{path: p, modTime: mt})
		}
	}
	if len(validSnaps) == 0 {
		return ""
	}
	sort.Slice(validSnaps, func(i, j int) bool {
		return validSnaps[i].modTime.After(validSnaps[j].modTime)
	})
	return validSnaps[0].path
}

// EnsureWorkerScript materializes embedded mlx_worker.py to ~/.cache/decision-cli/mlx_worker.py.
func EnsureWorkerScript() (string, error) {
	scriptPath, _, err := ensureWorkerScriptWithChange()
	return scriptPath, err
}

func ensureWorkerScriptWithChange() (string, bool, error) {
	scriptPath := config.ExpandPath("~/.cache/decision-cli/mlx_worker.py")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0755); err != nil {
		return "", false, err
	}
	existing, err := os.ReadFile(scriptPath)
	changed := false
	if err != nil || !bytes.Equal(existing, embeddedWorkerPy) {
		if err := os.WriteFile(scriptPath, embeddedWorkerPy, 0755); err != nil {
			return "", false, err
		}
		changed = true
	}
	return scriptPath, changed, nil
}

// WorkerScoreItem is a single item sent to mlx_worker.py (supports per-item primitive/rubrics/choices/score_levels/images overrides).
type WorkerScoreItem struct {
	ID          string               `json:"id"`
	Context     string               `json:"context"`
	Images      []string             `json:"images,omitempty"`
	ImageTokens int                  `json:"image_tokens,omitempty"`
	Primitive   string               `json:"primitive,omitempty"`
	Rubrics     string               `json:"rubrics,omitempty"`
	Choices     []types.ChoiceOption `json:"choices,omitempty"`
	ScoreLevels []string             `json:"score_levels,omitempty"`
	DebiasOrder *bool                `json:"debias_order,omitempty"`
}

// WorkerScoreResult is the raw logit result returned by mlx_worker.py for one item.
type WorkerScoreResult struct {
	ID           string             `json:"id"`
	LogitYes     float64            `json:"logit_yes"`
	LogitNo      float64            `json:"logit_no"`
	Margin       float64            `json:"margin"`
	OptionLogits map[string]float64 `json:"option_logits,omitempty"`
	LatencyMs    float64            `json:"latency_ms"`
}

// WorkerMultiParams specifies primitive-specific parameters for mlx_worker.py.
type WorkerMultiParams struct {
	Primitive   string               `json:"primitive"` // "noul" | "choice" | "score"
	Rubrics     string               `json:"rubrics"`
	Choices     []types.ChoiceOption `json:"choices,omitempty"`
	ScoreLevels []string             `json:"score_levels,omitempty"`
	DebiasOrder bool                 `json:"debias_order,omitempty"`
	ImageTokens int                  `json:"image_tokens,omitempty"`
}

// ScoreResponse is the JSON envelope returned by mlx_worker.py.
type ScoreResponse struct {
	OK          bool                `json:"ok"`
	Primitive   string              `json:"primitive,omitempty"`
	Error       string              `json:"error,omitempty"`
	Traceback   string              `json:"traceback,omitempty"`
	ModelPath   string              `json:"model_path,omitempty"`
	PosTokenIDs []int               `json:"pos_token_ids,omitempty"`
	NegTokenIDs []int               `json:"neg_token_ids,omitempty"`
	Results     []WorkerScoreResult `json:"results,omitempty"`
}

// ParseIdleSeconds parses duration strings like "5m" or "300s" into seconds.
func ParseIdleSeconds(s string) float64 {
	if s == "" {
		return 300.0
	}
	if d, err := time.ParseDuration(s); err == nil {
		return d.Seconds()
	}
	return 300.0
}

// EnsureDaemonRunning starts the Unix Domain Socket MLX worker daemon if it isn't already listening,
// and returns an active socket connection Function.
func EnsureDaemonRunning(cfg *types.AppConfig, rt ResolvedRuntime) (string, error) {
	if !rt.Valid {
		return "", fmt.Errorf("invalid MLX runtime: %s", rt.Error)
	}
	sockPath := config.ExpandPath(cfg.Runtime.WorkerSocket)
	if sockPath == "" {
		sockPath = config.ExpandPath("~/.cache/decision-cli/worker.sock")
	}

	scriptPath, scriptChanged, err := ensureWorkerScriptWithChange()
	if err != nil {
		return "", fmt.Errorf("failed to write mlx_worker.py: %w", err)
	}

	// Check if daemon is already alive
	if pingDaemon(sockPath) {
		if !scriptChanged {
			return sockPath, nil
		}
		// Restart daemon so updated mlx_worker.py takes effect
		shutdownDaemon(sockPath)
		time.Sleep(100 * time.Millisecond)
	}

	if err := os.MkdirAll(filepath.Dir(sockPath), 0755); err != nil {
		return "", err
	}

	idleSec := fmt.Sprintf("%.0f", ParseIdleSeconds(cfg.Runtime.IdleTimeout))
	args := append(append([]string{}, rt.Command[1:]...), scriptPath, "--socket", sockPath, "--idle-timeout", idleSec)
	cmd := exec.Command(rt.Command[0], args...)
	logFile, _ := os.OpenFile(config.ExpandPath("~/.cache/decision-cli/worker.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if logFile != nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start mlx_worker daemon: %w", err)
	}
	exited := make(chan struct{})
	// Detach wait in background
	go func() {
		_ = cmd.Wait()
		close(exited)
		if logFile != nil {
			_ = logFile.Close()
		}
	}()

	// Wait up to 3 seconds for socket to become responsive, or break immediately if daemon exited
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if pingDaemon(sockPath) {
			return sockPath, nil
		}
		select {
		case <-exited:
			return "", fmt.Errorf("mlx_worker daemon exited before binding socket %s", sockPath)
		case <-time.After(60 * time.Millisecond):
		}
	}
	return "", fmt.Errorf("mlx_worker daemon did not respond on socket %s within timeout", sockPath)
}

func shutdownDaemon(sockPath string) {
	conn, err := net.DialTimeout("unix", sockPath, 300*time.Millisecond)
	if err != nil {
		return
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(500 * time.Millisecond))
	_, _ = conn.Write([]byte(`{"action":"shutdown"}` + "\n"))
}

func pingDaemon(sockPath string) bool {
	conn, err := net.DialTimeout("unix", sockPath, 500*time.Millisecond)
	if err != nil {
		return false
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(1 * time.Second))
	_, err = conn.Write([]byte(`{"action":"ping"}` + "\n"))
	if err != nil {
		return false
	}
	reader := bufio.NewReader(conn)
	line, err := reader.ReadBytes('\n')
	if err != nil {
		return false
	}
	var resp struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(line, &resp); err != nil {
		return false
	}
	return resp.OK
}

// ScoreBatchViaMLX evaluates a batch of binary noul items using the resident Unix Domain Socket worker
// (or falls back to one-shot subprocess if socket bind is restricted).
func ScoreBatchViaMLX(cfg *types.AppConfig, rt ResolvedRuntime, model ResolvedModel, rubrics string, items []WorkerScoreItem) (*ScoreResponse, error) {
	return ScoreMultiStreamingViaMLX(cfg, rt, model, WorkerMultiParams{
		Primitive: "noul",
		Rubrics:   rubrics,
	}, items, nil, nil)
}

// ScoreBatchStreamingViaMLX evaluates a batch of binary noul items in a single worker invocation, calling onItem
// immediately as each item is scored so callers can persist to SQLite and emit live progress.
func ScoreBatchStreamingViaMLX(
	cfg *types.AppConfig,
	rt ResolvedRuntime,
	model ResolvedModel,
	rubrics string,
	items []WorkerScoreItem,
	onStatus func(status string),
	onItem func(res WorkerScoreResult),
) (*ScoreResponse, error) {
	return ScoreMultiStreamingViaMLX(cfg, rt, model, WorkerMultiParams{
		Primitive: "noul",
		Rubrics:   rubrics,
	}, items, onStatus, onItem)
}

// ScoreMultiViaMLX evaluates a batch of items for any primitive ("noul", "choice", "score").
func ScoreMultiViaMLX(
	cfg *types.AppConfig,
	rt ResolvedRuntime,
	model ResolvedModel,
	params WorkerMultiParams,
	items []WorkerScoreItem,
) (*ScoreResponse, error) {
	return ScoreMultiStreamingViaMLX(cfg, rt, model, params, items, nil, nil)
}

// ScoreMultiStreamingViaMLX evaluates a batch of items for any primitive ("noul", "choice", "score")
// in a single worker invocation, calling onItem immediately as each item is scored.
func ScoreMultiStreamingViaMLX(
	cfg *types.AppConfig,
	rt ResolvedRuntime,
	model ResolvedModel,
	params WorkerMultiParams,
	items []WorkerScoreItem,
	onStatus func(status string),
	onItem func(res WorkerScoreResult),
) (*ScoreResponse, error) {
	prim := params.Primitive
	if prim == "" {
		prim = "noul"
	}

	buildReq := func(stream bool) map[string]any {
		m := map[string]any{
			"action":          "score",
			"primitive":       prim,
			"model_path":      model.LocalPath,
			"prompt_style":    model.Spec.PromptStyle,
			"rubrics":         params.Rubrics,
			"positive_tokens": model.Spec.PositiveTokens,
			"negative_tokens": model.Spec.NegativeTokens,
			"debias_order":    params.DebiasOrder,
			"items":           items,
		}
		if stream {
			m["stream"] = true
		}
		if len(params.Choices) > 0 {
			m["choices"] = params.Choices
		}
		if len(params.ScoreLevels) > 0 {
			m["score_levels"] = params.ScoreLevels
		}
		if params.ImageTokens > 0 {
			m["image_tokens"] = params.ImageTokens
		}
		return m
	}

	// If no streaming callback needed, try Unix socket first
	if onItem == nil {
		reqPayload := buildReq(false)
		if reqBytes, err := json.Marshal(reqPayload); err == nil {
			if sockPath, err := EnsureDaemonRunning(cfg, rt); err == nil {
				if resp, sErr := sendSocketRequest(sockPath, reqBytes); sErr == nil {
					return resp, nil
				}
			}
		}
	}

	reqPayload := buildReq(true)
	reqBytes, err := json.Marshal(reqPayload)
	if err != nil {
		return nil, err
	}

	scriptPath, err := EnsureWorkerScript()
	if err != nil {
		return nil, err
	}
	args := append(append([]string{}, rt.Command[1:]...), scriptPath, "--oneshot")
	cmd := exec.Command(rt.Command[0], args...)
	cmd.Stdin = bytes.NewReader(reqBytes)

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start mlx_worker: %w", err)
	}

	var finalResp ScoreResponse
	var collected []WorkerScoreResult
	scanner := bufio.NewScanner(stdoutPipe)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 16*1024*1024)

	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var envelope struct {
			Type    string            `json:"type"`
			Status  string            `json:"status"`
			OK      *bool             `json:"ok"`
			Error   string            `json:"error"`
			Result  WorkerScoreResult `json:"result"`
			Results []WorkerScoreResult `json:"results"`
		}
		if err := json.Unmarshal(line, &envelope); err != nil {
			continue
		}
		switch envelope.Type {
		case "status":
			if onStatus != nil {
				onStatus(envelope.Status)
			}
		case "item":
			collected = append(collected, envelope.Result)
			if onItem != nil {
				onItem(envelope.Result)
			}
		case "done":
			finalResp.OK = true
			finalResp.Results = envelope.Results
		default:
			if envelope.OK != nil && !*envelope.OK {
				_ = cmd.Wait()
				return nil, fmt.Errorf("mlx_worker error: %s", envelope.Error)
			}
			if envelope.OK != nil && *envelope.OK && len(envelope.Results) > 0 {
				finalResp.OK = true
				finalResp.Results = envelope.Results
			}
		}
	}

	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("mlx_worker failed: %v (%s)", err, strings.TrimSpace(stderr.String()))
	}
	if !finalResp.OK {
		if len(collected) > 0 {
			finalResp.OK = true
			finalResp.Results = collected
		} else {
			return nil, fmt.Errorf("mlx_worker exited without valid done message (%s)", strings.TrimSpace(stderr.String()))
		}
	}
	return &finalResp, nil
}

func sendSocketRequest(sockPath string, reqBytes []byte) (*ScoreResponse, error) {
	conn, err := net.DialTimeout("unix", sockPath, 2*time.Second)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(10 * time.Minute))
	if _, err := conn.Write(append(reqBytes, '\n')); err != nil {
		return nil, err
	}
	reader := bufio.NewReaderSize(conn, 1024*1024)
	line, err := reader.ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	var resp ScoreResponse
	if err := json.Unmarshal(bytes.TrimSpace(line), &resp); err != nil {
		return nil, err
	}
	if !resp.OK {
		return nil, fmt.Errorf("mlx_worker daemon error: %s", resp.Error)
	}
	return &resp, nil
}
