package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cafe3310/decision-cli/internal/cache"
	"github.com/cafe3310/decision-cli/internal/config"
	"github.com/cafe3310/decision-cli/internal/engine"
	"github.com/cafe3310/decision-cli/internal/exporter"
	"github.com/cafe3310/decision-cli/internal/infra"
	"github.com/cafe3310/decision-cli/internal/types"
)

// ServerOptions 服务配置参数
type ServerOptions struct {
	Host         string
	Port         int
	DefaultModel string
	DebiasOrder  bool
	NoCache      bool
	ConfigPath   string
}

// RunServer 启动常驻 SystemOne HTTP API 服务
func RunServer(opts ServerOptions) error {
	cfg, cfgPath, _, err := config.LoadOrInit(opts.ConfigPath)
	if err != nil {
		return fmt.Errorf("加载配置失败: %w", err)
	}

	modelName := opts.DefaultModel
	if modelName == "" {
		if cfg.DefaultModel != "" {
			modelName = cfg.DefaultModel
		} else {
			modelName = "gemma-4-26b-a4b"
		}
	}

	store, err := cache.Open(cfg.Cache.DBPath)
	if err != nil {
		return fmt.Errorf("打开 SQLite 缓存失败: %w", err)
	}
	defer store.Close()

	mux := http.NewServeMux()

	// 1. POST /v1/systemone - 复合多问判定标准接口
	mux.HandleFunc("/v1/systemone", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		if r.Method != http.MethodPost {
			writeJSONError(w, r, t0, "", http.StatusMethodNotAllowed, "只支持 POST 方法")
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, fmt.Sprintf("读取请求体失败: %v", err))
			return
		}
		defer r.Body.Close()

		req, err := exporter.LoadSystemOneRequest(string(body))
		if err != nil {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, fmt.Sprintf("解析 SystemOne 请求失败: %v", err))
			return
		}

		// 若未在请求中显式指定模型，使用服务端默认模型
		reqModel := req.Model
		if reqModel == "" {
			reqModel = modelName
		}

		baseOpts := engine.Options{
			ModelName:   reqModel,
			DebiasOrder: opts.DebiasOrder,
			Threshold:   0.5,
			NoCache:     opts.NoCache,
		}

		resp, err := engine.EvaluateSystemOne(cfg, store, baseOpts, req, 0, false)
		if err != nil {
			writeJSONError(w, r, t0, reqModel, http.StatusInternalServerError, fmt.Sprintf("推理失败: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, resp)
		logServerAccess(r, http.StatusOK, time.Since(t0), reqModel, fmt.Sprintf("SystemOne 复合问卷 (%d 项判定)", len(resp.Answers)))
	})

	// 2. POST /v1/noun - 单条二元判定快捷接口
	mux.HandleFunc("/v1/noun", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		if r.Method != http.MethodPost {
			writeJSONError(w, r, t0, "", http.StatusMethodNotAllowed, "只支持 POST 方法")
			return
		}

		var payload struct {
			Rubrics     string   `json:"rubrics"`
			Context     string   `json:"context"`
			Image       string   `json:"image"`
			Images      []string `json:"images"`
			ImageTokens int      `json:"image_tokens"`
			Model       string   `json:"model"`
			Threshold   *float64 `json:"threshold"`
			Temperature *float64 `json:"temperature"`
			NoCache     *bool    `json:"no_cache"`
		}

		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, fmt.Sprintf("JSON 请求体解析失败: %v", err))
			return
		}

		var imgs []string
		if strings.TrimSpace(payload.Image) != "" {
			imgs = append(imgs, strings.TrimSpace(payload.Image))
		}
		imgs = append(imgs, payload.Images...)

		if strings.TrimSpace(payload.Rubrics) == "" || (strings.TrimSpace(payload.Context) == "" && len(imgs) == 0) {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, "rubrics 为必填字段，且 context 与 image/images 至少需提供一项")
			return
		}

		targetModel := modelName
		if payload.Model != "" {
			targetModel = payload.Model
		}

		rm, err := infra.ResolveModel(cfg, targetModel)
		if err != nil {
			writeJSONError(w, r, t0, targetModel, http.StatusBadRequest, fmt.Sprintf("解析模型失败: %v", err))
			return
		}

		th := 0.5
		thChanged := false
		if payload.Threshold != nil {
			th = *payload.Threshold
			thChanged = true
		}
		effThresh := config.EffectiveThreshold(cfg, rm.Spec, th, thChanged)

		temp := 1.0
		tempChanged := false
		if payload.Temperature != nil {
			temp = *payload.Temperature
			tempChanged = true
		}
		effTemp := config.EffectivePrimitiveTemperature(cfg, rm.Spec, "noul", temp, tempChanged)

		noCache := opts.NoCache
		if payload.NoCache != nil {
			noCache = *payload.NoCache
		}

		singleOpts := engine.Options{
			ModelName:   targetModel,
			Primitive:   "noul",
			Rubrics:     payload.Rubrics,
			Images:      imgs,
			ImageTokens: payload.ImageTokens,
			Threshold:   effThresh,
			Temperature: effTemp,
			NoCache:     noCache,
		}

		res, err := engine.EvaluateSingle(cfg, store, singleOpts, payload.Context)
		if err != nil {
			writeJSONError(w, r, t0, targetModel, http.StatusInternalServerError, fmt.Sprintf("推理失败: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, res)
		logServerAccess(r, http.StatusOK, time.Since(t0), targetModel, fmt.Sprintf("决策: %s (p=%.3f, cache: %s)", res.Decision, res.Probability, cacheTag(res.Cached)))
	})

	// 3. POST /v1/choose - 单条多分类选择快捷接口
	mux.HandleFunc("/v1/choose", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		if r.Method != http.MethodPost {
			writeJSONError(w, r, t0, "", http.StatusMethodNotAllowed, "只支持 POST 方法")
			return
		}

		var payload struct {
			Rubrics     string          `json:"rubrics"`
			Context     string          `json:"context"`
			Image       string          `json:"image"`
			Images      []string        `json:"images"`
			ImageTokens int             `json:"image_tokens"`
			Choices     json.RawMessage `json:"choices"` // 允许字符串 "a:desc,b:desc" 或对象数组 [{"key":"a","description":"desc"}]
			DebiasOrder *bool           `json:"debias_order"`
			Model       string          `json:"model"`
			Temperature *float64        `json:"temperature"`
			NoCache     *bool           `json:"no_cache"`
		}

		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, fmt.Sprintf("JSON 请求体解析失败: %v", err))
			return
		}

		var imgs []string
		if strings.TrimSpace(payload.Image) != "" {
			imgs = append(imgs, strings.TrimSpace(payload.Image))
		}
		imgs = append(imgs, payload.Images...)

		if strings.TrimSpace(payload.Rubrics) == "" || (strings.TrimSpace(payload.Context) == "" && len(imgs) == 0) {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, "rubrics 为必填字段，且 context 与 image/images 至少需提供一项")
			return
		}

		// 解析 choices
		var parsedChoices []types.ChoiceOption
		var choicesStr string
		if err := json.Unmarshal(payload.Choices, &choicesStr); err == nil {
			parsedChoices, err = engine.ParseChoicesFlag(choicesStr)
			if err != nil {
				writeJSONError(w, r, t0, "", http.StatusBadRequest, fmt.Sprintf("解析 choices 字符串失败: %v", err))
				return
			}
		} else if err := json.Unmarshal(payload.Choices, &parsedChoices); err != nil {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, "choices 必须为字符串或 [{\"key\":\"...\",\"description\":\"...\"}] 数组")
			return
		}

		targetModel := modelName
		if payload.Model != "" {
			targetModel = payload.Model
		}

		rm, err := infra.ResolveModel(cfg, targetModel)
		if err != nil {
			writeJSONError(w, r, t0, targetModel, http.StatusBadRequest, fmt.Sprintf("解析模型失败: %v", err))
			return
		}

		temp := 1.0
		tempChanged := false
		if payload.Temperature != nil {
			temp = *payload.Temperature
			tempChanged = true
		}
		effTemp := config.EffectivePrimitiveTemperature(cfg, rm.Spec, "choice", temp, tempChanged)

		debias := opts.DebiasOrder
		if payload.DebiasOrder != nil {
			debias = *payload.DebiasOrder
		}

		noCache := opts.NoCache
		if payload.NoCache != nil {
			noCache = *payload.NoCache
		}

		chooseOpts := engine.Options{
			ModelName:   targetModel,
			Primitive:   "choice",
			Rubrics:     payload.Rubrics,
			Choices:     parsedChoices,
			Images:      imgs,
			ImageTokens: payload.ImageTokens,
			DebiasOrder: debias,
			Temperature: effTemp,
			NoCache:     noCache,
		}

		res, err := engine.EvaluateSingleChoice(cfg, store, chooseOpts, payload.Context)
		if err != nil {
			writeJSONError(w, r, t0, targetModel, http.StatusInternalServerError, fmt.Sprintf("推理失败: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, res)
		logServerAccess(r, http.StatusOK, time.Since(t0), targetModel, fmt.Sprintf("选择: %s (置信度=%.3f, cache: %s)", res.Choice, res.Confidence, cacheTag(res.Cached)))
	})

	// 4. POST /v1/score - 单条分级打分快捷接口
	mux.HandleFunc("/v1/score", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		if r.Method != http.MethodPost {
			writeJSONError(w, r, t0, "", http.StatusMethodNotAllowed, "只支持 POST 方法")
			return
		}

		var payload struct {
			Rubrics     string          `json:"rubrics"`
			Context     string          `json:"context"`
			Image       string          `json:"image"`
			Images      []string        `json:"images"`
			ImageTokens int             `json:"image_tokens"`
			Levels      json.RawMessage `json:"levels"` // 允许字符串 "低,中,高" 或数组 ["低","中","高"]
			DebiasOrder *bool           `json:"debias_order"`
			Model       string          `json:"model"`
			Temperature *float64        `json:"temperature"`
			NoCache     *bool           `json:"no_cache"`
		}

		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, fmt.Sprintf("JSON 请求体解析失败: %v", err))
			return
		}

		var imgs []string
		if strings.TrimSpace(payload.Image) != "" {
			imgs = append(imgs, strings.TrimSpace(payload.Image))
		}
		imgs = append(imgs, payload.Images...)

		if strings.TrimSpace(payload.Rubrics) == "" || (strings.TrimSpace(payload.Context) == "" && len(imgs) == 0) {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, "rubrics 为必填字段，且 context 与 image/images 至少需提供一项")
			return
		}

		var parsedLevels []string
		var levelsStr string
		if err := json.Unmarshal(payload.Levels, &levelsStr); err == nil {
			parsedLevels, err = engine.ParseScoreFlag(levelsStr)
			if err != nil {
				writeJSONError(w, r, t0, "", http.StatusBadRequest, fmt.Sprintf("解析 levels 字符串失败: %v", err))
				return
			}
		} else if err := json.Unmarshal(payload.Levels, &parsedLevels); err != nil {
			writeJSONError(w, r, t0, "", http.StatusBadRequest, "levels 必须为逗号分隔字符串或字符串数组")
			return
		}

		targetModel := modelName
		if payload.Model != "" {
			targetModel = payload.Model
		}

		rm, err := infra.ResolveModel(cfg, targetModel)
		if err != nil {
			writeJSONError(w, r, t0, targetModel, http.StatusBadRequest, fmt.Sprintf("解析模型失败: %v", err))
			return
		}

		temp := 1.0
		tempChanged := false
		if payload.Temperature != nil {
			temp = *payload.Temperature
			tempChanged = true
		}
		effTemp := config.EffectivePrimitiveTemperature(cfg, rm.Spec, "score", temp, tempChanged)

		debias := opts.DebiasOrder
		if payload.DebiasOrder != nil {
			debias = *payload.DebiasOrder
		}

		noCache := opts.NoCache
		if payload.NoCache != nil {
			noCache = *payload.NoCache
		}

		scoreOpts := engine.Options{
			ModelName:   targetModel,
			Primitive:   "score",
			Rubrics:     payload.Rubrics,
			ScoreLevels: parsedLevels,
			Images:      imgs,
			ImageTokens: payload.ImageTokens,
			DebiasOrder: debias,
			Temperature: effTemp,
			NoCache:     noCache,
		}

		res, err := engine.EvaluateSingleScore(cfg, store, scoreOpts, payload.Context)
		if err != nil {
			writeJSONError(w, r, t0, targetModel, http.StatusInternalServerError, fmt.Sprintf("推理失败: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, res)
		logServerAccess(r, http.StatusOK, time.Since(t0), targetModel, fmt.Sprintf("打分: %s (期望均值=%.2f, cache: %s)", res.TopLabel, res.Score, cacheTag(res.Cached)))
	})

	// 5. GET /health - 健康检查
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		writeJSON(w, http.StatusOK, map[string]any{
			"status":        "ok",
			"default_model": modelName,
			"config_path":   cfgPath,
			"time":          time.Now().Format(time.RFC3339),
		})
		logServerAccess(r, http.StatusOK, time.Since(t0), modelName, "ok")
	})

	addr := net.JoinHostPort(opts.Host, strconv.Itoa(opts.Port))
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		if strings.Contains(err.Error(), "address already in use") {
			return fmt.Errorf("HTTP 服务启动失败: 端口 %d 已被占用。\n建议操作:\n  1. 使用 --port <新端口> 指定可用端口 (例如: decision-cli serve --port %d);\n  2. 或运行 `lsof -i :%d` 检查并终止占用进程。", opts.Port, opts.Port+1, opts.Port)
		}
		return fmt.Errorf("HTTP 服务监听失败 (%s): %w", addr, err)
	}

	server := &http.Server{
		Handler: corsMiddleware(mux),
	}

	fmt.Fprintf(os.Stderr, "🚀 decision-cli SystemOne API 服务已启动\n")
	fmt.Fprintf(os.Stderr, "   - 监听地址 : http://%s\n", addr)
	fmt.Fprintf(os.Stderr, "   - 默认模型 : %s\n", modelName)
	fmt.Fprintf(os.Stderr, "   - 服务端点 : POST http://%s/v1/systemone\n", addr)
	fmt.Fprintf(os.Stderr, "   - 快捷端点 : POST /v1/noun, /v1/choose, /v1/score\n")
	fmt.Fprintf(os.Stderr, "   - 健康检查 : GET http://%s/health\n", addr)
	fmt.Fprintf(os.Stderr, "   - 退出服务 : 按 Ctrl+C 优雅关闭\n\n")

	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "HTTP 服务异常退出: %v\n", err)
		}
	}()

	<-stopChan
	fmt.Fprintf(os.Stderr, "\n正在关闭 HTTP 服务...\n")

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	return server.Shutdown(ctx)
}

// corsMiddleware 为本地 API 请求添加标准 CORS 响应头并处理 OPTIONS 预检请求
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept")
		w.Header().Set("Access-Control-Max-Age", "86400")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			logServerAccess(r, http.StatusNoContent, 0, "-", "CORS preflight")
			return
		}

		next.ServeHTTP(w, r)
	})
}

func cacheTag(cached bool) string {
	if cached {
		return "HIT"
	}
	return "MISS"
}

func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err == nil && host != "" {
		return host
	}
	return r.RemoteAddr
}

func logServerAccess(r *http.Request, statusCode int, duration time.Duration, model string, summary string) {
	ts := time.Now().Format("15:04:05")
	ip := clientIP(r)
	modelTag := model
	if modelTag == "" {
		modelTag = "-"
	}
	durMs := float64(duration.Microseconds()) / 1000.0
	fmt.Fprintf(os.Stderr, "[%s] %-4s %-16s %d %6.1fms [%s] %s (%s)\n",
		ts, r.Method, r.URL.Path, statusCode, durMs, modelTag, summary, ip)
}

func writeJSON(w http.ResponseWriter, statusCode int, data any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(data)
}

func writeJSONError(w http.ResponseWriter, r *http.Request, t0 time.Time, model string, statusCode int, message string) {
	logServerAccess(r, statusCode, time.Since(t0), model, "错误: "+message)
	writeJSON(w, statusCode, map[string]any{
		"error": message,
	})
}
