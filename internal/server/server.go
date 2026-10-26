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
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "只支持 POST 方法")
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("读取请求体失败: %v", err))
			return
		}
		defer r.Body.Close()

		req, err := exporter.LoadSystemOneRequest(string(body))
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("解析 SystemOne 请求失败: %v", err))
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
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("推理失败: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, resp)
	})

	// 2. POST /v1/noun - 单条二元判定快捷接口
	mux.HandleFunc("/v1/noun", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "只支持 POST 方法")
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
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("JSON 请求体解析失败: %v", err))
			return
		}

		var imgs []string
		if strings.TrimSpace(payload.Image) != "" {
			imgs = append(imgs, strings.TrimSpace(payload.Image))
		}
		imgs = append(imgs, payload.Images...)

		if strings.TrimSpace(payload.Rubrics) == "" || (strings.TrimSpace(payload.Context) == "" && len(imgs) == 0) {
			writeJSONError(w, http.StatusBadRequest, "rubrics 为必填字段，且 context 与 image/images 至少需提供一项")
			return
		}

		targetModel := modelName
		if payload.Model != "" {
			targetModel = payload.Model
		}

		rm, _ := infra.ResolveModel(cfg, targetModel)
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
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("推理失败: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, res)
	})

	// 3. POST /v1/choose - 单条多分类选择快捷接口
	mux.HandleFunc("/v1/choose", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "只支持 POST 方法")
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
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("JSON 请求体解析失败: %v", err))
			return
		}

		var imgs []string
		if strings.TrimSpace(payload.Image) != "" {
			imgs = append(imgs, strings.TrimSpace(payload.Image))
		}
		imgs = append(imgs, payload.Images...)

		if strings.TrimSpace(payload.Rubrics) == "" || (strings.TrimSpace(payload.Context) == "" && len(imgs) == 0) {
			writeJSONError(w, http.StatusBadRequest, "rubrics 为必填字段，且 context 与 image/images 至少需提供一项")
			return
		}

		// 解析 choices
		var parsedChoices []types.ChoiceOption
		var choicesStr string
		if err := json.Unmarshal(payload.Choices, &choicesStr); err == nil {
			parsedChoices, err = engine.ParseChoicesFlag(choicesStr)
			if err != nil {
				writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("解析 choices 字符串失败: %v", err))
				return
			}
		} else if err := json.Unmarshal(payload.Choices, &parsedChoices); err != nil {
			writeJSONError(w, http.StatusBadRequest, "choices 必须为字符串或 [{\"key\":\"...\",\"description\":\"...\"}] 数组")
			return
		}

		targetModel := modelName
		if payload.Model != "" {
			targetModel = payload.Model
		}

		rm, _ := infra.ResolveModel(cfg, targetModel)
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
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("推理失败: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, res)
	})

	// 4. POST /v1/score - 单条分级打分快捷接口
	mux.HandleFunc("/v1/score", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSONError(w, http.StatusMethodNotAllowed, "只支持 POST 方法")
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
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("JSON 请求体解析失败: %v", err))
			return
		}

		var imgs []string
		if strings.TrimSpace(payload.Image) != "" {
			imgs = append(imgs, strings.TrimSpace(payload.Image))
		}
		imgs = append(imgs, payload.Images...)

		if strings.TrimSpace(payload.Rubrics) == "" || (strings.TrimSpace(payload.Context) == "" && len(imgs) == 0) {
			writeJSONError(w, http.StatusBadRequest, "rubrics 为必填字段，且 context 与 image/images 至少需提供一项")
			return
		}

		var parsedLevels []string
		var levelsStr string
		if err := json.Unmarshal(payload.Levels, &levelsStr); err == nil {
			parsedLevels, err = engine.ParseScoreFlag(levelsStr)
			if err != nil {
				writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("解析 levels 字符串失败: %v", err))
				return
			}
		} else if err := json.Unmarshal(payload.Levels, &parsedLevels); err != nil {
			writeJSONError(w, http.StatusBadRequest, "levels 必须为逗号分隔字符串或字符串数组")
			return
		}

		targetModel := modelName
		if payload.Model != "" {
			targetModel = payload.Model
		}

		rm, _ := infra.ResolveModel(cfg, targetModel)
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
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("推理失败: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, res)
	})

	// 5. GET /health - 健康检查
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"status":        "ok",
			"default_model": modelName,
			"config_path":   cfgPath,
			"time":          time.Now().Format(time.RFC3339),
		})
	})

	addr := net.JoinHostPort(opts.Host, strconv.Itoa(opts.Port))
	server := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	fmt.Fprintf(os.Stderr, "🚀 decision-cli SystemOne API 服务已启动\n")
	fmt.Fprintf(os.Stderr, "   - 监听地址 : http://%s\n", addr)
	fmt.Fprintf(os.Stderr, "   - 默认模型 : %s\n", modelName)
	fmt.Fprintf(os.Stderr, "   - 核心端点 : POST http://%s/v1/systemone\n", addr)
	fmt.Fprintf(os.Stderr, "   - 快捷端点 : POST /v1/noun, /v1/choose, /v1/score\n")
	fmt.Fprintf(os.Stderr, "   - 健康检查 : GET http://%s/health\n", addr)
	fmt.Fprintf(os.Stderr, "   - 退出服务 : 按 Ctrl+C 优雅关闭\n\n")

	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "HTTP 服务异常退出: %v\n", err)
		}
	}()

	<-stopChan
	fmt.Fprintf(os.Stderr, "\n正在关闭 HTTP 服务...\n")

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	return server.Shutdown(ctx)
}

func writeJSON(w http.ResponseWriter, statusCode int, data any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(data)
}

func writeJSONError(w http.ResponseWriter, statusCode int, message string) {
	writeJSON(w, statusCode, map[string]any{
		"error": message,
	})
}
