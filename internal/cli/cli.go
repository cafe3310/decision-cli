package cli

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/cafe3310/decision-cli/internal/cache"
	"github.com/cafe3310/decision-cli/internal/config"
	"github.com/cafe3310/decision-cli/internal/engine"
	"github.com/cafe3310/decision-cli/internal/exporter"
	"github.com/cafe3310/decision-cli/internal/infra"
	"github.com/cafe3310/decision-cli/internal/server"
	"github.com/cafe3310/decision-cli/internal/tuner"
	"github.com/cafe3310/decision-cli/internal/types"
	"github.com/spf13/cobra"
)

// RootFlags 保存根命令绑定的全局参数
type RootFlags struct {
	ConfigPath string
	ShowUsage  bool
}

// NewRootCommand 构建 decision-cli 的 Cobra 命令行树
func NewRootCommand() *cobra.Command {
	var flags RootFlags

	rootCmd := &cobra.Command{
		Use:   "decision-cli",
		Short: "macOS 本地 Apple Silicon (MLX) 非自回归概率决策模型 CLI 工具",
		Long: `decision-cli 是一款专为 macOS (Apple Silicon) 打造的本地高性能非自回归决策模型命令行工具。
基于 Apple MLX 推理框架，通过单步 Prefill 前向传播（max_tokens=1）提取原始 Logits 并计算校准概率，
配合 SQLite 内容寻址缓存实现毫秒级判定响应，原生支持文本与多模态图像（单图/多图/空间候选框选择）。

核心特性与子命令划分：
  1. 单条文本与多模态判别（支持 --context 文本与 --image 图像）：
     - noun   : 二元判定（输出“是”/“否”及置信概率）
     - choose : 互斥多分类判定（支持图像分类、多图对比与空间候选框 Box 择优）
     - score  : 分级量规连续打分（计算数学期望与离散概率分布）
  2. 批量数据处理（仅接受 --input 数据文件输入，支持 --image-field 图像字段）：
     - batch noun   : 批量二元判定
     - batch choose : 批量多分类标注
     - batch score  : 批量分级打分
  3. 复合问卷、API 服务与系统运维：
     - systemone : 评估 TypeSafe Jev /v1/systemone 复合多问请求（支持 assets 图像映射）
     - serve     : 启动常驻 SystemOne HTTP API 服务（监听指定 Host:Port）
     - kill      : 扫描并终止所有相关的 MLX 守护进程，释放统一内存
     - setup     : 初始化或修复自包含 MLX + Pillow 虚拟环境 (~/.cache/decision-cli/venv)
     - doctor    : 检查硬件环境、MLX 运行时并运行金丝雀推理自检
     - models    : 列出支持的模型及其本地路径解析状态
     - cache     : 查看、导出或清理 SQLite 内容寻址缓存

默认主力模型：gemma-4-26b-a4b（26B MoE + 27层视觉编码器，激活 3.8B 参数，强烈推荐文本与多模态主力使用）。
全局配置文件路径：~/.config/decision-cli/config.yaml（首次运行自动生成）。`,
		Example: `  # 查看面向 AI Agent 的结构化中文接入规范（含完整多模态与候选 Box 选择示例）：
  decision-cli --usage

  # 1. 单条文本二元判定：
  decision-cli noun --rubrics "是否包含讽刺语气" --context "你可真准时啊，会议都开完了你才来。"

  # 2. 多模态单图二元判定（传入 --image 时 --context 可省略）：
  decision-cli noun --rubrics "截图中是否出现了 HTTP 502 报错提示" --image ./error.png

  # 3. 多模态空间定位 / 九宫格候选 Box 选择（注意：--choices 内坐标使用全角逗号“，”避免被切分）：
  decision-cli choose \
    --image ./photo.jpg --image-tokens 560 --debias-order \
    --rubrics "将图像按 3x3 九宫格划分，判断哪个方块区域最帅" \
    --choices "grid_2:区域2（中上 [0，333，333，667]）,grid_5:区域5（正中 [333，333，667，667]）,grid_8:区域8（中下 [667，333，1000，667]）" \
    --context "图像坐标系为 [ymin，xmin，ymax，xmax]（0..1000 归一化）"

  # 4. 单条多分类选择（开启正反序去偏置）：
  decision-cli choose --rubrics "判断诉求类型" --choices "bug:故障,feat:新需求,ask:咨询" --context "导出按钮白屏了" --debias-order

  # 5. 单条分级打分：
  decision-cli score --rubrics "评估情绪激烈程度" --levels "平静,急躁,愤怒" --context "赶紧给我解决！"

  # 6. 批量多模态图片文件判定（通过 batch 子命令 + --input + --image-field）：
  decision-cli batch noun --rubrics "这个区域很帅" --input ./tiles.json --image-field image

  # 7. 环境健康检查：
  decision-cli doctor`,
		SilenceUsage:  true,
		SilenceErrors: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if flags.ShowUsage {
				fmt.Print(AgentUsageManual)
				return nil
			}
			return cmd.Help()
		},
	}

	pf := rootCmd.PersistentFlags()
	pf.StringVar(&flags.ConfigPath, "config", "", "全局配置文件路径（缺省为 ~/.config/decision-cli/config.yaml）")

	rootCmd.Flags().BoolVar(&flags.ShowUsage, "usage", false, "输出面向 AI Agent 与开发者的结构化中文使用手册")

	// 注册子命令
	rootCmd.AddCommand(newNounCommand(&flags))
	rootCmd.AddCommand(newChooseCommand(&flags))
	rootCmd.AddCommand(newScoreCommand(&flags))
	rootCmd.AddCommand(newBatchCommand(&flags))
	rootCmd.AddCommand(newSystemOneCommand(&flags))
	rootCmd.AddCommand(newServeCommand(&flags))
	rootCmd.AddCommand(newKillCommand(&flags))
	rootCmd.AddCommand(newSetupCommand(&flags))
	rootCmd.AddCommand(newDoctorCommand(&flags))
	rootCmd.AddCommand(newModelsCommand(&flags))
	rootCmd.AddCommand(newCacheCommand(&flags))
	rootCmd.AddCommand(newTuneCommand(&flags))

	return rootCmd
}

func parseImageFlagValues(raw []string) []string {
	var out []string
	for _, item := range raw {
		s := strings.TrimSpace(item)
		if s == "" {
			continue
		}
		if strings.HasPrefix(s, "data:image/") || strings.HasPrefix(s, "base64:") {
			out = append(out, s)
			continue
		}
		for _, part := range strings.Split(s, ",") {
			p := strings.TrimSpace(part)
			if p != "" {
				out = append(out, p)
			}
		}
	}
	return out
}

// =========================================================================
// 单条字符串与多模态图文判别子命令
// =========================================================================

func normalizeImageArgs(raw []string) []string {
	var out []string
	for _, item := range raw {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if strings.HasPrefix(item, "data:image/") {
			out = append(out, item)
			continue
		}
		for _, part := range strings.Split(item, ",") {
			if p := strings.TrimSpace(part); p != "" {
				out = append(out, p)
			}
		}
	}
	return out
}

func newNounCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		rubrics     string
		context     string
		images      []string
		imageTokens int
		model       string
		threshold   float64
		temperature float64
		calibA      float64
		calibB      float64
		format      string
		output      string
		noCache     bool
		cacheOnly   bool
	)

	cmd := &cobra.Command{
		Use:   "noun",
		Short: "单条二元判定（输入文本字符串与可选图片，输出“是”/“否”及置信概率）",
		Long: `noun 子命令对给定的单条文本或图文上下文执行二元准则评估。
直接提取目标 Token（“是”与“否”）的前向原始 Logits，计算温度校准后的正向概率。
支持通过 --image 传入单张或多张本地图像路径或 Base64，并可在文本中使用 {{image:1}} 引用。`,
		Example: `  # 判定是否属于系统报错（默认使用主力多模态模型 gemma-4-26b-a4b）：
  decision-cli noun \
    --rubrics "判断该反馈是否涉及核心支付功能的严重系统报错" \
    --context "微信支付点击后一直转圈，最后提示错误码 5002，钱扣了但订单未生效"

  # 多模态图文判定（检查报错截图是否包含 HTTP 502 错误）：
  decision-cli noun \
    --rubrics "判断截图中是否出现了 HTTP 502 Bad Gateway 报错提示" \
    --image ./error_screenshot.png \
    --context "用户反馈打开控制台白屏，附图如上"`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(rubrics) == "" {
				return fmt.Errorf("缺少 --rubrics 判定准则。\n建议操作: 请使用 --rubrics 提供自然语言判定准则。\n正确示例: decision-cli noun --rubrics \"判断是否包含严重系统报错\" --context \"微信支付报错 5002\"")
			}
			parsedImages := normalizeImageArgs(images)
			if strings.TrimSpace(context) == "" && len(parsedImages) == 0 {
				return fmt.Errorf("缺少待评估内容：--context 待评估文本或 --image 待评估图片至少需提供一项。\n正确示例 (纯文本): decision-cli noun --rubrics \"...\" --context \"待评估文本\"\n正确示例 (多模态): decision-cli noun --rubrics \"...\" --image ./screenshot.png")
			}

			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}

			modelName := model
			if !cmd.Flags().Changed("model") && cfg.DefaultModel != "" {
				modelName = cfg.DefaultModel
			}
			if modelName == "" {
				modelName = "gemma-4-26b-a4b"
			}
			rm, _ := infra.ResolveModel(cfg, modelName)
			effThresh := config.EffectiveThreshold(cfg, rm.Spec, threshold, cmd.Flags().Changed("threshold"))
			effTemp := config.EffectivePrimitiveTemperature(cfg, rm.Spec, "noul", temperature, cmd.Flags().Changed("temperature"))

			var ptrA, ptrB *float64
			if cmd.Flags().Changed("calib-a") {
				ptrA = &calibA
			} else if rm.Spec.CalibA != nil {
				ptrA = rm.Spec.CalibA
			}
			if cmd.Flags().Changed("calib-b") {
				ptrB = &calibB
			} else if rm.Spec.CalibB != nil {
				ptrB = rm.Spec.CalibB
			}

			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()

			opts := engine.Options{
				ModelName:   modelName,
				Primitive:   "noul",
				Rubrics:     rubrics,
				Images:      parsedImages,
				ImageTokens: imageTokens,
				Threshold:   effThresh,
				Temperature: effTemp,
				CalibA:      ptrA,
				CalibB:      ptrB,
				NoCache:     noCache,
				CacheOnly:   cacheOnly,
			}

			res, err := engine.EvaluateSingle(cfg, store, opts, context)
			if err != nil {
				return err
			}
			return exporter.FormatAndWrite(res, format, output)
		},
	}

	cmd.Flags().StringVar(&rubrics, "rubrics", "", "自然语言判定准则")
	cmd.Flags().StringVar(&context, "context", "", "待评估的单条文本字符串（可内嵌 {{image:1}} 占位符；传入 --image 时可省略）")
	cmd.Flags().StringArrayVar(&images, "image", nil, "待评估图像文件路径或 Base64（可多次指定以传入多张图片，或用逗号分隔路径）")
	cmd.Flags().IntVar(&imageTokens, "image-tokens", 280, "单张图片的最大视觉软 Token 预算（可选：70 | 140 | 280 | 560 | 1120）")
	cmd.Flags().StringVar(&model, "model", "gemma-4-26b-a4b", "模型别名或物理路径（多模态支持：gemma-4-26b-a4b、gemma-4-e4b）")
	cmd.Flags().Float64Var(&threshold, "threshold", 0.5, "判定为“是”的正向概率阈值（0.0 ~ 1.0）")
	cmd.Flags().Float64Var(&temperature, "temperature", 1.0, "Logit 概率校准温度（> 0.0）")
	cmd.Flags().Float64Var(&calibA, "calib-a", 0.0, "Platt 仿射校准缩放系数 A（未指定时默认使用 1/T）")
	cmd.Flags().Float64Var(&calibB, "calib-b", 0.0, "Platt 仿射校准平移常数 B（未指定时默认为 0.0）")
	cmd.Flags().StringVar(&format, "format", "json", "输出格式：json | yaml | compact | label-only")
	cmd.Flags().StringVar(&output, "output", "", "结果保存文件路径（缺省输出至 stdout）")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "跳过 SQLite 本地缓存读取与写入")
	cmd.Flags().BoolVar(&cacheOnly, "cache-only", false, "仅查本地缓存，不拉起模型推理")

	return cmd
}

func newChooseCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		rubrics     string
		context     string
		images      []string
		imageTokens int
		choices     string
		debiasOrder bool
		model       string
		temperature float64
		format      string
		output      string
		noCache     bool
		cacheOnly   bool
	)

	cmd := &cobra.Command{
		Use:   "choose",
		Short: "单条多分类判定（输入文本字符串与可选图片，在多个候选类别中计算 Softmax 分布并择优输出）",
		Long: `choose 子命令在给定的互斥候选类别集合中，提取各选项 Token 的单步 Logits 并归一化 Softmax 概率。
选项格式为英文逗号分隔的键值对，例如 "key1:描述一,key2:描述二"（注意：描述内部如需逗号或冒号，请使用全角“，”“：”）。
支持使用 --debias-order 运行正反序双向前向推理以消除大模型位置偏置。
支持通过 --image 传入单张或多张图片，实现图像分类、双图 A/B 对比或空间候选框 Box 定位（Gemma 4 原生坐标系为 0..1000 归一化的 [ymin，xmin，ymax，xmax]）。`,
		Example: `  # 1. 用户反馈核心诉求多分类判定：
  decision-cli choose \
    --rubrics "判断用户反馈的核心诉求类型" \
    --choices "bug:系统故障报错,feature:新功能建议,sales:商务咨询与报价,noise:闲聊无明确诉求" \
    --context "请问系统支持离线私有化部署到医疗内网吗？大概费用怎么核算？" \
    --debias-order

  # 2. 多模态空间定位 / 九宫格候选 Box 选择（描述内坐标用全角逗号“，”）：
  decision-cli choose \
    --image ./photo.jpg --image-tokens 560 --debias-order \
    --rubrics "将图像按 3行×3列 划分为 9 个区域，选出目标所在的候选框或最符合准则的区域" \
    --choices "grid_1:区域1（左上 [0，0，333，333]）,grid_2:区域2（中上 [0，333，333，667]）,grid_5:区域5（正中 [333，333，667，667]）" \
    --context "图像坐标系为 [ymin，xmin，ymax，xmax]（0..1000 归一化，左上角为原点）"

  # 3. 双图 A/B 视觉对比选择：
  decision-cli choose \
    --rubrics "对比两张 UI 截图的排版留白与信息层级可读性，选出更优版本" \
    --choices "plan_a:第一张图更清晰易读,plan_b:第二张图更清晰易读" \
    --image ./ui_v1.png --image ./ui_v2.png \
    --context "方案 A：{{image:1}}\n方案 B：{{image:2}}" \
    --debias-order`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(rubrics) == "" {
				return fmt.Errorf("缺少 --rubrics 分类指导准则。\n建议操作: 请使用 --rubrics 提供自然语言分类准则。\n正确示例: decision-cli choose --rubrics \"判断诉求类型\" --choices \"bug:故障,feat:新功能\" --context \"白屏报错\"")
			}
			parsedImages := normalizeImageArgs(images)
			if strings.TrimSpace(context) == "" && len(parsedImages) == 0 {
				return fmt.Errorf("缺少待评估内容：--context 待评估文本或 --image 待评估图片至少需提供一项。\n正确示例: decision-cli choose --rubrics \"...\" --choices \"...\" --context \"待评估文本\"")
			}
			if strings.TrimSpace(choices) == "" {
				return fmt.Errorf("缺少 --choices 候选类别集合。\n建议操作: 请使用逗号分隔各类别，冒号分隔键与描述。\n正确示例: --choices \"bug:系统故障,feat:新功能建议,ask:日常咨询\"")
			}

			parsedChoices, err := engine.ParseChoicesFlag(choices)
			if err != nil {
				return err
			}

			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}

			modelName := model
			if !cmd.Flags().Changed("model") && cfg.DefaultModel != "" {
				modelName = cfg.DefaultModel
			}
			if modelName == "" {
				modelName = "gemma-4-26b-a4b"
			}
			rm, _ := infra.ResolveModel(cfg, modelName)
			effTemp := config.EffectivePrimitiveTemperature(cfg, rm.Spec, "choice", temperature, cmd.Flags().Changed("temperature"))

			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()

			opts := engine.Options{
				ModelName:   modelName,
				Primitive:   "choice",
				Rubrics:     rubrics,
				Choices:     parsedChoices,
				Images:      parsedImages,
				ImageTokens: imageTokens,
				DebiasOrder: debiasOrder,
				Temperature: effTemp,
				NoCache:     noCache,
				CacheOnly:   cacheOnly,
			}

			res, err := engine.EvaluateSingleChoice(cfg, store, opts, context)
			if err != nil {
				return err
			}
			return exporter.FormatAndWrite(res, format, output)
		},
	}

	cmd.Flags().StringVar(&rubrics, "rubrics", "", "自然语言分类指导准则")
	cmd.Flags().StringVar(&context, "context", "", "待评估的单条文本字符串（可内嵌 {{image:1}} 占位符；传入 --image 时可省略）")
	cmd.Flags().StringArrayVar(&images, "image", nil, "待评估图像文件路径或 Base64（可多次指定以传入多张图片，或用逗号分隔路径）")
	cmd.Flags().IntVar(&imageTokens, "image-tokens", 280, "单张图片的最大视觉软 Token 预算（可选：70 | 140 | 280 | 560 | 1120）")
	cmd.Flags().StringVar(&choices, "choices", "", "多分类候选集合，逗号分隔选项，冒号分隔键与描述（例：\"bug:故障,feat:新功能\"）")
	cmd.Flags().BoolVar(&debiasOrder, "debias-order", false, "执行正反序双向推理并对冲位置偏置")
	cmd.Flags().StringVar(&model, "model", "gemma-4-26b-a4b", "模型别名或物理路径（多模态支持：gemma-4-26b-a4b、gemma-4-e4b）")
	cmd.Flags().Float64Var(&temperature, "temperature", 1.0, "Logit 概率校准温度（> 0.0）")
	cmd.Flags().StringVar(&format, "format", "json", "输出格式：json | yaml | compact")
	cmd.Flags().StringVar(&output, "output", "", "结果保存文件路径（缺省输出至 stdout）")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "跳过 SQLite 本地缓存读取与写入")
	cmd.Flags().BoolVar(&cacheOnly, "cache-only", false, "仅查本地缓存，不拉起模型推理")

	return cmd
}

func newScoreCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		rubrics     string
		context     string
		images      []string
		imageTokens int
		levels      string
		debiasOrder bool
		model       string
		temperature float64
		format      string
		output      string
		noCache     bool
		cacheOnly   bool
	)

	cmd := &cobra.Command{
		Use:   "score",
		Short: "单条分级量规连续打分（输入文本字符串与可选图片，计算阶梯等级分布与期望均值）",
		Long: `score 子命令依据从 0 到 M-1 递增的等级刻度描述，评估目标文本或图片所属等级。
输出离散概率分布以及加权数学期望（expected_score）。
支持使用 --debias-order 消除选项位置偏置，并支持通过 --image 传入单张或多张图片进行视觉打分。`,
		Example: `  # 评估用户客服对话中的情绪负面激烈程度：
  decision-cli score \
    --rubrics "评估用户客服对话中表达出的负面情绪严重程度" \
    --levels "完全平静礼貌,轻微催促质疑,语气急躁指责,严重愤怒使用攻击性语言" \
    --context "你们客服到底看没看我发的消息？我都等了半小时了，每次都是机器人自动回复敷衍！" \
    --debias-order

  # 评估 UI 页面截图的视觉整洁度：
  decision-cli score \
    --rubrics "评估页面截图的排版整洁度与视觉可用性" \
    --levels "0分_布局错乱无法阅读,1分_部分元素重叠拥挤,2分_结构基本清晰,3分_排版精致层次分明" \
    --image ./dashboard.png \
    --debias-order`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(rubrics) == "" {
				return fmt.Errorf("缺少 --rubrics 评分指导准则。\n建议操作: 请使用 --rubrics 提供自然语言评分准则。\n正确示例: decision-cli score --rubrics \"评估情绪激烈程度\" --levels \"平静,急躁,愤怒\" --context \"怎么回事？\"")
			}
			parsedImages := normalizeImageArgs(images)
			if strings.TrimSpace(context) == "" && len(parsedImages) == 0 {
				return fmt.Errorf("缺少待评估内容：--context 待评估文本或 --image 待评估图片至少需提供一项。\n正确示例: decision-cli score --rubrics \"...\" --levels \"...\" --context \"待评估文本\"")
			}
			if strings.TrimSpace(levels) == "" {
				return fmt.Errorf("缺少 --levels 等级刻度列表。\n建议操作: 请提供逗号分隔的升序等级刻度（支持 2 到 10 级）。\n正确示例: --levels \"完全平静,轻微焦躁,严重愤怒辱骂\"")
			}

			parsedLevels, err := engine.ParseScoreFlag(levels)
			if err != nil {
				return err
			}

			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}

			modelName := model
			if !cmd.Flags().Changed("model") && cfg.DefaultModel != "" {
				modelName = cfg.DefaultModel
			}
			if modelName == "" {
				modelName = "gemma-4-26b-a4b"
			}
			rm, _ := infra.ResolveModel(cfg, modelName)
			effTemp := config.EffectivePrimitiveTemperature(cfg, rm.Spec, "score", temperature, cmd.Flags().Changed("temperature"))

			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()

			opts := engine.Options{
				ModelName:   modelName,
				Primitive:   "score",
				Rubrics:     rubrics,
				ScoreLevels: parsedLevels,
				Images:      parsedImages,
				ImageTokens: imageTokens,
				DebiasOrder: debiasOrder,
				Temperature: effTemp,
				NoCache:     noCache,
				CacheOnly:   cacheOnly,
			}

			res, err := engine.EvaluateSingleScore(cfg, store, opts, context)
			if err != nil {
				return err
			}
			return exporter.FormatAndWrite(res, format, output)
		},
	}

	cmd.Flags().StringVar(&rubrics, "rubrics", "", "自然语言评分指导准则")
	cmd.Flags().StringVar(&context, "context", "", "待评估的单条文本字符串（可内嵌 {{image:1}} 占位符；传入 --image 时可省略）")
	cmd.Flags().StringArrayVar(&images, "image", nil, "待评估图像文件路径或 Base64（可多次指定以传入多张图片，或用逗号分隔路径）")
	cmd.Flags().IntVar(&imageTokens, "image-tokens", 280, "单张图片的最大视觉软 Token 预算（可选：70 | 140 | 280 | 560 | 1120）")
	cmd.Flags().StringVar(&levels, "levels", "", "升序等级刻度描述，逗号分隔（例：\"平静,轻微急躁,严重愤怒\"）")
	cmd.Flags().BoolVar(&debiasOrder, "debias-order", false, "执行正反序双向推理并对冲位置偏置")
	cmd.Flags().StringVar(&model, "model", "gemma-4-26b-a4b", "模型别名或物理路径（多模态支持：gemma-4-26b-a4b、gemma-4-e4b）")
	cmd.Flags().Float64Var(&temperature, "temperature", 1.0, "Logit 概率校准温度（> 0.0）")
	cmd.Flags().StringVar(&format, "format", "json", "输出格式：json | yaml | compact")
	cmd.Flags().StringVar(&output, "output", "", "结果保存文件路径（缺省输出至 stdout）")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "跳过 SQLite 本地缓存读取与写入")
	cmd.Flags().BoolVar(&cacheOnly, "cache-only", false, "仅查本地缓存，不拉起模型推理")

	return cmd
}

// =========================================================================
// 批量数据文件处理子命令组 (`decision-cli batch`)
// =========================================================================

func newBatchCommand(rootFlags *RootFlags) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "batch",
		Short: "批量数据文件判定与标注工具组（仅接受数据文件输入）",
		Long: `batch 命令组专为批量数据文件流水线设计。
仅接受文件路径输入（通过 --input），支持 .json、.jsonl、.yaml、.yml 格式。
复用常驻 MLX Worker 单次加载驻留内存与 SQLite 缓存，支持每秒数百条高吞吐判定。`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}

	cmd.AddCommand(newBatchNounCommand(rootFlags))
	cmd.AddCommand(newBatchChooseCommand(rootFlags))
	cmd.AddCommand(newBatchScoreCommand(rootFlags))

	return cmd
}

func newBatchNounCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		rubrics         string
		input           string
		output          string
		format          string
		model           string
		threshold       float64
		temperature     float64
		calibA          float64
		calibB          float64
		idField         string
		contextTemplate string
		imageField      string
		imageBaseDir    string
		imageTokens     int
		filterPassed    bool
		includeRecord   bool
		progress        bool
		noCache         bool
		cacheOnly       bool
	)

	cmd := &cobra.Command{
		Use:   "noun",
		Short: "批量文件二元判定（输入数据文件，支持文本与图片字段，输出结构化统计与判定列表）",
		Long: `batch noun 对数据文件中的每条记录执行二元准则评估。
自动识别记录中的 image / images 字段（或通过 --image-field 指定），支持图文混合批量评估。
可结合 --filter-passed 仅输出判定为“是”的条目。仅接受数据文件输入。`,
		Example: `  # 批量过滤涉密隐私记录并附带原始数据：
  decision-cli batch noun \
    --rubrics "判断该反馈是否涉及数据安全或隐私泄露风险" \
    --input feedbacks.jsonl \
    --output security-alerts.json \
    --filter-passed \
    --include-record \
    --progress`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(rubrics) == "" {
				return fmt.Errorf("缺少 --rubrics 判定准则。\n建议操作: 请使用 --rubrics 提供自然语言判定准则。\n正确示例: decision-cli batch noun --rubrics \"判断是否涉及隐私泄露\" --input ./data.json")
			}
			if strings.TrimSpace(input) == "" {
				return fmt.Errorf("缺少 --input 批量数据文件路径。\n建议操作: 请通过 --input 传入 .json, .jsonl 或 .yaml 数据文件。\n正确示例: decision-cli batch noun --rubrics \"...\" --input ./records.jsonl")
			}

			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}

			modelName := model
			if !cmd.Flags().Changed("model") && cfg.DefaultModel != "" {
				modelName = cfg.DefaultModel
			}
			if modelName == "" {
				modelName = "gemma-4-26b-a4b"
			}
			rm, _ := infra.ResolveModel(cfg, modelName)
			effThresh := config.EffectiveThreshold(cfg, rm.Spec, threshold, cmd.Flags().Changed("threshold"))
			effTemp := config.EffectivePrimitiveTemperature(cfg, rm.Spec, "noul", temperature, cmd.Flags().Changed("temperature"))

			var ptrA, ptrB *float64
			if cmd.Flags().Changed("calib-a") {
				ptrA = &calibA
			} else if rm.Spec.CalibA != nil {
				ptrA = rm.Spec.CalibA
			}
			if cmd.Flags().Changed("calib-b") {
				ptrB = &calibB
			} else if rm.Spec.CalibB != nil {
				ptrB = rm.Spec.CalibB
			}

			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()

			opts := engine.Options{
				ModelName:      modelName,
				Primitive:      "noul",
				Rubrics:        rubrics,
				ImageTokens:    imageTokens,
				Threshold:      effThresh,
				Temperature:    effTemp,
				CalibA:         ptrA,
				CalibB:         ptrB,
				NoCache:        noCache,
				CacheOnly:      cacheOnly,
				FilterPassed:   filterPassed,
				IncludeRecord:  includeRecord,
				StreamProgress: progress,
			}

			items, err := exporter.LoadBatchItemsWithImages(input, idField, contextTemplate, imageField, imageBaseDir)
			if err != nil {
				return err
			}
			report, err := engine.EvaluateBatch(cfg, store, opts, items)
			if err != nil {
				return err
			}
			return exporter.FormatAndWrite(report, format, output)
		},
	}

	cmd.Flags().StringVar(&rubrics, "rubrics", "", "自然语言判定准则")
	cmd.Flags().StringVar(&input, "input", "", "批量输入数据文件路径（.json, .jsonl, .yaml）")
	cmd.Flags().StringVar(&output, "output", "", "结果保存文件路径（缺省输出至 stdout）")
	cmd.Flags().StringVar(&format, "format", "json", "输出格式：json | yaml | jsonl")
	cmd.Flags().StringVar(&model, "model", "gemma-4-26b-a4b", "模型别名或物理路径")
	cmd.Flags().Float64Var(&threshold, "threshold", 0.5, "判定为“是”的正向概率阈值（0.0 ~ 1.0）")
	cmd.Flags().Float64Var(&temperature, "temperature", 1.0, "Logit 概率校准温度")
	cmd.Flags().Float64Var(&calibA, "calib-a", 0.0, "Platt 仿射校准缩放系数 A（未指定时默认使用 1/T）")
	cmd.Flags().Float64Var(&calibB, "calib-b", 0.0, "Platt 仿射校准平移常数 B（未指定时默认为 0.0）")
	cmd.Flags().StringVar(&idField, "id-field", "id", "输入数据中唯一主键字段名称")
	cmd.Flags().StringVar(&contextTemplate, "context-template", "", "拼接多字段为 context 的 Go 模板字符串")
	cmd.Flags().StringVar(&imageField, "image-field", "", "输入记录中图片路径字段名（缺省自动识别 images / image）")
	cmd.Flags().StringVar(&imageBaseDir, "image-base-dir", "", "解析记录中相对图片路径的基准目录")
	cmd.Flags().IntVar(&imageTokens, "image-tokens", 280, "单张图片分配的视觉 Soft Token 预算（70 | 140 | 280 | 560 | 1120）")
	cmd.Flags().BoolVar(&filterPassed, "filter-passed", false, "仅输出判定为“是”的条目")
	cmd.Flags().BoolVar(&includeRecord, "include-record", false, "在输出结果中附带原始输入对象")
	cmd.Flags().BoolVar(&progress, "progress", false, "在 stderr 实时流式打印 JSONL 处理进度")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "跳过 SQLite 本地缓存读取与写入")
	cmd.Flags().BoolVar(&cacheOnly, "cache-only", false, "仅查本地缓存，不拉起模型推理")

	return cmd
}

func newBatchChooseCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		rubrics         string
		choices         string
		debiasOrder     bool
		input           string
		output          string
		format          string
		model           string
		temperature     float64
		idField         string
		contextTemplate string
		imageField      string
		imageBaseDir    string
		imageTokens     int
		includeRecord   bool
		progress        bool
		noCache         bool
		cacheOnly       bool
	)

	cmd := &cobra.Command{
		Use:   "choose",
		Short: "批量文件多分类标注（输入数据文件，支持文本与图片字段，对每条记录输出分类归因）",
		Long: `batch choose 对输入数据文件执行批量多分类推理。
自动识别记录中的 image / images 字段（或通过 --image-field 指定），支持使用 --debias-order 消除选项位置偏置。`,
		Example: `  # 批量对用户工单进行类别归因：
  decision-cli batch choose \
    --rubrics "用户反馈问题归类" \
    --choices "ui:界面设计缺陷,backend:后端接口报错,network:本地网络超时,doc:文档不全" \
    --input tickets.json \
    --output classified-tickets.json \
    --debias-order`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(rubrics) == "" {
				return fmt.Errorf("缺少 --rubrics 分类指导准则。\n建议操作: 请使用 --rubrics 提供自然语言分类指导准则。\n正确示例: decision-cli batch choose --rubrics \"问题归因\" --choices \"ui:界面,backend:后端\" --input ./tickets.json")
			}
			if strings.TrimSpace(input) == "" {
				return fmt.Errorf("缺少 --input 批量数据文件路径。\n建议操作: 请通过 --input 传入 .json, .jsonl 或 .yaml 数据文件。\n正确示例: decision-cli batch choose --rubrics \"...\" --choices \"...\" --input ./tickets.json")
			}
			if strings.TrimSpace(choices) == "" {
				return fmt.Errorf("缺少 --choices 候选类别集合。\n建议操作: 请使用逗号分隔各类别，冒号分隔键与描述。\n正确示例: --choices \"bug:系统故障,feat:新功能建议,ask:日常咨询\"")
			}

			parsedChoices, err := engine.ParseChoicesFlag(choices)
			if err != nil {
				return err
			}

			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}

			modelName := model
			if !cmd.Flags().Changed("model") && cfg.DefaultModel != "" {
				modelName = cfg.DefaultModel
			}
			if modelName == "" {
				modelName = "gemma-4-26b-a4b"
			}
			rm, _ := infra.ResolveModel(cfg, modelName)
			effTemp := config.EffectivePrimitiveTemperature(cfg, rm.Spec, "choice", temperature, cmd.Flags().Changed("temperature"))

			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()

			opts := engine.Options{
				ModelName:      modelName,
				Primitive:      "choice",
				Rubrics:        rubrics,
				Choices:        parsedChoices,
				ImageTokens:    imageTokens,
				DebiasOrder:    debiasOrder,
				Temperature:    effTemp,
				NoCache:        noCache,
				CacheOnly:      cacheOnly,
				IncludeRecord:  includeRecord,
				StreamProgress: progress,
			}

			items, err := exporter.LoadBatchItemsWithImages(input, idField, contextTemplate, imageField, imageBaseDir)
			if err != nil {
				return err
			}
			report, err := engine.EvaluateBatch(cfg, store, opts, items)
			if err != nil {
				return err
			}
			return exporter.FormatAndWrite(report, format, output)
		},
	}

	cmd.Flags().StringVar(&rubrics, "rubrics", "", "自然语言分类指导准则")
	cmd.Flags().StringVar(&choices, "choices", "", "多分类候选集合，逗号分隔选项，冒号分隔键与描述")
	cmd.Flags().BoolVar(&debiasOrder, "debias-order", false, "执行正反序双向推理并对冲位置偏置")
	cmd.Flags().StringVar(&input, "input", "", "批量输入数据文件路径（.json, .jsonl, .yaml）")
	cmd.Flags().StringVar(&output, "output", "", "结果保存文件路径（缺省输出至 stdout）")
	cmd.Flags().StringVar(&format, "format", "json", "输出格式：json | yaml | jsonl")
	cmd.Flags().StringVar(&model, "model", "gemma-4-26b-a4b", "模型别名或物理路径")
	cmd.Flags().Float64Var(&temperature, "temperature", 1.0, "Logit 概率校准温度")
	cmd.Flags().StringVar(&idField, "id-field", "id", "输入数据中唯一主键字段名称")
	cmd.Flags().StringVar(&contextTemplate, "context-template", "", "拼接多字段为 context 的 Go 模板字符串")
	cmd.Flags().StringVar(&imageField, "image-field", "", "输入记录中图片路径字段名（缺省自动识别 images / image）")
	cmd.Flags().StringVar(&imageBaseDir, "image-base-dir", "", "解析记录中相对图片路径的基准目录")
	cmd.Flags().IntVar(&imageTokens, "image-tokens", 280, "单张图片分配的视觉 Soft Token 预算（70 | 140 | 280 | 560 | 1120）")
	cmd.Flags().BoolVar(&includeRecord, "include-record", false, "在输出结果中附带原始输入对象")
	cmd.Flags().BoolVar(&progress, "progress", false, "在 stderr 实时流式打印 JSONL 处理进度")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "跳过 SQLite 本地缓存读取与写入")
	cmd.Flags().BoolVar(&cacheOnly, "cache-only", false, "仅查本地缓存，不拉起模型推理")

	return cmd
}

func newBatchScoreCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		rubrics         string
		levels          string
		debiasOrder     bool
		input           string
		output          string
		format          string
		model           string
		temperature     float64
		idField         string
		contextTemplate string
		imageField      string
		imageBaseDir    string
		imageTokens     int
		includeRecord   bool
		progress        bool
		noCache         bool
		cacheOnly       bool
	)

	cmd := &cobra.Command{
		Use:   "score",
		Short: "批量文件分级量规打分（输入数据文件，支持文本与图片字段，对每条记录输出等级与期望分值）",
		Long: `batch score 对输入数据文件执行批量量规打分。
自动识别记录中的 image / images 字段（或通过 --image-field 指定），输出每条记录的离散等级概率与期望值（expected_score）。`,
		Example: `  # 批量对故障报告进行严重度量规评级：
  decision-cli batch score \
    --rubrics "工单处理紧急度评级" \
    --levels "P3常规低优,P2重要跟进,P1高优阻断,P0核心崩溃即刻响应" \
    --input bug-reports.yaml \
    --output bug-scores.yaml \
    --format yaml \
    --debias-order`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(rubrics) == "" {
				return fmt.Errorf("缺少 --rubrics 评分指导准则。\n建议操作: 请使用 --rubrics 提供自然语言评分准则。\n正确示例: decision-cli batch score --rubrics \"评估情绪\" --levels \"平静,急躁,愤怒\" --input ./data.json")
			}
			if strings.TrimSpace(input) == "" {
				return fmt.Errorf("缺少 --input 批量数据文件路径。\n建议操作: 请通过 --input 传入 .json, .jsonl 或 .yaml 数据文件。\n正确示例: decision-cli batch score --rubrics \"...\" --levels \"...\" --input ./data.json")
			}
			if strings.TrimSpace(levels) == "" {
				return fmt.Errorf("缺少 --levels 等级刻度列表。\n建议操作: 请提供逗号分隔的升序等级刻度（支持 2 到 10 级）。\n正确示例: --levels \"低风险,中风险,高风险\"")
			}

			parsedLevels, err := engine.ParseScoreFlag(levels)
			if err != nil {
				return err
			}

			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}

			modelName := model
			if !cmd.Flags().Changed("model") && cfg.DefaultModel != "" {
				modelName = cfg.DefaultModel
			}
			if modelName == "" {
				modelName = "gemma-4-26b-a4b"
			}
			rm, _ := infra.ResolveModel(cfg, modelName)
			effTemp := config.EffectivePrimitiveTemperature(cfg, rm.Spec, "score", temperature, cmd.Flags().Changed("temperature"))

			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()

			opts := engine.Options{
				ModelName:      modelName,
				Primitive:      "score",
				Rubrics:        rubrics,
				ScoreLevels:    parsedLevels,
				ImageTokens:    imageTokens,
				DebiasOrder:    debiasOrder,
				Temperature:    effTemp,
				NoCache:        noCache,
				CacheOnly:      cacheOnly,
				IncludeRecord:  includeRecord,
				StreamProgress: progress,
			}

			items, err := exporter.LoadBatchItemsWithImages(input, idField, contextTemplate, imageField, imageBaseDir)
			if err != nil {
				return err
			}
			report, err := engine.EvaluateBatch(cfg, store, opts, items)
			if err != nil {
				return err
			}
			return exporter.FormatAndWrite(report, format, output)
		},
	}

	cmd.Flags().StringVar(&rubrics, "rubrics", "", "自然语言评分指导准则")
	cmd.Flags().StringVar(&levels, "levels", "", "升序等级刻度描述，逗号分隔")
	cmd.Flags().BoolVar(&debiasOrder, "debias-order", false, "执行正反序双向推理并对冲位置偏置")
	cmd.Flags().StringVar(&input, "input", "", "批量输入数据文件路径（.json, .jsonl, .yaml）")
	cmd.Flags().StringVar(&output, "output", "", "结果保存文件路径（缺省输出至 stdout）")
	cmd.Flags().StringVar(&format, "format", "json", "输出格式：json | yaml | jsonl")
	cmd.Flags().StringVar(&model, "model", "gemma-4-26b-a4b", "模型别名或物理路径")
	cmd.Flags().Float64Var(&temperature, "temperature", 1.0, "Logit 概率校准温度")
	cmd.Flags().StringVar(&idField, "id-field", "id", "输入数据中唯一主键字段名称")
	cmd.Flags().StringVar(&contextTemplate, "context-template", "", "拼接多字段为 context 的 Go 模板字符串")
	cmd.Flags().StringVar(&imageField, "image-field", "", "输入记录中图片路径字段名（缺省自动识别 images / image）")
	cmd.Flags().StringVar(&imageBaseDir, "image-base-dir", "", "解析记录中相对图片路径的基准目录")
	cmd.Flags().IntVar(&imageTokens, "image-tokens", 280, "单张图片分配的视觉 Soft Token 预算（70 | 140 | 280 | 560 | 1120）")
	cmd.Flags().BoolVar(&includeRecord, "include-record", false, "在输出结果中附带原始输入对象")
	cmd.Flags().BoolVar(&progress, "progress", false, "在 stderr 实时流式打印 JSONL 处理进度")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "跳过 SQLite 本地缓存读取与写入")
	cmd.Flags().BoolVar(&cacheOnly, "cache-only", false, "仅查本地缓存，不拉起模型推理")

	return cmd
}

// =========================================================================
// 复合问卷与运维管理子命令
// =========================================================================

func newSystemOneCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		inputPathOrJSON string
		outputPath      string
		format          string
		model           string
		rawImages       []string
		imageTokens     int
		temperature     float64
		calibA          float64
		calibB          float64
		debiasOrder     bool
		noCache         bool
		cacheOnly       bool
	)

	cmd := &cobra.Command{
		Use:   "systemone",
		Short: "评估 TypeSafe Jev /v1/systemone 复合多问请求（支持共享文本与多图上下文同时作答多项判定）",
		Long: `systemone 子命令接收符合 TypeSafe Jev /v1/systemone 协议的请求定义（JSON/YAML 文件或内联字符串）。
支持在请求 JSON 中通过 images / assets 字段或通过命令行 --image 传入多张图片，
单次模型前向同时计算多个不同类型（noul 二元、choice 多选、score 打分）的判定问题。`,
		Example: `  # 评估复合问题请求文件：
  decision-cli systemone --input request.json --output response.json --debias-order

  # 附带多张图片评估复合问题：
  decision-cli systemone --input request.json --image ./before.png --image ./after.png --debias-order`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			if strings.TrimSpace(inputPathOrJSON) == "" && len(args) > 0 {
				inputPathOrJSON = args[0]
			}
			if strings.TrimSpace(inputPathOrJSON) == "" {
				return fmt.Errorf("--input 参数为必填项（需提供请求文件路径或内联 JSON 字符串）")
			}
			req, err := exporter.LoadSystemOneRequest(inputPathOrJSON)
			if err != nil {
				return err
			}
			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()

			var ptrA, ptrB *float64
			if cmd.Flags().Changed("calib-a") {
				ptrA = &calibA
			}
			if cmd.Flags().Changed("calib-b") {
				ptrB = &calibB
			}

			baseOpts := engine.Options{
				ModelName:   model,
				Images:      parseImageFlagValues(rawImages),
				ImageTokens: imageTokens,
				DebiasOrder: debiasOrder,
				Threshold:   0.5,
				CalibA:      ptrA,
				CalibB:      ptrB,
				NoCache:     noCache,
				CacheOnly:   cacheOnly,
			}
			resp, err := engine.EvaluateSystemOne(cfg, store, baseOpts, req, temperature, cmd.Flags().Changed("temperature"))
			if err != nil {
				return err
			}
			return exporter.FormatAndWrite(resp, format, outputPath)
		},
	}

	cmd.Flags().StringVar(&inputPathOrJSON, "input", "", "/v1/systemone JSON/YAML 文件路径或内联 JSON 字符串")
	cmd.Flags().StringVar(&outputPath, "output", "", "结果保存文件路径（缺省输出至 stdout）")
	cmd.Flags().StringVar(&format, "format", "json", "输出格式：json | yaml | compact")
	cmd.Flags().StringVar(&model, "model", "", "覆盖指定模型别名（缺省使用请求中的 model 或 gemma-4-26b-a4b）")
	cmd.Flags().StringArrayVar(&rawImages, "image", nil, "附加给本次复合请求的本地图片路径或 data:image Base64")
	cmd.Flags().IntVar(&imageTokens, "image-tokens", 280, "单张图片分配的视觉 Soft Token 预算（70 | 140 | 280 | 560 | 1120）")
	cmd.Flags().Float64Var(&temperature, "temperature", 0, "统一覆盖所有题目的校准温度")
	cmd.Flags().Float64Var(&calibA, "calib-a", 0.0, "Platt 仿射校准缩放系数 A（未指定时默认使用 1/T）")
	cmd.Flags().Float64Var(&calibB, "calib-b", 0.0, "Platt 仿射校准平移常数 B（未指定时默认为 0.0）")
	cmd.Flags().BoolVar(&debiasOrder, "debias-order", false, "对 choice 与 score 题目执行正反序双向去偏置")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "跳过 SQLite 本地缓存读取与写入")
	cmd.Flags().BoolVar(&cacheOnly, "cache-only", false, "仅查本地缓存，不拉起模型推理")

	return cmd
}

func newSetupCommand(rootFlags *RootFlags) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "setup",
		Short: "初始化或修复项目自包含的 Python/MLX 虚拟环境 (~/.cache/decision-cli/venv)",
		Long: `setup 子命令通过 uv 创建并校验 decision-cli 专用的独立 Python 虚拟环境（默认位于 ~/.cache/decision-cli/venv），
安装 mlx、mlx-lm 与 pillow 依赖，确保项目运行完全自包含，不依赖任何外部项目的虚拟环境。`,
		Example: `  # 初始化自包含 MLX 运行时环境：
  decision-cli setup`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			fmt.Printf("🔧 正在检查/构建自包含 MLX Python 虚拟环境 (%s)...\n", cfg.Runtime.VenvDir)
			pyBin, err := infra.EnsureManagedVenv(cfg)
			if err != nil {
				return err
			}
			rt := infra.ResolveRuntime(cfg)
			if !rt.Valid {
				return fmt.Errorf("环境校验失败: %s", rt.Error)
			}
			fmt.Printf("✅ 自包含环境就绪: %s (mlx=%s, mlx_lm=%s, device=%s)\n", pyBin, rt.MLXVersion, rt.MLXLMVer, rt.DeviceInfo)
			return nil
		},
	}
	return cmd
}

func newDoctorCommand(rootFlags *RootFlags) *cobra.Command {
	var targetModel string
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "自检 Apple Silicon 硬件、配置、MLX 运行时并执行金丝雀单步推理测试",
		Long: `doctor 子命令依次自检：
  1. 系统硬件与生效的 config.yaml 配置；
  2. Python 虚拟环境与 MLX/mlx-lm 框架依赖；
  3. 各受支持模型的本地物理路径解析；
  4. 对模型执行正向例、负向例与反转例的金丝雀单步推理验证。`,
		Example: `  # 全量自检：
  decision-cli doctor

  # 仅测试指定模型：
  decision-cli doctor --model gemma-4-26b-a4b`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, cfgPath, created, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			return infra.RunDoctor(cfg, cfgPath, created, targetModel)
		},
	}
	cmd.Flags().StringVar(&targetModel, "model", "", "仅针对特定模型执行金丝雀推理测试")
	return cmd
}

func newModelsCommand(rootFlags *RootFlags) *cobra.Command {
	var format string
	cmd := &cobra.Command{
		Use:   "models",
		Short: "列出受支持的模型清单及其本地磁盘路径解析状态",
		Long:  `models 子命令展示受支持的模型、推荐定位、内存占用预估及当前机器上的物理就绪状态（READY / MISSING）。`,
		Example: `  # 表格展示：
  decision-cli models

  # JSON 结构输出：
  decision-cli models --format json`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, cfgPath, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			aliases := []string{"gemma-4-26b-a4b", "gemma-4-e4b", "qwen3-reranker-4b"}
			var list []infra.ResolvedModel
			for _, a := range aliases {
				rm, _ := infra.ResolveModel(cfg, a)
				list = append(list, rm)
			}
			if format == "json" || format == "yaml" {
				return exporter.FormatAndWrite(map[string]any{
					"config_path":   cfgPath,
					"default_model": cfg.DefaultModel,
					"scan_dirs":     cfg.Models.ScanDirs,
					"models":        list,
				}, format, "")
			}
			fmt.Printf("配置文件路径 : %s\n", cfgPath)
			fmt.Printf("默认模型     : %s\n", cfg.DefaultModel)
			fmt.Printf("扫描目录     : %v\n\n", cfg.Models.ScanDirs)
			for _, m := range list {
				status := "MISSING"
				if m.CachedOnDisk {
					status = "READY"
				}
				defMark := " "
				if m.Alias == cfg.DefaultModel {
					defMark = "*"
				}
				modalityTag := "纯文本 Text-Only"
				if m.Spec.SupportsVision {
					modalityTag = "多模态 Vision+Text"
				}
				fmt.Printf("%s %-18s [%-7s] [%s] 仓库=%s (~%.1fGB, 校准温度=%.2f, 默认阈值=%.2f)\n",
					defMark, m.Alias, status, modalityTag, m.Spec.HFRepoID, m.Spec.ApproxMemoryGB,
					config.EffectiveTemperature(cfg, m.Spec, 0, false),
					config.EffectiveThreshold(cfg, m.Spec, 0, false))
				if m.CachedOnDisk {
					fmt.Printf("    物理路径 : %s (%s)\n", m.LocalPath, m.Source)
				} else {
					fmt.Printf("    下载提示 : 运行 decision-cli models download %s (或 huggingface-cli download %s)\n", m.Alias, m.Spec.HFRepoID)
					fmt.Printf("    配置提示 : 可在 %s 中的 models.paths[%s] 配置物理路径或放置在 scan_dirs 目录下\n", cfgPath, m.Alias)
				}
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&format, "format", "table", "输出格式：table | json | yaml")
	cmd.AddCommand(newModelsDownloadCommand(rootFlags))
	return cmd
}

func newModelsDownloadCommand(rootFlags *RootFlags) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "download [model-name]",
		Short: "通过 huggingface-cli 下载指定模型权重到本地扫描目录",
		Long: `download 子命令解析目标模型的 Hugging Face 仓库 ID，
调用 huggingface-cli 下载至本地 scan_dirs 目录，并打印下载后的具体物理路径。`,
		Example: `  # 下载默认推荐的主力多模态模型 gemma-4-26b-a4b：
  decision-cli models download gemma-4-26b-a4b

  # 下载轻量备选模型 gemma-4-e4b：
  decision-cli models download gemma-4-e4b`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			modelArg := args[0]
			alias, localPath, err := infra.DownloadModel(cfg, modelArg)
			if err != nil {
				return err
			}
			fmt.Printf("\n✅ 模型权重已就绪！\n")
			fmt.Printf("   - 模型别名     : %s\n", alias)
			fmt.Printf("   - 本地物理路径 : %s\n", localPath)
			fmt.Printf("   - 使用示范     : decision-cli noun --model %s --rubrics \"是否符合要求\" --context \"待评估文本\"\n", alias)
			return nil
		},
	}
	return cmd
}

func newCacheCommand(rootFlags *RootFlags) *cobra.Command {
	cacheCmd := &cobra.Command{
		Use:   "cache",
		Short: "查看、导出或清理 SQLite 内容寻址判定缓存",
		Long:  `cache 命令组管理本地 SQLite 缓存库（~/.cache/decision-cli/cache.db）。`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}

	var statsFormat string
	statsCmd := &cobra.Command{
		Use:   "stats",
		Short: "展示 SQLite 缓存条目数与命中率统计",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()
			st, err := store.Stats()
			if err != nil {
				return err
			}
			return exporter.FormatAndWrite(st, statsFormat, "")
		},
	}
	statsCmd.Flags().StringVar(&statsFormat, "format", "json", "输出格式：json | yaml")

	var expRubrics, expModel, expOutput, expFormat string
	var expThreshold, expTemp, expCalibA, expCalibB float64
	exportCmd := &cobra.Command{
		Use:   "export",
		Short: "将缓存的判定记录导出为 YAML 或 JSON 文件",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()
			entries, err := store.QueryEntries(expRubrics, expModel)
			if err != nil {
				return err
			}

			var ptrA, ptrB *float64
			if cmd.Flags().Changed("calib-a") {
				ptrA = &expCalibA
			}
			if cmd.Flags().Changed("calib-b") {
				ptrB = &expCalibB
			}

			results := make([]types.BatchItemResult, 0, len(entries))
			passedCount := 0
			rejectedCount := 0
			for _, e := range entries {
				rm, _ := infra.ResolveModel(cfg, e.ModelID)
				rowTemp := config.EffectiveTemperature(cfg, rm.Spec, expTemp, cmd.Flags().Changed("temperature"))
				rowThresh := config.EffectiveThreshold(cfg, rm.Spec, expThreshold, cmd.Flags().Changed("threshold"))

				curA := ptrA
				curB := ptrB
				if curA == nil && rm.Spec.CalibA != nil {
					curA = rm.Spec.CalibA
				}
				if curB == nil && rm.Spec.CalibB != nil {
					curB = rm.Spec.CalibB
				}

				prob, conf, passed, dec := engine.CalibrateDecision(e.LogitYes, e.LogitNo, engine.Options{
					Threshold:   rowThresh,
					Temperature: rowTemp,
					CalibA:      curA,
					CalibB:      curB,
				})
				if passed {
					passedCount++
				} else {
					rejectedCount++
				}
				results = append(results, types.BatchItemResult{
					ID:          e.RecordID,
					Primitive:   e.Primitive,
					Decision:    dec,
					Passed:      passed,
					Probability: cache.Round4(prob),
					Confidence:  cache.Round4(conf),
					Logits: &types.LogitDetails{
						Yes:    cache.Round4(e.LogitYes),
						No:     cache.Round4(e.LogitNo),
						Margin: cache.Round4(e.LogitMargin),
					},
					Cached:    true,
					LatencyMs: cache.Round4(e.LatencyMs),
					CacheKey:  e.CacheKey,
					Context:   e.Context,
				})
			}
			rep := &types.BatchDecisionReport{
				ExportedAt:  time.Now().Format(time.RFC3339),
				Model:       expModel,
				Rubrics:     expRubrics,
				Threshold:   expThreshold,
				Temperature: expTemp,
				CalibA:      ptrA,
				CalibB:      ptrB,
				Summary: types.BatchSummary{
					Total:         len(results),
					PassedCount:   passedCount,
					RejectedCount: rejectedCount,
					CacheHits:     len(results),
				},
				Results: results,
			}
			return exporter.FormatAndWrite(rep, expFormat, expOutput)
		},
	}
	exportCmd.Flags().StringVar(&expRubrics, "rubrics", "", "按判定准则关键词过滤导出记录")
	exportCmd.Flags().StringVar(&expModel, "model", "", "按模型别名过滤导出记录")
	exportCmd.Flags().StringVar(&expOutput, "output", "", "导出文件保存路径（缺省输出至 stdout）")
	exportCmd.Flags().StringVar(&expFormat, "format", "json", "输出格式：json | yaml | jsonl")
	exportCmd.Flags().Float64Var(&expThreshold, "threshold", 0.5, "用于计算判定结论的概率阈值")
	exportCmd.Flags().Float64Var(&expTemp, "temperature", 1.0, "用于计算校准概率的温度值")
	exportCmd.Flags().Float64Var(&expCalibA, "calib-a", 0.0, "Platt 仿射校准缩放系数 A（未指定时默认使用 1/T）")
	exportCmd.Flags().Float64Var(&expCalibB, "calib-b", 0.0, "Platt 仿射校准平移常数 B（未指定时默认为 0.0）")

	var clrRubrics, clrModel string
	clearCmd := &cobra.Command{
		Use:   "clear",
		Short: "清空 SQLite 缓存记录（支持按准则或模型定向删除）",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()
			if err := store.Clear(clrRubrics, clrModel); err != nil {
				return err
			}
			fmt.Fprintln(os.Stderr, "缓存已成功清空。")
			return nil
		},
	}
	clearCmd.Flags().StringVar(&clrRubrics, "rubrics", "", "仅清除匹配该判定准则的缓存条目")
	clearCmd.Flags().StringVar(&clrModel, "model", "", "仅清除匹配该模型的缓存条目")

	cacheCmd.AddCommand(statsCmd, exportCmd, clearCmd)
	return cacheCmd
}

func newServeCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		host        string
		port        int
		model       string
		debiasOrder bool
		noCache     bool
	)

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "启动常驻 SystemOne HTTP API 服务（监听指定 Host:Port）",
		Long: `serve 子命令拉起一个轻量级常驻 HTTP API 服务，
提供与 TypeSafe Jev 协议兼容的 /v1/systemone 接口，以及 /v1/noun、/v1/choose、/v1/score 等快捷端点。
所有请求共享本地预热驻留的 MLX Worker 与 SQLite 缓存，为外部程序提供毫秒级决策服务。`,
		Example: `  # 在默认 127.0.0.1:8080 启动 API 服务：
  decision-cli serve

  # 暴露给局域网并指定端口与模型：
  decision-cli serve --host 0.0.0.0 --port 8100 --model gemma-4-26b-a4b --debias-order`,
		RunE: func(cmd *cobra.Command, args []string) error {
			opts := server.ServerOptions{
				Host:         host,
				Port:         port,
				DefaultModel: model,
				DebiasOrder:  debiasOrder,
				NoCache:      noCache,
				ConfigPath:   rootFlags.ConfigPath,
			}
			return server.RunServer(opts)
		},
	}

	cmd.Flags().StringVar(&host, "host", "127.0.0.1", "HTTP 服务监听地址")
	cmd.Flags().IntVar(&port, "port", 8080, "HTTP 服务监听端口")
	cmd.Flags().StringVar(&model, "model", "gemma-4-26b-a4b", "服务默认使用的模型别名或物理路径")
	cmd.Flags().BoolVar(&debiasOrder, "debias-order", false, "默认开启选项去偏置")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "跳过 SQLite 本地缓存读取与写入")

	return cmd
}

func newKillCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		force  bool
		dryRun bool
	)

	cmd := &cobra.Command{
		Use:   "kill",
		Short: "扫描并终止所有与 decision-cli 相关的 MLX 守护进程，释放统一内存",
		Long: `kill 子命令扫描系统进程表，自动识别并列出所有相关的 MLX 推理后台进程。
默认先尝试优雅退出，若超时或指定了 --force 则执行 SIGKILL 强杀，并清理残留的 Unix Domain Socket 文件。`,
		Example: `  # 扫描并优雅终止所有相关 MLX 进程：
  decision-cli kill

  # 仅查看当前运行中的相关 MLX 进程（不执行终止）：
  decision-cli kill --dry-run

  # 强制杀死相关进程：
  decision-cli kill --force`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}

			fmt.Println("🔍 正在扫描系统中的相关 MLX 推理进程...")
			procs, err := infra.FindRelatedMLXProcesses(cfg)
			if err != nil {
				return err
			}

			if len(procs) == 0 {
				fmt.Println("✨ 未发现运行中的 decision-cli MLX 相关进程，当前环境整洁。")
				return nil
			}

			fmt.Printf("⚠️ 发现 %d 个与 decision-cli 相关的 MLX 进程：\n", len(procs))
			for _, p := range procs {
				cmdSummary := p.Command
				if len(cmdSummary) > 90 {
					cmdSummary = cmdSummary[:87] + "..."
				}
				fmt.Printf("  - [PID: %d] (内存占用: %s) %s\n", p.PID, p.MemoryDisplay(), cmdSummary)
			}

			if dryRun {
				fmt.Println("\n(--dry-run 模式：仅列出匹配进程，未执行终止动作)")
				return nil
			}

			fmt.Println("\n⚡ 正在终止相关进程...")
			logs, err := infra.KillRelatedMLXProcesses(cfg, procs, force)
			if err != nil {
				return err
			}
			for _, log := range logs {
				fmt.Printf("  - %s\n", log)
			}

			fmt.Println("\n✅ 所有相关 MLX 进程已清理完毕，已释放统一内存。")
			return nil
		},
	}

	cmd.Flags().BoolVar(&force, "force", false, "直接发送 SIGKILL 强杀进程")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "仅列出找到的相关进程，不执行终止")

	return cmd
}

// =========================================================================
// 离线交叉验证阈值与校准调优命令 (`decision-cli tune`)
// =========================================================================

func newTuneCommand(rootFlags *RootFlags) *cobra.Command {
	var (
		inputPath    string
		labelField   string
		minPrecision float64
		minRecall    float64
		beta         float64
		folds        int
		temperature  float64
		format       string
		outputPath   string
	)

	cmd := &cobra.Command{
		Use:   "tune",
		Short: "基于已评测数据或缓存，零推理开销进行分层交叉验证阈值调优与 Platt 校准",
		Long: `tune 子命令基于含有原始 Logits (或 logit margin) 和真值标签的 JSON/JSONL/YAML 数据文件，
完全在本地离线执行分层五折交叉验证（Stratified 5-Fold Cross Validation）。
根据业务要求的最低 Precision / Recall 门槛，在训练折上拟合 Platt 仿射校准参数 (A, B) 并搜索最优截断阈值，
在验证折上对比默认基准 (0.50) 与调优后表现，输出最优生产部署参数与 Brier 分数改善幅度。`,
		Example: `  # 在要求 P>=80%, R>=80% 条件下调优决策阈值：
  decision-cli tune --input eval_results.json --min-precision 0.8 --min-recall 0.8

  # 输出调优结果报告为 HTML 图表：
  decision-cli tune --input eval_results.json --format html --output tune_report.html`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if strings.TrimSpace(inputPath) == "" && len(args) > 0 {
				inputPath = args[0]
			}
			if strings.TrimSpace(inputPath) == "" {
				return fmt.Errorf("缺少 --input 评测数据文件路径。\n示例: decision-cli tune --input eval_results.json --min-precision 0.8 --min-recall 0.8")
			}

			samples, err := tuner.LoadSamplesFromFile(inputPath, labelField)
			if err != nil {
				return fmt.Errorf("解析数据文件失败: %w", err)
			}

			cfg, _, _, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			defTemp := temperature
			if !cmd.Flags().Changed("temperature") && cfg.DefaultTemperature > 0 {
				defTemp = cfg.DefaultTemperature
			}
			if defTemp <= 0 {
				defTemp = 4.5
			}

			opts := tuner.TuneOptions{
				MinPrecision: minPrecision,
				MinRecall:    minRecall,
				Beta:         beta,
				DefaultTemp:  defTemp,
				NumFolds:     folds,
			}

			report, err := tuner.RunCrossValidation(samples, opts)
			if err != nil {
				return fmt.Errorf("交叉验证调优执行失败: %w", err)
			}

			if format == "json" || format == "yaml" {
				return exporter.FormatAndWrite(report, format, outputPath)
			} else if format == "html" {
				htmlContent := generateTuneHTMLReport(report, inputPath, opts)
				if outputPath != "" {
					return os.WriteFile(outputPath, []byte(htmlContent), 0644)
				}
				fmt.Println(htmlContent)
				return nil
			}

			// 默认打印人类可读终端报告
			printTuneTerminalReport(report, inputPath, opts)
			if outputPath != "" {
				_ = exporter.FormatAndWrite(report, "json", outputPath)
			}
			return nil
		},
	}

	cmd.Flags().StringVarP(&inputPath, "input", "i", "", "包含 logits 与标签的评测数据文件路径 (.json, .jsonl, .yaml)")
	cmd.Flags().StringVar(&labelField, "label-field", "", "指定真值标签字段名（缺省自动匹配 label/target/expected/ground_truth）")
	cmd.Flags().Float64Var(&minPrecision, "min-precision", 0.80, "业务最低查准率门槛 (Precision >= X)")
	cmd.Flags().Float64Var(&minRecall, "min-recall", 0.80, "业务最低查全率门槛 (Recall >= X)")
	cmd.Flags().Float64Var(&beta, "beta", 1.0, "F-beta 权重 (1.0 代表 F1，0.5 侧重查准率，2.0 侧重查全率)")
	cmd.Flags().IntVar(&folds, "folds", 5, "分层交叉验证折数")
	cmd.Flags().Float64Var(&temperature, "temperature", 4.5, "对比评估的默认基准温度 (如 Gemma 4 默认 4.5)")
	cmd.Flags().StringVar(&format, "format", "table", "输出格式: table | json | yaml | html")
	cmd.Flags().StringVar(&outputPath, "output", "", "保存报告文件路径 (可选)")

	return cmd
}

func printTuneTerminalReport(rep *tuner.TuneReport, srcFile string, opts tuner.TuneOptions) {
	fmt.Println("================================================================================")
	fmt.Println("           decision-cli tune — 离线分层交叉验证与 Platt 概率校准报告")
	fmt.Println("================================================================================")
	fmt.Printf("数据来源文件 : %s\n", srcFile)
	fmt.Printf("样本规模统计 : 总计 %d 条 (正例: %d, 负例: %d, 真实正例基率: %.2f%%)\n",
		rep.TotalSamples, rep.PositiveSamples, rep.NegativeSamples, rep.BaseRate*100)
	fmt.Printf("业务目标约束 : Precision >= %.1f%%, Recall >= %.1f%% (优化指标: F_%.1f)\n\n",
		opts.MinPrecision*100, opts.MinRecall*100, opts.Beta)

	fmt.Println("[各折交叉验证明细 (Validation Folds)]")
	fmt.Printf("%-6s | %-32s | %-36s | %-16s | %-6s\n",
		"Fold", "默认基准 (τ=0.50, T="+fmt.Sprintf("%.1f", opts.DefaultTemp)+")", "调优后 (最优阈值 τ* + Platt)", "Platt 参数 (A, B)", "达标?")
	fmt.Println(strings.Repeat("-", 108))

	for _, f := range rep.Folds {
		defStr := fmt.Sprintf("P:%.1f%% R:%.1f%% F1:%.3f Brier:%.3f",
			f.DefaultMetrics.Precision*100, f.DefaultMetrics.Recall*100, f.DefaultMetrics.F1, f.DefaultMetrics.Brier)
		tunedStr := fmt.Sprintf("τ=%.2f P:%.1f%% R:%.1f%% F1:%.3f B:%.3f",
			f.OptimalThresh, f.TunedMetrics.Precision*100, f.TunedMetrics.Recall*100, f.TunedMetrics.F1, f.TunedMetrics.Brier)
		abStr := fmt.Sprintf("A=%.4f B=%.4f", f.PlattA, f.PlattB)
		qualStr := "是"
		if !f.MetRequirements {
			qualStr = "否(已回退)"
		}
		fmt.Printf("Fold %-2d | %-32s | %-36s | %-16s | %-6s\n",
			f.FoldIndex, defStr, tunedStr, abStr, qualStr)
	}

	fmt.Println(strings.Repeat("-", 108))
	fmt.Println("\n[5-Fold CV 综合指标对比汇总 (Mean)]")
	fmt.Printf("%-24s | %-22s | %-22s | %-16s\n", "评估维度 / 指标", "默认基线 (τ=0.50)", "调优后 (推荐配置)", "改善幅度")
	fmt.Println(strings.Repeat("-", 90))

	fmt.Printf("%-24s | %-22.1f%% | %-22.1f%% | %+.1f%%\n",
		"准确率 (Accuracy)", rep.MeanDefault.Accuracy*100, rep.MeanTuned.Accuracy*100,
		(rep.MeanTuned.Accuracy-rep.MeanDefault.Accuracy)*100)

	pMeet := ""
	if rep.MeanTuned.Precision >= opts.MinPrecision {
		pMeet = " (已达标)"
	}
	fmt.Printf("%-24s | %-22.1f%% | %-22.1f%%%s | %+.1f%%\n",
		"查准率 (Precision)", rep.MeanDefault.Precision*100, rep.MeanTuned.Precision*100, pMeet,
		(rep.MeanTuned.Precision-rep.MeanDefault.Precision)*100)

	rMeet := ""
	if rep.MeanTuned.Recall >= opts.MinRecall {
		rMeet = " (已达标)"
	}
	fmt.Printf("%-24s | %-22.1f%% | %-22.1f%%%s | %+.1f%%\n",
		"查全率 (Recall)", rep.MeanDefault.Recall*100, rep.MeanTuned.Recall*100, rMeet,
		(rep.MeanTuned.Recall-rep.MeanDefault.Recall)*100)

	fmt.Printf("%-24s | %-22.4f | %-22.4f | %+.4f\n",
		"F1 分数", rep.MeanDefault.F1, rep.MeanTuned.F1, rep.MeanTuned.F1-rep.MeanDefault.F1)

	brierChange := ((rep.MeanTuned.Brier - rep.MeanDefault.Brier) / math.Max(rep.MeanDefault.Brier, 1e-6)) * 100
	fmt.Printf("%-24s | %-22.4f | %-22.4f | %+.1f%% (越低越好)\n",
		"Brier 分数 (概率校准误差)", rep.MeanDefault.Brier, rep.MeanTuned.Brier, brierChange)

	eceChange := ((rep.MeanTuned.ECE - rep.MeanDefault.ECE) / math.Max(rep.MeanDefault.ECE, 1e-6)) * 100
	fmt.Printf("%-24s | %-22.4f | %-22.4f | %+.1f%% (越低越好)\n",
		"ECE 期望校准误差", rep.MeanDefault.ECE, rep.MeanTuned.ECE, eceChange)

	fmt.Println(strings.Repeat("-", 90))

	fmt.Println("\n🎯 生产环境部署建议参数 (Production Flags):")
	fmt.Printf("   --threshold %.2f --calib-a %.4f --calib-b %.4f\n\n",
		rep.RecommendedThresh, rep.RecommendedCalibA, rep.RecommendedCalibB)
}

func generateTuneHTMLReport(rep *tuner.TuneReport, srcFile string, opts tuner.TuneOptions) string {
	var rows strings.Builder
	for _, f := range rep.Folds {
		qualBadge := `<span class="badge badge-success">达标</span>`
		if !f.MetRequirements {
			qualBadge = `<span class="badge badge-warning">未达标(回退)</span>`
		}
		rows.WriteString(fmt.Sprintf(`<tr>
  <td>Fold %d</td>
  <td>P: %.1f%% | R: %.1f%% | F1: %.3f (Brier: %.3f)</td>
  <td><strong>τ=%.2f</strong> | P: %.1f%% | R: %.1f%% | F1: %.3f (Brier: %.3f)</td>
  <td><code>A=%.4f, B=%.4f</code></td>
  <td>%s</td>
</tr>`, f.FoldIndex,
			f.DefaultMetrics.Precision*100, f.DefaultMetrics.Recall*100, f.DefaultMetrics.F1, f.DefaultMetrics.Brier,
			f.OptimalThresh, f.TunedMetrics.Precision*100, f.TunedMetrics.Recall*100, f.TunedMetrics.F1, f.TunedMetrics.Brier,
			f.PlattA, f.PlattB, qualBadge))
	}

	brierPct := ((rep.MeanTuned.Brier - rep.MeanDefault.Brier) / math.Max(rep.MeanDefault.Brier, 1e-6)) * 100

	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>decision-cli tune — 交叉验证与校准报告</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0d1117; color: #c9d1d9; padding: 28px; line-height: 1.6; }
    .container { max-width: 960px; margin: 0 auto; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    h1 { color: #fff; margin-top: 0; font-size: 22px; }
    .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
    .stat-box { background: #21262d; border-radius: 6px; padding: 12px 16px; border: 1px solid #30363d; }
    .stat-val { font-size: 24px; font-weight: bold; color: #58a6ff; }
    .stat-sub { font-size: 12px; color: #8b949e; }
    table { width: 100%%; border-collapse: collapse; margin-top: 14px; font-size: 13px; }
    th, td { border: 1px solid #30363d; padding: 10px 12px; text-align: left; }
    th { background: #21262d; color: #fff; }
    .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; }
    .badge-success { background: rgba(63,185,80,0.2); color: #3fb950; }
    .badge-warning { background: rgba(210,153,34,0.2); color: #d29922; }
    code { background: #21262d; padding: 2px 6px; border-radius: 4px; color: #ff7b72; font-family: monospace; }
    .rec-box { background: rgba(88,166,255,0.1); border: 1px solid #58a6ff; border-radius: 6px; padding: 16px; margin-top: 20px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>decision-cli tune — 离线分层五折交叉验证与概率校准报告</h1>
      <p style="color:#8b949e; font-size:13px;">数据源: %s | 总样本: %d (正例基率: %.1f%%) | 约束: P&ge;%.0f%%, R&ge;%.0f%%</p>
      
      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-val">%.1f%%</div>
          <div class="stat-sub">调优后查准率 (原 %.1f%%)</div>
        </div>
        <div class="stat-box">
          <div class="stat-val">%.1f%%</div>
          <div class="stat-sub">调优后查全率 (原 %.1f%%)</div>
        </div>
        <div class="stat-box">
          <div class="stat-val">%.3f</div>
          <div class="stat-sub">调优后 F1 (原 %.3f)</div>
        </div>
        <div class="stat-box">
          <div class="stat-val" style="color:#3fb950;">%+.1f%%</div>
          <div class="stat-sub">Brier 误差变化 (越负越优)</div>
        </div>
      </div>

      <h3>分折交叉验证明细</h3>
      <table>
        <thead>
          <tr>
            <th>Fold</th>
            <th>默认基线 (τ=0.50)</th>
            <th>调优后 (τ* + Platt)</th>
            <th>拟合参数 (A, B)</th>
            <th>达标情况</th>
          </tr>
        </thead>
        <tbody>
          %s
        </tbody>
      </table>

      <div class="rec-box">
        <h4 style="margin:0 0 8px 0; color:#58a6ff;">🎯 推荐生产部署参数</h4>
        <code>--threshold %.2f --calib-a %.4f --calib-b %.4f</code>
      </div>
    </div>
  </div>
</body>
</html>`, srcFile, rep.TotalSamples, rep.BaseRate*100, opts.MinPrecision*100, opts.MinRecall*100,
		rep.MeanTuned.Precision*100, rep.MeanDefault.Precision*100,
		rep.MeanTuned.Recall*100, rep.MeanDefault.Recall*100,
		rep.MeanTuned.F1, rep.MeanDefault.F1,
		brierPct,
		rows.String(),
		rep.RecommendedThresh, rep.RecommendedCalibA, rep.RecommendedCalibB)
}


