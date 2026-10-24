package cli

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/cafe3310/decision-cli/internal/cache"
	"github.com/cafe3310/decision-cli/internal/config"
	"github.com/cafe3310/decision-cli/internal/engine"
	"github.com/cafe3310/decision-cli/internal/exporter"
	"github.com/cafe3310/decision-cli/internal/infra"
	"github.com/cafe3310/decision-cli/internal/types"
	"github.com/spf13/cobra"
)

// RootFlags holds CLI flags bound to the root command.
type RootFlags struct {
	ConfigPath      string
	ShowUsage       bool
	Rubrics         string
	Context         string
	InputPath       string
	OutputPath      string
	Format          string
	Model           string
	Threshold       float64
	Temperature     float64
	IDField         string
	ContextTemplate string
	FilterPassed    bool
	IncludeRecord   bool
	NoCache         bool
	CacheOnly       bool
	Progress        bool
}

// NewRootCommand constructs the Cobra CLI command tree for decision-cli.
func NewRootCommand() *cobra.Command {
	var flags RootFlags

	rootCmd := &cobra.Command{
		Use:   "decision-cli",
		Short: "Local Apple Silicon (MLX) non-autoregressive rubric decision CLI with SQLite cache",
		Long: `decision-cli evaluates whether text contexts satisfy a natural-language rubric ("是" / "否")
using single-pass prefill logits on Apple Silicon (MLX) and deterministic SQLite caching.

Global configuration is stored at ~/.config/decision-cli/config.yaml (auto-created on first run).
Run 'decision-cli --usage' for the complete Agent & Developer JSON/YAML integration manual.`,
		SilenceUsage:  true,
		SilenceErrors: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if flags.ShowUsage {
				fmt.Print(AgentUsageManual)
				return nil
			}

			cfg, cfgPath, _, err := config.LoadOrInit(flags.ConfigPath)
			if err != nil {
				return err
			}
			_ = cfgPath

			if strings.TrimSpace(flags.Rubrics) == "" {
				return fmt.Errorf("--rubrics is required (or run 'decision-cli --help', 'decision-cli --usage', or 'decision-cli doctor')")
			}

			modelName := flags.Model
			if modelName == "" {
				modelName = cfg.DefaultModel
			}
			rm, _ := infra.ResolveModel(cfg, modelName)
			threshold := config.EffectiveThreshold(cfg, rm.Spec, flags.Threshold, cmd.Flags().Changed("threshold"))
			temperature := config.EffectiveTemperature(cfg, rm.Spec, flags.Temperature, cmd.Flags().Changed("temperature"))

			store, err := cache.Open(cfg.Cache.DBPath)
			if err != nil {
				return err
			}
			defer store.Close()

			opts := engine.Options{
				ModelName:      modelName,
				Rubrics:        flags.Rubrics,
				Threshold:      threshold,
				Temperature:    temperature,
				NoCache:        flags.NoCache,
				CacheOnly:      flags.CacheOnly,
				FilterPassed:   flags.FilterPassed,
				IncludeRecord:  flags.IncludeRecord,
				StreamProgress: flags.Progress,
			}

			// Batch mode if --input / --batch-file is provided
			if strings.TrimSpace(flags.InputPath) != "" {
				items, err := exporter.LoadBatchItems(flags.InputPath, flags.IDField, flags.ContextTemplate)
				if err != nil {
					return err
				}
				report, err := engine.EvaluateBatch(cfg, store, opts, items)
				if err != nil {
					return err
				}
				return exporter.FormatAndWrite(report, flags.Format, flags.OutputPath)
			}

			// Single item mode
			if strings.TrimSpace(flags.Context) == "" {
				return fmt.Errorf("either --context <text> or --input <file.{yaml,json,jsonl}> must be provided")
			}
			res, err := engine.EvaluateSingle(cfg, store, opts, flags.Context)
			if err != nil {
				return err
			}
			return exporter.FormatAndWrite(res, flags.Format, flags.OutputPath)
		},
	}

	pf := rootCmd.PersistentFlags()
	pf.StringVar(&flags.ConfigPath, "config", "", "Path to config file (default ~/.config/decision-cli/config.yaml)")

	f := rootCmd.Flags()
	f.BoolVar(&flags.ShowUsage, "usage", false, "Print structured Markdown manual for AI agents and developers")
	f.StringVarP(&flags.Rubrics, "rubrics", "r", "", "Natural-language rubric / decision criteria")
	f.StringVarP(&flags.Context, "context", "c", "", "Single item text context to evaluate")
	f.StringVarP(&flags.InputPath, "input", "i", "", "Batch input file (.yaml, .yml, .json, .jsonl)")
	f.StringVar(&flags.InputPath, "batch-file", "", "Alias for --input")
	f.StringVarP(&flags.OutputPath, "output", "o", "", "Output file path (.yaml, .json, .jsonl)")
	f.StringVarP(&flags.Format, "format", "f", "json", "Output format: json | yaml | jsonl | compact | label-only")
	f.StringVarP(&flags.Model, "model", "m", "", "Model alias or path (gemma-4-26b-a4b | gemma-4-e4b | qwen3-reranker-4b)")
	f.Float64Var(&flags.Threshold, "threshold", 0.5, "Probability threshold for decision=='是' (0.0 ~ 1.0)")
	f.Float64Var(&flags.Temperature, "temperature", 1.0, "Logit calibration temperature (> 0.0)")
	f.StringVar(&flags.IDField, "id-field", "id", "Record ID field name in batch mode")
	f.StringVar(&flags.ContextTemplate, "context-template", "", "Go text/template for constructing context in batch mode")
	f.BoolVar(&flags.FilterPassed, "filter-passed", false, "In batch mode, include only items where decision=='是'")
	f.BoolVar(&flags.IncludeRecord, "include-record", false, "In batch mode, attach original raw record to each output item")
	f.BoolVar(&flags.NoCache, "no-cache", false, "Bypass SQLite cache read and write")
	f.BoolVar(&flags.CacheOnly, "cache-only", false, "Never invoke MLX model; return only cached results")
	f.BoolVar(&flags.Progress, "progress", false, "In batch mode, stream JSONL progress and incremental item results to stderr")

	// Subcommands
	rootCmd.AddCommand(newDoctorCommand(&flags))
	rootCmd.AddCommand(newModelsCommand(&flags))
	rootCmd.AddCommand(newCacheCommand(&flags))

	return rootCmd
}

func newDoctorCommand(rootFlags *RootFlags) *cobra.Command {
	var targetModel string
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Check Apple Silicon environment, active config.yaml, MLX runtime, model paths, and run canary inference",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, cfgPath, created, err := config.LoadOrInit(rootFlags.ConfigPath)
			if err != nil {
				return err
			}
			return infra.RunDoctor(cfg, cfgPath, created, targetModel)
		},
	}
	cmd.Flags().StringVarP(&targetModel, "model", "m", "", "Specific model alias to test during canary inference")
	return cmd
}

func newModelsCommand(rootFlags *RootFlags) *cobra.Command {
	var format string
	cmd := &cobra.Command{
		Use:   "models",
		Short: "List supported models and their local path resolution status",
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
			fmt.Printf("Config File   : %s\n", cfgPath)
			fmt.Printf("Default Model : %s\n", cfg.DefaultModel)
			fmt.Printf("Scan Dirs     : %v\n\n", cfg.Models.ScanDirs)
			for _, m := range list {
				status := "MISSING"
				if m.CachedOnDisk {
					status = "READY"
				}
				defMark := " "
				if m.Alias == cfg.DefaultModel {
					defMark = "*"
				}
				fmt.Printf("%s %-18s [%-7s] repo=%s (~%.1fGB, calibration_temp=%.2f, threshold=%.2f)\n",
					defMark, m.Alias, status, m.Spec.HFRepoID, m.Spec.ApproxMemoryGB,
					config.EffectiveTemperature(cfg, m.Spec, 0, false),
					config.EffectiveThreshold(cfg, m.Spec, 0, false))
				if m.CachedOnDisk {
					fmt.Printf("    path   : %s (%s)\n", m.LocalPath, m.Source)
				} else {
					fmt.Printf("    hint   : configure models.paths[%s] in %s or place weights in scan_dirs\n", m.Alias, cfgPath)
				}
			}
			return nil
		},
	}
	cmd.Flags().StringVarP(&format, "format", "f", "table", "Output format: table | json | yaml")
	return cmd
}

func newCacheCommand(rootFlags *RootFlags) *cobra.Command {
	cacheCmd := &cobra.Command{
		Use:   "cache",
		Short: "Inspect, export, or clear the deterministic SQLite decision cache",
	}

	var statsFormat string
	statsCmd := &cobra.Command{
		Use:   "stats",
		Short: "Show SQLite cache entry and hit statistics",
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
	statsCmd.Flags().StringVarP(&statsFormat, "format", "f", "json", "Output format: json | yaml")

	var expRubrics, expModel, expOutput, expFormat string
	var expThreshold, expTemp float64
	exportCmd := &cobra.Command{
		Use:   "export",
		Short: "Export cached decisions to YAML or JSON",
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
			results := make([]types.BatchItemResult, 0, len(entries))
			passedCount := 0
			rejectedCount := 0
			for _, e := range entries {
				rm, _ := infra.ResolveModel(cfg, e.ModelID)
				rowTemp := config.EffectiveTemperature(cfg, rm.Spec, expTemp, cmd.Flags().Changed("temperature"))
				rowThresh := config.EffectiveThreshold(cfg, rm.Spec, expThreshold, cmd.Flags().Changed("threshold"))
				prob, conf, passed, dec := cache.CalibrateFromLogits(e.LogitYes, e.LogitNo, rowTemp, rowThresh)
				if passed {
					passedCount++
				} else {
					rejectedCount++
				}
				results = append(results, types.BatchItemResult{
					ID:          e.RecordID,
					Decision:    dec,
					Passed:      passed,
					Probability: cache.Round4(prob),
					Confidence:  cache.Round4(conf),
					Logits: types.LogitDetails{
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
	exportCmd.Flags().StringVarP(&expRubrics, "rubrics", "r", "", "Filter exported cache entries by rubrics text")
	exportCmd.Flags().StringVarP(&expModel, "model", "m", "", "Filter exported cache entries by model ID")
	exportCmd.Flags().StringVarP(&expOutput, "output", "o", "", "Output file path (.yaml or .json)")
	exportCmd.Flags().StringVarP(&expFormat, "format", "f", "json", "Output format: json | yaml | jsonl")
	exportCmd.Flags().Float64Var(&expThreshold, "threshold", 0.5, "Threshold used to compute decision")
	exportCmd.Flags().Float64Var(&expTemp, "temperature", 1.0, "Temperature used to compute probability")

	var clrRubrics, clrModel string
	clearCmd := &cobra.Command{
		Use:   "clear",
		Short: "Delete cached entries (all or filtered by --rubrics / --model)",
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
			fmt.Fprintln(os.Stderr, "Cache cleared.")
			return nil
		},
	}
	clearCmd.Flags().StringVarP(&clrRubrics, "rubrics", "r", "", "Clear only entries matching this rubrics text")
	clearCmd.Flags().StringVarP(&clrModel, "model", "m", "", "Clear only entries matching this model")

	cacheCmd.AddCommand(statsCmd, exportCmd, clearCmd)
	return cacheCmd
}
