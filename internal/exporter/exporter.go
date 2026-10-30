package exporter

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	"github.com/cafe3310/decision-cli/internal/types"
	"gopkg.in/yaml.v3"
)

// LoadBatchItems reads a batch input file (.json, .jsonl, .yaml, .yml) and extracts BatchItemInput records.
func LoadBatchItems(inputPath, idField, contextTemplate string) ([]types.BatchItemInput, error) {
	return LoadBatchItemsWithImages(inputPath, idField, contextTemplate, "", "")
}

// LoadBatchItemsWithImages reads a batch input file and extracts ID, Context, and per-record Images.
func LoadBatchItemsWithImages(inputPath, idField, contextTemplate, imageField, imageBaseDir string) ([]types.BatchItemInput, error) {
	rawBytes, err := os.ReadFile(inputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read input file %s: %w", inputPath, err)
	}

	ext := strings.ToLower(filepath.Ext(inputPath))
	var rawRecords []map[string]any

	if ext == ".jsonl" {
		scanner := bufio.NewScanner(bytes.NewReader(rawBytes))
		buf := make([]byte, 0, 1024*1024)
		scanner.Buffer(buf, 10*1024*1024)
		lineNum := 0
		for scanner.Scan() {
			lineNum++
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(line), &obj); err != nil {
				return nil, fmt.Errorf("jsonl parse error at line %d: %w", lineNum, err)
			}
			rawRecords = append(rawRecords, obj)
		}
		if err := scanner.Err(); err != nil {
			return nil, err
		}
	} else if ext == ".yaml" || ext == ".yml" {
		var generic any
		if err := yaml.Unmarshal(rawBytes, &generic); err != nil {
			return nil, fmt.Errorf("yaml parse error in %s: %w", inputPath, err)
		}
		rawRecords, err = extractRecordSlice(generic)
		if err != nil {
			return nil, err
		}
	} else {
		// Default JSON (with fallback to YAML if JSON fails)
		var generic any
		if err := json.Unmarshal(rawBytes, &generic); err != nil {
			if yErr := yaml.Unmarshal(rawBytes, &generic); yErr != nil {
				return nil, fmt.Errorf("failed to parse %s as JSON (%v) or YAML (%v)", inputPath, err, yErr)
			}
		}
		rawRecords, err = extractRecordSlice(generic)
		if err != nil {
			return nil, err
		}
	}

	var tmpl *template.Template
	if strings.TrimSpace(contextTemplate) != "" {
		tmpl, err = template.New("ctx").Option("missingkey=zero").Parse(contextTemplate)
		if err != nil {
			return nil, fmt.Errorf("invalid --context-template: %w", err)
		}
	}

	inputDir := filepath.Dir(inputPath)
	items := make([]types.BatchItemInput, 0, len(rawRecords))
	for idx, rec := range rawRecords {
		id := extractRecordID(rec, idField, idx)
		ctxStr, err := buildRecordContext(rec, tmpl)
		if err != nil {
			return nil, fmt.Errorf("failed to build context for record %s: %w", id, err)
		}
		imgs := extractRecordImages(rec, imageField, imageBaseDir, inputDir)
		items = append(items, types.BatchItemInput{
			ID:      id,
			Context: ctxStr,
			Images:  imgs,
			Raw:     rec,
		})
	}
	return items, nil
}

func extractRecordImages(rec map[string]any, imageField, imageBaseDir, inputDir string) []string {
	var fields []string
	if strings.TrimSpace(imageField) != "" {
		fields = []string{strings.TrimSpace(imageField)}
	} else {
		fields = []string{"images", "image"}
	}
	var rawList []string
	for _, f := range fields {
		val, ok := rec[f]
		if !ok || val == nil {
			continue
		}
		switch v := val.(type) {
		case string:
			s := strings.TrimSpace(v)
			if s != "" {
				rawList = append(rawList, s)
			}
		case []string:
			for _, elem := range v {
				s := strings.TrimSpace(elem)
				if s != "" {
					rawList = append(rawList, s)
				}
			}
		case []any:
			for _, elem := range v {
				s := strings.TrimSpace(fmt.Sprintf("%v", elem))
				if s != "" && s != "<nil>" {
					rawList = append(rawList, s)
				}
			}
		}
		if len(rawList) > 0 {
			break
		}
	}
	if len(rawList) == 0 {
		return nil
	}
	resolved := make([]string, 0, len(rawList))
	for _, img := range rawList {
		if strings.HasPrefix(img, "data:image/") || strings.HasPrefix(img, "base64:") || strings.HasPrefix(img, "file://") || filepath.IsAbs(img) || strings.HasPrefix(img, "~/") {
			resolved = append(resolved, img)
			continue
		}
		if imageBaseDir != "" {
			resolved = append(resolved, filepath.Join(imageBaseDir, img))
			continue
		}
		// If relative path does not exist in CWD, check relative to batch input file directory
		if _, err := os.Stat(img); err != nil && inputDir != "" && inputDir != "." {
			cand := filepath.Join(inputDir, img)
			if _, cErr := os.Stat(cand); cErr == nil {
				resolved = append(resolved, cand)
				continue
			}
		}
		resolved = append(resolved, img)
	}
	return resolved
}

func extractRecordSlice(generic any) ([]map[string]any, error) {
	switch v := generic.(type) {
	case []any:
		out := make([]map[string]any, 0, len(v))
		for i, elem := range v {
			if m, ok := elem.(map[string]any); ok {
				out = append(out, m)
			} else {
				out = append(out, map[string]any{"id": fmt.Sprintf("item-%d", i+1), "context": fmt.Sprintf("%v", elem)})
			}
		}
		return out, nil
	case map[string]any:
		// Check common wrapper keys in hierarchical dataset envelopes
		for _, key := range []string{"insights", "records", "items", "results", "data"} {
			if arr, ok := v[key]; ok {
				if slice, isSlice := arr.([]any); isSlice {
					return extractRecordSlice(slice)
				}
			}
		}
		return []map[string]any{v}, nil
	default:
		return nil, fmt.Errorf("unsupported batch root structure %T", generic)
	}
}

func extractRecordID(rec map[string]any, idField string, idx int) string {
	candidates := []string{idField, "id", "insight_id", "record_id", "uid", "key"}
	for _, k := range candidates {
		if k == "" {
			continue
		}
		if val, ok := rec[k]; ok && val != nil {
			s := strings.TrimSpace(fmt.Sprintf("%v", val))
			if s != "" {
				return s
			}
		}
	}
	return fmt.Sprintf("item-%04d", idx+1)
}

func buildRecordContext(rec map[string]any, tmpl *template.Template) (string, error) {
	if tmpl != nil {
		var buf bytes.Buffer
		if err := tmpl.Execute(&buf, rec); err != nil {
			return "", err
		}
		return strings.TrimSpace(buf.String()), nil
	}

	// Default field extraction from text/content/context fields
	for _, k := range []string{"context", "text", "content"} {
		if val, ok := rec[k]; ok && val != nil {
			s := strings.TrimSpace(fmt.Sprintf("%v", val))
			if s != "" {
				return s, nil
			}
		}
	}

	// Compose context from standard structured feedback and annotation fields
	var parts []string
	for _, field := range []string{"dimension", "target_model", "service_scenario", "user_expression", "suggested_action", "title", "summary", "raw_excerpt", "raw_text"} {
		if val, ok := rec[field]; ok && val != nil {
			s := strings.TrimSpace(fmt.Sprintf("%v", val))
			if s != "" && s != "<nil>" {
				parts = append(parts, fmt.Sprintf("%s: %s", field, s))
			}
		}
	}
	if len(parts) > 0 {
		return strings.Join(parts, "\n"), nil
	}

	// Fallback: compact JSON representation of the map
	b, err := json.Marshal(rec)
	if err != nil {
		return fmt.Sprintf("%v", rec), nil
	}
	return string(b), nil
}

// LoadSystemOneRequest reads a TypeSafe Jev /v1/systemone request from a JSON/YAML file or raw JSON string.
func LoadSystemOneRequest(pathOrRaw string) (*types.SystemOneRequest, error) {
	trimmed := strings.TrimSpace(pathOrRaw)
	var rawBytes []byte
	var err error
	if strings.HasPrefix(trimmed, "{") {
		rawBytes = []byte(trimmed)
	} else {
		rawBytes, err = os.ReadFile(trimmed)
		if err != nil {
			return nil, fmt.Errorf("failed to read systemone input %s: %w", trimmed, err)
		}
	}
	var req types.SystemOneRequest
	if err := json.Unmarshal(rawBytes, &req); err != nil {
		if yErr := yaml.Unmarshal(rawBytes, &req); yErr != nil {
			return nil, fmt.Errorf("failed to parse systemone request as JSON (%v) or YAML (%v)", err, yErr)
		}
	}
	if len(req.Questions) == 0 {
		return nil, fmt.Errorf("systemone request must contain at least one question in 'questions'")
	}
	return &req, nil
}
// Supported formats: json, yaml, jsonl, compact, label-only.
func FormatAndWrite(v any, format string, outputPath string) error {
	// If outputPath is given and format is auto/default, infer file format for the file write
	if outputPath != "" {
		fileFormat := inferFormatFromPath(outputPath, format)
		fileBytes, err := serialize(v, fileFormat)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
			return err
		}
		if err := os.WriteFile(outputPath, fileBytes, 0644); err != nil {
			return fmt.Errorf("failed to write output file %s: %w", outputPath, err)
		}
	}

	stdoutBytes, err := serialize(v, format)
	if err != nil {
		return err
	}
	_, err = os.Stdout.Write(stdoutBytes)
	return err
}

func inferFormatFromPath(path, fallback string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".yaml", ".yml":
		return "yaml"
	case ".jsonl":
		return "jsonl"
	case ".json":
		return "json"
	default:
		if fallback == "" {
			return "json"
		}
		return fallback
	}
}

func serialize(v any, format string) ([]byte, error) {
	format = strings.ToLower(strings.TrimSpace(format))
	if format == "" {
		format = "json"
	}

	switch format {
	case "yaml", "yml":
		var buf bytes.Buffer
		enc := yaml.NewEncoder(&buf)
		enc.SetIndent(2)
		if err := enc.Encode(v); err != nil {
			return nil, err
		}
		_ = enc.Close()
		return buf.Bytes(), nil

	case "jsonl":
		if rep, ok := v.(*types.BatchDecisionReport); ok {
			var buf bytes.Buffer
			for _, item := range rep.Results {
				line, err := json.Marshal(item)
				if err != nil {
					return nil, err
				}
				buf.Write(line)
				buf.WriteByte('\n')
			}
			return buf.Bytes(), nil
		}
		b, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		return append(b, '\n'), nil

	case "compact":
		b, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		return append(b, '\n'), nil

	case "label-only":
		switch val := v.(type) {
		case *types.SingleDecisionResult:
			return []byte(val.Decision + "\n"), nil
		case types.SingleDecisionResult:
			return []byte(val.Decision + "\n"), nil
		case *types.SingleChoiceResult:
			return []byte(val.Choice + "\n"), nil
		case types.SingleChoiceResult:
			return []byte(val.Choice + "\n"), nil
		case *types.SingleScoreResult:
			return []byte(fmt.Sprintf("%.4f\t%s\n", val.Score, val.TopLabel)), nil
		case types.SingleScoreResult:
			return []byte(fmt.Sprintf("%.4f\t%s\n", val.Score, val.TopLabel)), nil
		case *types.BatchDecisionReport:
			var sb strings.Builder
			for _, r := range val.Results {
				if r.Primitive == "choice" {
					sb.WriteString(fmt.Sprintf("%s\t%s\t%.4f\n", r.ID, r.Choice, r.Confidence))
				} else if r.Primitive == "score" && r.Score != nil {
					sb.WriteString(fmt.Sprintf("%s\t%.4f\t%s\n", r.ID, *r.Score, r.Decision))
				} else {
					sb.WriteString(fmt.Sprintf("%s\t%s\t%.4f\n", r.ID, r.Decision, r.Probability))
				}
			}
			return []byte(sb.String()), nil
		}
		b, err := json.Marshal(v)
		return append(b, '\n'), err

	default: // "json"
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		enc.SetIndent("", "  ")
		if err := enc.Encode(v); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil
	}
}
