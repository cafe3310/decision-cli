package cache

import (
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cafe3310/decision-cli/internal/config"
	"github.com/cafe3310/decision-cli/internal/types"
	_ "modernc.org/sqlite"
)

const CacheKeySchemaVersion = "v1"

var SupportedImageTokens = []int{70, 140, 280, 560, 1120}

// NormalizeImageTokens snaps token budget to the nearest Gemma 4 supported soft token budget {70, 140, 280, 560, 1120}.
func NormalizeImageTokens(tokens int) int {
	if tokens <= 0 {
		return 280
	}
	best := 280
	bestDiff := math.MaxInt32
	for _, s := range SupportedImageTokens {
		d := tokens - s
		if d < 0 {
			d = -d
		}
		if d < bestDiff {
			bestDiff = d
			best = s
		}
	}
	return best
}

// ResolveAndHashImages resolves file paths to absolute paths, validates existence or base64 data,
// and computes a deterministic content-addressable signature over all image bytes and token budget.
func ResolveAndHashImages(images []string, imageTokens int) (resolved []string, imageSig string, normTokens int, err error) {
	if len(images) == 0 {
		return nil, "", 0, nil
	}
	normTokens = NormalizeImageTokens(imageTokens)
	resolved = make([]string, 0, len(images))
	hashes := make([]string, 0, len(images))

	for idx, raw := range images {
		spec := strings.TrimSpace(raw)
		if spec == "" {
			continue
		}
		var data []byte
		var finalSpec string
		switch {
		case strings.HasPrefix(spec, "data:image/"):
			comma := strings.Index(spec, ",")
			if comma < 0 {
				return nil, "", 0, fmt.Errorf("image #%d: invalid data:image URI (missing comma)", idx+1)
			}
			decoded, bErr := base64.StdEncoding.DecodeString(spec[comma+1:])
			if bErr != nil {
				return nil, "", 0, fmt.Errorf("image #%d: invalid base64 in data URI: %w", idx+1, bErr)
			}
			data = decoded
			finalSpec = spec
		case strings.HasPrefix(spec, "base64:"):
			decoded, bErr := base64.StdEncoding.DecodeString(strings.TrimPrefix(spec, "base64:"))
			if bErr != nil {
				return nil, "", 0, fmt.Errorf("image #%d: invalid base64 payload: %w", idx+1, bErr)
			}
			data = decoded
			finalSpec = spec
		default:
			p := strings.TrimPrefix(spec, "file://")
			p = config.ExpandPath(p)
			if abs, aErr := filepath.Abs(p); aErr == nil {
				p = abs
			}
			fileBytes, fErr := os.ReadFile(p)
			if fErr != nil {
				return nil, "", 0, fmt.Errorf("无法读取图片 #%d (%s): %w", idx+1, raw, fErr)
			}
			data = fileBytes
			finalSpec = p
		}
		sum := sha256.Sum256(data)
		hashes = append(hashes, hex.EncodeToString(sum[:]))
		resolved = append(resolved, finalSpec)
	}
	if len(resolved) == 0 {
		return nil, "", 0, nil
	}
	imageSig = fmt.Sprintf("imgtok=%d|%s", normTokens, strings.Join(hashes, ","))
	return resolved, imageSig, normTokens, nil
}

// NormalizeText trims surrounding whitespace and normalizes CRLF to LF.
func NormalizeText(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	return strings.TrimSpace(s)
}

// HashString returns the lowercase hex SHA-256 of s.
func HashString(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

// ComputeCacheKey computes the deterministic content-addressable key:
// SHA256("v1|" + canonicalModelID + "|" + NormalizeText(rubrics) + "|" + NormalizeText(context))
// Different models strictly produce different cache keys and never collide.
func ComputeCacheKey(canonicalModelID, rubrics, context string) string {
	return ComputeCacheKeyWithImages(canonicalModelID, rubrics, context, "")
}

// ComputeCacheKeyWithImages includes the content-addressed image signature when non-empty.
func ComputeCacheKeyWithImages(canonicalModelID, rubrics, context, imageSig string) string {
	if imageSig == "" {
		payload := fmt.Sprintf("%s|%s|%s|%s",
			CacheKeySchemaVersion,
			strings.TrimSpace(canonicalModelID),
			NormalizeText(rubrics),
			NormalizeText(context),
		)
		return HashString(payload)
	}
	payload := fmt.Sprintf("v2|%s|noul|%s|%s|%s",
		strings.TrimSpace(canonicalModelID),
		NormalizeText(rubrics),
		NormalizeText(context),
		imageSig,
	)
	return HashString(payload)
}

// ComputeMultiCacheKey computes the deterministic content-addressable key for choice/score primitives:
// SHA256("v2|" + canonicalModelID + "|" + primitive + "|" + debiasFlag + "|" + NormalizeText(rubrics) + "|" + criteriaJSON + "|" + NormalizeText(context))
func ComputeMultiCacheKey(canonicalModelID, primitive, rubrics, criteriaJSON, context string, debiasOrder bool) string {
	return ComputeMultiCacheKeyWithImages(canonicalModelID, primitive, rubrics, criteriaJSON, context, debiasOrder, "")
}

// ComputeMultiCacheKeyWithImages includes the content-addressed image signature when non-empty.
func ComputeMultiCacheKeyWithImages(canonicalModelID, primitive, rubrics, criteriaJSON, context string, debiasOrder bool, imageSig string) string {
	debiasStr := "0"
	if debiasOrder {
		debiasStr = "1"
	}
	if imageSig == "" {
		payload := fmt.Sprintf("v2|%s|%s|%s|%s|%s|%s",
			strings.TrimSpace(canonicalModelID),
			strings.TrimSpace(primitive),
			debiasStr,
			NormalizeText(rubrics),
			strings.TrimSpace(criteriaJSON),
			NormalizeText(context),
		)
		return HashString(payload)
	}
	payload := fmt.Sprintf("v2|%s|%s|%s|%s|%s|%s|%s",
		strings.TrimSpace(canonicalModelID),
		strings.TrimSpace(primitive),
		debiasStr,
		NormalizeText(rubrics),
		strings.TrimSpace(criteriaJSON),
		NormalizeText(context),
		imageSig,
	)
	return HashString(payload)
}

// CalibrateChoiceFromLogits computes temperature-scaled Softmax probabilities over candidate choices.
func CalibrateChoiceFromLogits(rawLogits map[string]float64, choices []types.ChoiceOption, temperature float64) (winner string, confidence float64, probs map[string]float64) {
	if temperature <= 0 {
		temperature = 1.0
	}
	probs = make(map[string]float64, len(choices))
	if len(choices) == 0 {
		return "", 0, probs
	}

	maxScaled := math.Inf(-1)
	scaled := make([]float64, len(choices))
	for i, c := range choices {
		v, ok := rawLogits[c.Key]
		if !ok {
			v = -100.0
		}
		s := v / temperature
		scaled[i] = s
		if s > maxScaled {
			maxScaled = s
		}
	}

	sumExp := 0.0
	exps := make([]float64, len(choices))
	for i, s := range scaled {
		e := math.Exp(s - maxScaled)
		exps[i] = e
		sumExp += e
	}

	bestProb := -1.0
	for i, c := range choices {
		p := 0.0
		if sumExp > 0 {
			p = exps[i] / sumExp
		}
		probs[c.Key] = Round4(p)
		if p > bestProb {
			bestProb = p
			winner = c.Key
		}
	}
	confidence = Round4(bestProb)
	return winner, confidence, probs
}

// CalibrateScoreFromLogits computes temperature-scaled Softmax probabilities and expected score over ordinal levels.
func CalibrateScoreFromLogits(rawLogits map[string]float64, levels []string, temperature float64) (expectedScore, normScore float64, topLevel int, topLabel string, confidence float64, probs map[string]float64, legend map[string]string) {
	if temperature <= 0 {
		temperature = 1.0
	}
	mCount := len(levels)
	probs = make(map[string]float64, mCount)
	legend = make(map[string]string, mCount)
	if mCount == 0 {
		return 0, 0, 0, "", 0, probs, legend
	}

	maxScaled := math.Inf(-1)
	scaled := make([]float64, mCount)
	for m := 0; m < mCount; m++ {
		key := fmt.Sprintf("%d", m)
		legend[key] = levels[m]
		v, ok := rawLogits[key]
		if !ok {
			v = -100.0
		}
		s := v / temperature
		scaled[m] = s
		if s > maxScaled {
			maxScaled = s
		}
	}

	sumExp := 0.0
	exps := make([]float64, mCount)
	for m, s := range scaled {
		e := math.Exp(s - maxScaled)
		exps[m] = e
		sumExp += e
	}

	bestProb := -1.0
	expected := 0.0
	for m := 0; m < mCount; m++ {
		key := fmt.Sprintf("%d", m)
		p := 0.0
		if sumExp > 0 {
			p = exps[m] / sumExp
		}
		probs[key] = Round4(p)
		expected += float64(m) * p
		if p > bestProb {
			bestProb = p
			topLevel = m
			topLabel = levels[m]
		}
	}
	expectedScore = Round4(expected)
	if mCount > 1 {
		normScore = Round4(expected / float64(mCount-1))
	}
	confidence = Round4(bestProb)
	return expectedScore, normScore, topLevel, topLabel, confidence, probs, legend
}
// CalibrateFromLogitsWithAffine computes P(yes), confidence, and decision ("是"/"否") from raw logits
// using an affine Platt scaling transformation:
//   scaled = calibA * (logitYes - logitNo) + calibB
//   P(yes) = sigmoid(scaled)
func CalibrateFromLogitsWithAffine(logitYes, logitNo, calibA, calibB, threshold float64) (prob float64, confidence float64, passed bool, decision string) {
	if threshold <= 0 {
		threshold = 0.5
	}
	margin := logitYes - logitNo
	scaled := calibA*margin + calibB
	if scaled >= 0 {
		z := math.Exp(-scaled)
		prob = 1.0 / (1.0 + z)
	} else {
		z := math.Exp(scaled)
		prob = z / (1.0 + z)
	}
	passed = prob >= threshold
	if passed {
		decision = "是"
	} else {
		decision = "否"
	}
	if prob >= 1.0-prob {
		confidence = prob
	} else {
		confidence = 1.0 - prob
	}
	return prob, confidence, passed, decision
}

// CalibrateFromLogits computes P(yes), confidence, and decision ("是"/"否") from raw logits,
// temperature, and threshold (affine scaling with A = 1/temperature, B = 0).
func CalibrateFromLogits(logitYes, logitNo, temperature, threshold float64) (prob float64, confidence float64, passed bool, decision string) {
	if temperature <= 0 {
		temperature = 1.0
	}
	return CalibrateFromLogitsWithAffine(logitYes, logitNo, 1.0/temperature, 0.0, threshold)
}

// Round4 rounds a float64 to 4 decimal places for clean output.
func Round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}

// Store manages the SQLite decision_cache database using pure Go modernc.org/sqlite.
type Store struct {
	dbPath string
	db     *sql.DB
}

// Open initializes the SQLite database file and schema at dbPath.
func Open(dbPath string) (*Store, error) {
	resolved := config.ExpandPath(dbPath)
	if err := os.MkdirAll(filepath.Dir(resolved), 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}
	db, err := sql.Open("sqlite", resolved+"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)")
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite db: %w", err)
	}
	s := &Store{dbPath: resolved, db: db}
	if err := s.initSchema(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

func (s *Store) Path() string {
	return s.dbPath
}

func (s *Store) initSchema() error {
	schema := `
CREATE TABLE IF NOT EXISTS decision_cache (
    cache_key       TEXT PRIMARY KEY,
    model_id        TEXT NOT NULL,
    primitive       TEXT DEFAULT 'noul',
    criteria_json   TEXT DEFAULT '',
    raw_logits_json TEXT DEFAULT '',
    rubric_hash     TEXT NOT NULL,
    rubrics         TEXT NOT NULL,
    record_id       TEXT DEFAULT '',
    context_hash    TEXT NOT NULL,
    context         TEXT NOT NULL,
    logit_yes       REAL NOT NULL,
    logit_no        REAL NOT NULL,
    logit_margin    REAL NOT NULL,
    probability     REAL NOT NULL,
    latency_ms      REAL NOT NULL,
    created_at      TEXT NOT NULL,
    hit_count       INTEGER DEFAULT 0,
    last_accessed   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_rubric_hash ON decision_cache(rubric_hash);
CREATE INDEX IF NOT EXISTS idx_cache_model_id ON decision_cache(model_id);
CREATE INDEX IF NOT EXISTS idx_cache_record_id ON decision_cache(record_id);
`
	if _, err := s.db.Exec(schema); err != nil {
		return err
	}
	// Smoothly migrate existing v1 databases by adding columns if missing
	_, _ = s.db.Exec(`ALTER TABLE decision_cache ADD COLUMN primitive TEXT DEFAULT 'noul'`)
	_, _ = s.db.Exec(`ALTER TABLE decision_cache ADD COLUMN criteria_json TEXT DEFAULT ''`)
	_, _ = s.db.Exec(`ALTER TABLE decision_cache ADD COLUMN raw_logits_json TEXT DEFAULT ''`)
	return nil
}

// GetBatch retrieves cached entries for a slice of cacheKeys and increments their hit_count.
func (s *Store) GetBatch(keys []string) (map[string]types.CacheEntry, error) {
	result := make(map[string]types.CacheEntry, len(keys))
	if len(keys) == 0 {
		return result, nil
	}

	const chunkSize = 400
	now := time.Now().Format(time.RFC3339)
	for i := 0; i < len(keys); i += chunkSize {
		end := i + chunkSize
		if end > len(keys) {
			end = len(keys)
		}
		chunk := keys[i:end]
		placeholders := make([]string, len(chunk))
		args := make([]any, len(chunk))
		for idx, k := range chunk {
			placeholders[idx] = "?"
			args[idx] = k
		}
		query := fmt.Sprintf(`SELECT cache_key, model_id, COALESCE(primitive, 'noul'), COALESCE(criteria_json, ''), COALESCE(raw_logits_json, ''), rubric_hash, rubrics, record_id, context_hash, context, logit_yes, logit_no, logit_margin, probability, latency_ms, created_at, hit_count, last_accessed FROM decision_cache WHERE cache_key IN (%s)`, strings.Join(placeholders, ","))
		rows, err := s.db.Query(query, args...)
		if err != nil {
			return nil, err
		}
		var hitKeys []any
		var hitPlaceholders []string
		for rows.Next() {
			var r types.CacheEntry
			if err := rows.Scan(
				&r.CacheKey, &r.ModelID, &r.Primitive, &r.CriteriaJSON, &r.RawLogitsJSON,
				&r.RubricHash, &r.Rubrics, &r.RecordID,
				&r.ContextHash, &r.Context, &r.LogitYes, &r.LogitNo, &r.LogitMargin,
				&r.Probability, &r.LatencyMs, &r.CreatedAt, &r.HitCount, &r.LastAccessed,
			); err != nil {
				_ = rows.Close()
				return nil, err
			}
			result[r.CacheKey] = r
			hitKeys = append(hitKeys, r.CacheKey)
			hitPlaceholders = append(hitPlaceholders, "?")
		}
		_ = rows.Close()

		if len(hitKeys) > 0 {
			updArgs := append([]any{now}, hitKeys...)
			updSQL := fmt.Sprintf(`UPDATE decision_cache SET hit_count = hit_count + 1, last_accessed = ? WHERE cache_key IN (%s)`, strings.Join(hitPlaceholders, ","))
			_, _ = s.db.Exec(updSQL, updArgs...)
		}
	}
	return result, nil
}

// PutBatch inserts or updates a batch of CacheEntry records in a single SQLite transaction.
func (s *Store) PutBatch(entries []types.CacheEntry) error {
	if len(entries) == 0 {
		return nil
	}
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare(`INSERT INTO decision_cache (cache_key, model_id, primitive, criteria_json, raw_logits_json, rubric_hash, rubrics, record_id, context_hash, context, logit_yes, logit_no, logit_margin, probability, latency_ms, created_at, hit_count, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET primitive=excluded.primitive, criteria_json=excluded.criteria_json, raw_logits_json=excluded.raw_logits_json, record_id=excluded.record_id, logit_yes=excluded.logit_yes, logit_no=excluded.logit_no, logit_margin=excluded.logit_margin, probability=excluded.probability, latency_ms=excluded.latency_ms, last_accessed=excluded.last_accessed`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	defer stmt.Close()

	now := time.Now().Format(time.RFC3339)
	for _, e := range entries {
		prim := e.Primitive
		if prim == "" {
			prim = "noul"
		}
		createdAt := e.CreatedAt
		if createdAt == "" {
			createdAt = now
		}
		lastAccessed := e.LastAccessed
		if lastAccessed == "" {
			lastAccessed = now
		}
		if _, err := stmt.Exec(
			e.CacheKey, e.ModelID, prim, e.CriteriaJSON, e.RawLogitsJSON,
			e.RubricHash, e.Rubrics, e.RecordID,
			e.ContextHash, e.Context, e.LogitYes, e.LogitNo, e.LogitMargin,
			e.Probability, e.LatencyMs, createdAt, e.HitCount, lastAccessed,
		); err != nil {
			_ = tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

// CacheStats holds summary statistics for the SQLite cache.
type CacheStats struct {
	DBPath          string         `json:"db_path" yaml:"db_path"`
	TotalEntries    int            `json:"total_entries" yaml:"total_entries"`
	TotalHits       int            `json:"total_hits" yaml:"total_hits"`
	ByModel         map[string]int `json:"by_model" yaml:"by_model"`
	DistinctRubrics int            `json:"distinct_rubrics" yaml:"distinct_rubrics"`
}

// Stats returns aggregated statistics about the cache database.
func (s *Store) Stats() (*CacheStats, error) {
	stats := &CacheStats{
		DBPath:  s.dbPath,
		ByModel: map[string]int{},
	}
	row := s.db.QueryRow(`SELECT COUNT(*), COALESCE(SUM(hit_count), 0), COUNT(DISTINCT rubric_hash) FROM decision_cache`)
	if err := row.Scan(&stats.TotalEntries, &stats.TotalHits, &stats.DistinctRubrics); err != nil {
		return nil, err
	}

	rows, err := s.db.Query(`SELECT model_id, COUNT(*) FROM decision_cache GROUP BY model_id`)
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var m string
			var c int
			if rows.Scan(&m, &c) == nil {
				stats.ByModel[m] = c
			}
		}
	}
	return stats, nil
}

// QueryEntries returns cached entries optionally filtered by rubrics text or model_id.
func (s *Store) QueryEntries(rubricsFilter, modelFilter string) ([]types.CacheEntry, error) {
	var conds []string
	var args []any
	if rubricsFilter != "" {
		rHash := HashString(NormalizeText(rubricsFilter))
		conds = append(conds, `(rubric_hash = ? OR rubrics LIKE ?)`)
		args = append(args, rHash, "%"+rubricsFilter+"%")
	}
	if modelFilter != "" {
		conds = append(conds, `model_id LIKE ?`)
		args = append(args, "%"+modelFilter+"%")
	}
	where := ""
	if len(conds) > 0 {
		where = "WHERE " + strings.Join(conds, " AND ")
	}
	query := fmt.Sprintf(`SELECT cache_key, model_id, COALESCE(primitive, 'noul'), COALESCE(criteria_json, ''), COALESCE(raw_logits_json, ''), rubric_hash, rubrics, record_id, context_hash, context, logit_yes, logit_no, logit_margin, probability, latency_ms, created_at, hit_count, last_accessed FROM decision_cache %s ORDER BY created_at DESC`, where)
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []types.CacheEntry
	for rows.Next() {
		var r types.CacheEntry
		if err := rows.Scan(
			&r.CacheKey, &r.ModelID, &r.Primitive, &r.CriteriaJSON, &r.RawLogitsJSON,
			&r.RubricHash, &r.Rubrics, &r.RecordID,
			&r.ContextHash, &r.Context, &r.LogitYes, &r.LogitNo, &r.LogitMargin,
			&r.Probability, &r.LatencyMs, &r.CreatedAt, &r.HitCount, &r.LastAccessed,
		); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, nil
}

// Clear removes cache entries (all entries, or filtered by rubric_hash / model_id).
func (s *Store) Clear(rubricsFilter, modelFilter string) error {
	var conds []string
	var args []any
	if rubricsFilter != "" {
		rHash := HashString(NormalizeText(rubricsFilter))
		conds = append(conds, `rubric_hash = ?`)
		args = append(args, rHash)
	}
	if modelFilter != "" {
		conds = append(conds, `model_id LIKE ?`)
		args = append(args, "%"+modelFilter+"%")
	}
	where := ""
	if len(conds) > 0 {
		where = "WHERE " + strings.Join(conds, " AND ")
	}
	_, err := s.db.Exec(fmt.Sprintf("DELETE FROM decision_cache %s", where), args...)
	return err
}
