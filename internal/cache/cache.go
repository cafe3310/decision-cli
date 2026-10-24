package cache

import (
	"crypto/sha256"
	"database/sql"
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
	payload := fmt.Sprintf("%s|%s|%s|%s",
		CacheKeySchemaVersion,
		strings.TrimSpace(canonicalModelID),
		NormalizeText(rubrics),
		NormalizeText(context),
	)
	return HashString(payload)
}

// CalibrateFromLogits computes P(yes), confidence, and decision ("是"/"否") from raw logits,
// temperature, and threshold.
func CalibrateFromLogits(logitYes, logitNo, temperature, threshold float64) (prob float64, confidence float64, passed bool, decision string) {
	if temperature <= 0 {
		temperature = 1.0
	}
	if threshold <= 0 {
		threshold = 0.5
	}
	margin := (logitYes - logitNo) / temperature
	if margin >= 0 {
		z := math.Exp(-margin)
		prob = 1.0 / (1.0 + z)
	} else {
		z := math.Exp(margin)
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
	_, err := s.db.Exec(schema)
	return err
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
		query := fmt.Sprintf(`SELECT cache_key, model_id, rubric_hash, rubrics, record_id, context_hash, context, logit_yes, logit_no, logit_margin, probability, latency_ms, created_at, hit_count, last_accessed FROM decision_cache WHERE cache_key IN (%s)`, strings.Join(placeholders, ","))
		rows, err := s.db.Query(query, args...)
		if err != nil {
			return nil, err
		}
		var hitKeys []any
		var hitPlaceholders []string
		for rows.Next() {
			var r types.CacheEntry
			if err := rows.Scan(
				&r.CacheKey, &r.ModelID, &r.RubricHash, &r.Rubrics, &r.RecordID,
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
	stmt, err := tx.Prepare(`INSERT INTO decision_cache (cache_key, model_id, rubric_hash, rubrics, record_id, context_hash, context, logit_yes, logit_no, logit_margin, probability, latency_ms, created_at, hit_count, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET record_id=excluded.record_id, logit_yes=excluded.logit_yes, logit_no=excluded.logit_no, logit_margin=excluded.logit_margin, probability=excluded.probability, latency_ms=excluded.latency_ms, last_accessed=excluded.last_accessed`)
	if err != nil {
		_ = tx.Rollback()
		return err
	}
	defer stmt.Close()

	now := time.Now().Format(time.RFC3339)
	for _, e := range entries {
		createdAt := e.CreatedAt
		if createdAt == "" {
			createdAt = now
		}
		lastAccessed := e.LastAccessed
		if lastAccessed == "" {
			lastAccessed = now
		}
		if _, err := stmt.Exec(
			e.CacheKey, e.ModelID, e.RubricHash, e.Rubrics, e.RecordID,
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
	query := fmt.Sprintf(`SELECT cache_key, model_id, rubric_hash, rubrics, record_id, context_hash, context, logit_yes, logit_no, logit_margin, probability, latency_ms, created_at, hit_count, last_accessed FROM decision_cache %s ORDER BY created_at DESC`, where)
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []types.CacheEntry
	for rows.Next() {
		var r types.CacheEntry
		if err := rows.Scan(
			&r.CacheKey, &r.ModelID, &r.RubricHash, &r.Rubrics, &r.RecordID,
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
