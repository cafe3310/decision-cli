package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCORSMiddleware(t *testing.T) {
	mockHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	handler := corsMiddleware(mockHandler)

	// 1. 测试 OPTIONS 预检请求
	reqOpt := httptest.NewRequest(http.MethodOptions, "/v1/noun", nil)
	recOpt := httptest.NewRecorder()
	handler.ServeHTTP(recOpt, reqOpt)

	if recOpt.Code != http.StatusNoContent {
		t.Fatalf("预检请求预期状态码 204，实际为: %d", recOpt.Code)
	}
	if origin := recOpt.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Errorf("预期 Allow-Origin 为 *，实际为: %s", origin)
	}
	if methods := recOpt.Header().Get("Access-Control-Allow-Methods"); methods == "" {
		t.Errorf("预期返回 Allow-Methods，实际为空")
	}

	// 2. 测试普通 GET 请求带上 CORS 标头
	reqGet := httptest.NewRequest(http.MethodGet, "/health", nil)
	recGet := httptest.NewRecorder()
	handler.ServeHTTP(recGet, reqGet)

	if recGet.Code != http.StatusOK {
		t.Fatalf("GET 预期状态码 200，实际为: %d", recGet.Code)
	}
	if origin := recGet.Header().Get("Access-Control-Allow-Origin"); origin != "*" {
		t.Errorf("预期 Allow-Origin 为 *，实际为: %s", origin)
	}
}
