package infra

import (
	"bufio"
	"bytes"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cafe3310/decision-cli/internal/config"
	"github.com/cafe3310/decision-cli/internal/types"
)

// MLXProcessInfo 描述一个被识别为与 decision-cli 相关的 MLX 进程
type MLXProcessInfo struct {
	PID     int
	RSSKB   int
	Command string
}

// MemoryMB 返回友好的内存大小展示
func (p MLXProcessInfo) MemoryDisplay() string {
	if p.RSSKB <= 0 {
		return "未知"
	}
	if p.RSSKB > 1024*1024 {
		return fmt.Sprintf("%.1f GB", float64(p.RSSKB)/(1024*1024))
	}
	return fmt.Sprintf("%.1f MB", float64(p.RSSKB)/1024)
}

// FindRelatedMLXProcesses 扫描系统进程，找出所有与 decision-cli 相关的 MLX 推理进程
func FindRelatedMLXProcesses(cfg *types.AppConfig) ([]MLXProcessInfo, error) {
	cmd := exec.Command("ps", "-eo", "pid,rss,command")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("执行 ps 命令失败: %w", err)
	}

	myPID := os.Getpid()
	var matches []MLXProcessInfo

	scanner := bufio.NewScanner(&out)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "PID") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) < 3 {
			continue
		}

		pid, err := strconv.Atoi(parts[0])
		if err != nil || pid == myPID {
			continue
		}

		rssKB, _ := strconv.Atoi(parts[1])
		fullCmd := strings.Join(parts[2:], " ")

		// 排除自身相关命令或调试进程
		if strings.Contains(fullCmd, "decision-cli kill") ||
			strings.Contains(fullCmd, "grep") {
			continue
		}

		// 判定是否与 decision-cli 及其 MLX Worker 相关
		isRelated := false
		if strings.Contains(fullCmd, "mlx_worker.py") {
			isRelated = true
		} else if strings.Contains(fullCmd, "worker.sock") && strings.Contains(fullCmd, "decision-cli") {
			isRelated = true
		} else if strings.Contains(fullCmd, "mlx") && strings.Contains(fullCmd, "decision-cli") {
			isRelated = true
		}

		if isRelated {
			matches = append(matches, MLXProcessInfo{
				PID:     pid,
				RSSKB:   rssKB,
				Command: fullCmd,
			})
		}
	}

	return matches, nil
}

// KillRelatedMLXProcesses 终止指定的关联进程，并清理 socket 文件
func KillRelatedMLXProcesses(cfg *types.AppConfig, procs []MLXProcessInfo, force bool) ([]string, error) {
	var logs []string

	// 1. 如果当前活跃的 socket 存在，先尝试发送优雅退出
	sockPath := config.ExpandPath(cfg.Runtime.WorkerSocket)
	if sockPath == "" {
		sockPath = config.ExpandPath("~/.cache/decision-cli/worker.sock")
	}
	if conn, err := net.DialTimeout("unix", sockPath, 200*time.Millisecond); err == nil {
		_ = conn.SetDeadline(time.Now().Add(500 * time.Millisecond))
		_, _ = conn.Write([]byte(`{"action":"shutdown"}` + "\n"))
		conn.Close()
	}

	// 2. 终止每一个匹配的进程
	for _, p := range procs {
		if !isProcessAlive(p.PID) {
			logs = append(logs, fmt.Sprintf("进程 [PID: %d] 已自行退出", p.PID))
			continue
		}

		if force {
			if err := syscall.Kill(p.PID, syscall.SIGKILL); err != nil {
				logs = append(logs, fmt.Sprintf("强制终止 [PID: %d] 失败: %v", p.PID, err))
			} else {
				logs = append(logs, fmt.Sprintf("已强制终止 [PID: %d] (SIGKILL)", p.PID))
			}
			continue
		}

		// 先尝试 SIGTERM
		_ = syscall.Kill(p.PID, syscall.SIGTERM)
		// 等待最多 500ms
		terminated := false
		for i := 0; i < 5; i++ {
			time.Sleep(100 * time.Millisecond)
			if !isProcessAlive(p.PID) {
				terminated = true
				break
			}
		}

		if terminated {
			logs = append(logs, fmt.Sprintf("已终止 [PID: %d] (SIGTERM 优雅退出)", p.PID))
		} else {
			// 超时则升级为 SIGKILL
			if err := syscall.Kill(p.PID, syscall.SIGKILL); err != nil {
				logs = append(logs, fmt.Sprintf("强制终止 [PID: %d] 失败: %v", p.PID, err))
			} else {
				logs = append(logs, fmt.Sprintf("已强制终止 [PID: %d] (SIGKILL 强杀)", p.PID))
			}
		}
	}

	// 3. 清理残留的 socket 文件
	if _, err := os.Stat(sockPath); err == nil {
		if err := os.Remove(sockPath); err == nil {
			logs = append(logs, fmt.Sprintf("已清理残留 Socket: %s", sockPath))
		} else {
			logs = append(logs, fmt.Sprintf("清理 Socket 失败: %v", err))
		}
	}

	// 4. 清理可能的临时 sock
	globPattern := config.ExpandPath("~/.cache/decision-cli/*.sock")
	if files, err := filepath.Glob(globPattern); err == nil {
		for _, f := range files {
			_ = os.Remove(f)
		}
	}

	return logs, nil
}

func isProcessAlive(pid int) bool {
	// 发送信号 0，若无错误则说明进程存活
	err := syscall.Kill(pid, 0)
	return err == nil
}
