//go:build !darwin

package statusbar

import "fmt"

// Run is a stub on non-macOS systems.
func Run() error {
	return fmt.Errorf("系统状态栏图标目前仅支持 macOS (Cocoa) 环境")
}

// IsBarRunning returns false on non-macOS systems.
func IsBarRunning() bool {
	return false
}
