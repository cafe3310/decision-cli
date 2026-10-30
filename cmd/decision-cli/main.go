package main

import (
	"fmt"
	"os"
	"runtime"

	"github.com/cafe3310/decision-cli/internal/cli"
	"github.com/cafe3310/decision-cli/internal/statusbar"
)

func init() {
	// Ensure the main goroutine stays on OS Thread 0, mandatory for macOS Cocoa AppKit / NSStatusItem
	runtime.LockOSThread()
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == "bar" {
		_ = statusbar.Run()
		return
	}

	rootCmd := cli.NewRootCommand()
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
