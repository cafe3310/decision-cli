package cli

import (
	"bytes"
	"testing"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

func TestNoShorthandFlagsAcrossAllCommands(t *testing.T) {
	rootCmd := NewRootCommand()

	var checkCmd func(cmd *cobra.Command)
	checkCmd = func(cmd *cobra.Command) {
		cmd.Flags().VisitAll(func(f *pflag.Flag) {
			if f.Name == "help" {
				return // pflag 内部自带的 -h 允许保留
			}
			if f.Shorthand != "" {
				t.Errorf("命令 %q 的参数 --%s 包含缩写 -%s，违反全长命名参数契约", cmd.CommandPath(), f.Name, f.Shorthand)
			}
		})
		for _, sub := range cmd.Commands() {
			checkCmd(sub)
		}
	}

	checkCmd(rootCmd)
}

func TestSubcommandHierarchy(t *testing.T) {
	rootCmd := NewRootCommand()

	expectedRootSubs := []string{"noun", "score", "choose", "batch", "systemone", "serve", "kill", "doctor", "models", "cache"}
	foundSubs := make(map[string]bool)
	for _, c := range rootCmd.Commands() {
		foundSubs[c.Name()] = true
	}

	for _, exp := range expectedRootSubs {
		if !foundSubs[exp] {
			t.Errorf("缺少顶层子命令: %s", exp)
		}
	}

	// 检查 batch 子命令
	batchCmd, _, err := rootCmd.Find([]string{"batch"})
	if err != nil || batchCmd == nil {
		t.Fatalf("查找 batch 命令失败: %v", err)
	}
	expectedBatchSubs := []string{"noun", "score", "choose"}
	foundBatchSubs := make(map[string]bool)
	for _, c := range batchCmd.Commands() {
		foundBatchSubs[c.Name()] = true
	}
	for _, exp := range expectedBatchSubs {
		if !foundBatchSubs[exp] {
			t.Errorf("缺少 batch 下级子命令: %s", exp)
		}
	}
}

func TestNounValidation(t *testing.T) {
	rootCmd := NewRootCommand()
	buf := new(bytes.Buffer)
	rootCmd.SetOut(buf)
	rootCmd.SetErr(buf)
	rootCmd.SetArgs([]string{"noun"})

	err := rootCmd.Execute()
	if err == nil {
		t.Fatalf("期望缺少 --rubrics 时报错，但执行成功")
	}
}
