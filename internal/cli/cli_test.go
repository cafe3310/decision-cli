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

func TestModelsDownloadSubcommand(t *testing.T) {
	rootCmd := NewRootCommand()
	cmd, _, err := rootCmd.Find([]string{"models", "download"})
	if err != nil || cmd == nil {
		t.Fatalf("查找 models download 命令失败: %v", err)
	}
	if cmd.Name() != "download" {
		t.Errorf("预期子命令为 download，实际为: %s", cmd.Name())
	}
}

func TestValidationActionableErrors(t *testing.T) {
	rootCmd := NewRootCommand()

	testCases := []struct {
		name string
		args []string
	}{
		{"noun_missing_rubric", []string{"noun"}},
		{"noun_missing_context", []string{"noun", "--rubrics", "测试"}},
		{"choose_missing_choices", []string{"choose", "--rubrics", "测试", "--context", "测试"}},
		{"score_missing_levels", []string{"score", "--rubrics", "测试", "--context", "测试"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			buf := new(bytes.Buffer)
			rootCmd.SetOut(buf)
			rootCmd.SetErr(buf)
			rootCmd.SetArgs(tc.args)
			err := rootCmd.Execute()
			if err == nil {
				t.Fatalf("用例 %s 预期执行失败但成功", tc.name)
			}
			errStr := err.Error()
			if !bytes.Contains([]byte(errStr), []byte("建议操作")) && !bytes.Contains([]byte(errStr), []byte("正确示例")) {
				t.Errorf("用例 %s 报错缺少下一步行动建议: %s", tc.name, errStr)
			}
		})
	}
}
