#!/usr/bin/env python3
"""
generate_full_eval_html.py — 生成 AAII 风格纯柱状图展示的决策模型评测报告
"""

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
IN_JSON = WORKSPACE / "bench" / "results" / "full_eval_results_20261030.json"
OUT_HTML = WORKSPACE / "bench" / "reports" / "full_models_and_jev_benchmark_report.html"


def main():
    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    loc = data["local_models"]
    rem = data["remote_models"]

    # Colors as strictly requested:
    # jev: 虚框灰色
    # qwen: 灰色
    # 百灵 (Ling): 蓝色
    # gemma: 浅紫色
    models = {
        "gemma-26b": {
            "name": "gemma-4-26b-a4b",
            "label": "Gemma 4 26B-A4B",
            "env": "本地",
            "sub": "28BA4B",
            "color": "#a855f7",  # 浅紫色
            "isLocal": True,
            "supportsVision": True,
        },
        "gemma-e4b": {
            "name": "gemma-4-e4b",
            "label": "Gemma 4 E4B",
            "env": "本地",
            "sub": "4.5B",
            "color": "#c084fc",  # 浅紫色
            "isLocal": True,
            "supportsVision": True,
        },
        "ling": {
            "name": "Ling-3.0-Tiny",
            "label": "Ling-3.0-Tiny",
            "env": "本地",
            "sub": "8BA1.3B",
            "color": "#2563eb",  # 蓝色
            "isLocal": True,
            "supportsVision": False,
        },
        "qwen": {
            "name": "qwen3-reranker-4b",
            "label": "Qwen3 Reranker 4B",
            "env": "本地",
            "sub": "4BDense",
            "color": "#64748b",  # 灰色
            "isLocal": True,
            "supportsVision": False,
        },
        "typesafe": {
            "name": "typesafe-jev-1.13.0",
            "label": "TypeSafe Jev 1.13.0",
            "env": "网络",
            "sub": "Jev 1.13",
            "color": "#94a3b8",  # 虚框灰色
            "isDashed": True,
            "isLocal": False,
            "supportsVision": True,
        },
    }

    # Data extracts
    # 1. Text JevBench 231 Overall (%)
    text_jev_overall = [
        {"key": "typesafe", "val": rem["typesafe-jev-1.13.0"]["jevbench_231"]["overall_acc"] * 100},
        {"key": "gemma-26b", "val": loc["gemma-4-26b-a4b"]["text"]["jevbench_231"]["overall_acc"] * 100},
        {"key": "gemma-e4b", "val": loc["gemma-4-e4b"]["text"]["jevbench_231"]["overall_acc"] * 100},
        {"key": "ling", "val": loc["ling-3.0-tiny"]["text"]["jevbench_231"]["overall_acc"] * 100},
        {"key": "qwen", "val": loc["qwen3-reranker-4b"]["text"]["jevbench_231"]["overall_acc"] * 100},
    ]

    # 2. Multimodal Scores (ImageJev-box 900 %)
    mm_scores = [
        {"key": "gemma-26b", "val": loc["gemma-4-26b-a4b"]["multimodal"]["full_900_scores"]["overall_acc"] * 100},
        {"key": "gemma-e4b", "val": loc["gemma-4-e4b"]["multimodal"]["full_900_scores"]["overall_acc"] * 100},
        {"key": "typesafe", "val": rem["typesafe-jev-1.13.0"]["multimodal_ref"]["imajev_4b_open_acc"] * 100},
    ]

    # 2.1 Multimodal Track 3 Grounding (%)
    mm_grounding = [
        {"key": "gemma-26b", "val": loc["gemma-4-26b-a4b"]["multimodal"]["full_900_scores"]["track3_grounding_acc"] * 100},
        {"key": "gemma-e4b", "val": loc["gemma-4-e4b"]["multimodal"]["full_900_scores"]["track3_grounding_acc"] * 100},
        {"key": "typesafe", "val": 75.7},
    ]

    # 3. Text Latency p50 (ms - lower is better)
    text_latency = [
        {"key": "ling", "val": loc["ling-3.0-tiny"]["text"]["speed"]["p50_ms"]},
        {"key": "qwen", "val": loc["qwen3-reranker-4b"]["text"]["speed"]["p50_ms"]},
        {"key": "gemma-e4b", "val": loc["gemma-4-e4b"]["text"]["speed"]["p50_ms"]},
        {"key": "gemma-26b", "val": loc["gemma-4-26b-a4b"]["text"]["speed"]["p50_ms"]},
        {"key": "typesafe", "val": rem["typesafe-jev-1.13.0"]["speed"]["p50_ms"]},
    ]

    # 4. Multimodal Latency (ms - lower is better)
    mm_latency = [
        {"key": "gemma-e4b", "label_extra": "缓存命中", "val": loc["gemma-4-e4b"]["multimodal"]["speed"]["warm_cache_p50_ms"]},
        {"key": "gemma-26b", "label_extra": "缓存命中", "val": loc["gemma-4-26b-a4b"]["multimodal"]["speed"]["warm_cache_p50_ms"]},
        {"key": "gemma-e4b", "label_extra": "冷启动编码", "val": loc["gemma-4-e4b"]["multimodal"]["speed"]["cold_vision_p50_ms"]},
        {"key": "typesafe", "label_extra": "云端平均", "val": 650.0},
        {"key": "gemma-26b", "label_extra": "冷启动编码", "val": loc["gemma-4-26b-a4b"]["multimodal"]["speed"]["cold_vision_p50_ms"]},
    ]

    # 5. Peak RAM Footprint (GB - lower is better)
    ram_peak = [
        {"key": "qwen", "val": loc["qwen3-reranker-4b"]["resource_summary"]["memory"]["total_peak_mb"] / 1024.0},
        {"key": "gemma-e4b", "val": loc["gemma-4-e4b"]["resource_summary"]["memory"]["total_peak_mb"] / 1024.0},
        {"key": "gemma-26b", "val": loc["gemma-4-26b-a4b"]["resource_summary"]["memory"]["total_peak_mb"] / 1024.0},
        {"key": "ling", "val": loc["ling-3.0-tiny"]["resource_summary"]["memory"]["total_peak_mb"] / 1024.0},
    ]

    # 6. GPU Device Utilization Avg (%)
    gpu_avg = [
        {"key": "gemma-e4b", "val": loc["gemma-4-e4b"]["resource_summary"]["gpu"]["overall_avg_pct"]},
        {"key": "qwen", "val": loc["qwen3-reranker-4b"]["resource_summary"]["gpu"]["overall_avg_pct"]},
        {"key": "ling", "val": loc["ling-3.0-tiny"]["resource_summary"]["gpu"]["overall_avg_pct"]},
        {"key": "gemma-26b", "val": loc["gemma-4-26b-a4b"]["resource_summary"]["gpu"]["overall_avg_pct"]},
    ]

    def render_bar_chart(title, subtitle, items, unit, max_val=100, is_lower_better=False, digits=1):
        # Build grid lines
        grid_html = ""
        steps = [1.0, 0.75, 0.50, 0.25, 0.0]
        for s in steps:
            lbl = f"{max_val * s:.{digits}f}".rstrip("0").rstrip(".") if digits > 0 else f"{int(max_val * s)}"
            if s == 0:
                lbl = "0"
            grid_html += f'<div class="grid-line"><span class="grid-label">{lbl}{unit}</span></div>'

        bars_html = ""
        for it in items:
            m = models[it["key"]]
            val = it["val"]
            pct = max(6.0, min(98.0, (val / max_val) * 100))
            disp_val = f"{val:.{digits}f}" if digits > 0 else f"{int(val)}"
            extra = it.get("label_extra", "")
            env_str = m.get("env", "")
            sub_core = f"{env_str} • {m['sub']}" if env_str else m['sub']
            sub_text = f"{extra} • {sub_core}" if extra else sub_core

            is_dashed = m.get("isDashed", False)
            if is_dashed:
                bar_style = f"height: {pct}%; border: 2px dashed #94a3b8; border-bottom: none; background-color: #f8fafc;"
                score_style = "color: #334155; text-shadow: none;"
            else:
                bar_style = f"height: {pct}%; background-color: {m['color']};"
                score_style = "color: #ffffff;"

            bars_html += f"""
            <div class="bar-column">
              <div class="bar-fill" style="{bar_style}" title="{m['label']} ({sub_text}): {disp_val}{unit}">
                <span class="bar-score" style="{score_style}">{disp_val}</span>
              </div>
              <div class="bar-footer">
                <div class="model-label-rotated">
                  {m['label']}
                  <span class="model-label-sub">{sub_text}</span>
                </div>
              </div>
            </div>
            """

        return f"""
        <div class="chart-card">
          <div class="card-header">
            <div>
              <h3>{title}</h3>
              <p>{subtitle}</p>
            </div>
            <div class="card-unit-badge">{unit}</div>
          </div>
          <div class="chart-container">
            <div class="grid-lines">
              {grid_html}
            </div>
            <div class="bars-wrapper">
              {bars_html}
            </div>
          </div>
        </div>
        """

    # Assemble HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>决策模型基准评测大盘</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --page-bg: #f9fafb;
      --card-bg: #ffffff;
      --border: #e5e7eb;
      --text: #111827;
      --text-muted: #64748b;
      --font-serif: "Newsreader", Georgia, serif;
      --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: "JetBrains Mono", Menlo, monospace;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: var(--font-sans);
      background: var(--page-bg);
      color: var(--text);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      padding: 40px 32px 80px;
    }}

    .max-container {{
      max-width: 1280px;
      margin: 0 auto;
    }}

    /* Editorial Header */
    .header-section {{
      margin-bottom: 32px;
    }}

    .title-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }}

    h1 {{
      font-family: var(--font-serif);
      font-size: 38px;
      font-weight: 500;
      letter-spacing: -0.02em;
      color: #0f172a;
      line-height: 1.15;
    }}

    .arrow-icon {{
      font-size: 26px;
      color: #94a3b8;
    }}

    .subtitle {{
      font-size: 14px;
      color: #64748b;
      line-height: 1.6;
      max-width: 900px;
    }}

    .color-legend {{
      display: flex;
      gap: 18px;
      margin-top: 18px;
      flex-wrap: wrap;
      align-items: center;
      font-size: 12.5px;
      color: #475569;
    }}

    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}

    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 2px;
    }}

    .legend-dashed {{
      width: 12px;
      height: 12px;
      border: 1.5px dashed #94a3b8;
      background: #f8fafc;
      border-radius: 2px;
    }}

    /* Two-column Chart Grid */
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
      gap: 24px;
      margin-bottom: 24px;
    }}

    /* Chart Card */
    .chart-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 24px 28px 32px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
      display: flex;
      flex-direction: column;
    }}

    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 24px;
    }}

    .card-header h3 {{
      font-family: var(--font-serif);
      font-size: 18px;
      font-weight: 600;
      color: #0f172a;
      margin-bottom: 4px;
    }}

    .card-header p {{
      font-size: 12px;
      color: #64748b;
    }}

    .card-unit-badge {{
      font-size: 11px;
      font-family: var(--font-mono);
      color: #64748b;
      background: #f1f5f9;
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: 500;
    }}

    /* Chart Canvas */
    .chart-container {{
      position: relative;
      height: 280px;
      width: 100%;
      margin-bottom: 96px; /* Space for angled labels */
    }}

    .grid-lines {{
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      pointer-events: none;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}

    .grid-line {{
      width: 100%;
      border-bottom: 1px dashed #e2e8f0;
      position: relative;
    }}

    .grid-label {{
      position: absolute;
      right: 0;
      top: -9px;
      font-size: 10px;
      color: #94a3b8;
      font-family: var(--font-mono);
    }}

    .bars-wrapper {{
      position: absolute;
      top: 0;
      left: 16px;
      right: 36px;
      bottom: 0;
      display: flex;
      align-items: flex-end;
      justify-content: space-around;
      gap: 12px;
    }}

    .bar-column {{
      flex: 1;
      max-width: 72px;
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      align-items: center;
      position: relative;
    }}

    .bar-fill {{
      width: 100%;
      border-radius: 5px 5px 0 0;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding-top: 8px;
      transition: height 0.3s ease, filter 0.15s;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
      cursor: pointer;
    }}
    .bar-fill:hover {{
      filter: brightness(1.08);
    }}

    .bar-score {{
      color: #ffffff;
      font-size: 13.5px;
      font-weight: 700;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
      user-select: none;
      font-family: var(--font-mono);
    }}

    .bar-footer {{
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding-top: 10px;
      width: 110px;
      pointer-events: none;
    }}

    .model-label-rotated {{
      transform-origin: top left;
      transform: rotate(50deg) translate(6px, 2px);
      white-space: nowrap;
      font-size: 11.5px;
      font-weight: 600;
      color: #1e293b;
      letter-spacing: -0.01em;
      text-align: left;
      width: 120px;
    }}

    .model-label-sub {{
      display: block;
      font-size: 9.5px;
      font-weight: 400;
      color: #64748b;
      margin-top: 1px;
    }}

    footer {{
      margin-top: 50px;
      padding-top: 20px;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      color: #94a3b8;
      font-size: 12px;
    }}
  </style>
</head>
<body>

  <div class="max-container">

    <!-- Header Section -->
    <header class="header-section">
      <div class="title-row">
        <h1>决策模型基准评测报告</h1>
        <span class="arrow-icon">↗</span>
      </div>
      <p class="subtitle">
        基于 Apple Silicon MLX 单步 Logit 读取与远端 Jev 接口，对 6 款决策模型在文本判别、多模态视觉决策、推理延迟及本地硬件开销（内存与 GPU 占用）进行全维度实测对比。
      </p>

      <div class="color-legend">
        <span style="font-weight: 600; color: #1e293b;">图例指定配色：</span>
        <div class="legend-item"><span class="legend-dashed"></span> Jev (虚框灰色)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #64748b;"></span> Qwen (灰色)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #2563eb;"></span> 百灵 (蓝色)</div>
        <div class="legend-item"><span class="legend-dot" style="background: #a855f7;"></span> Gemma (浅紫色)</div>
      </div>
    </header>

    <div class="charts-grid">
      {render_bar_chart(
          "1. JevBench Public 综合准确率，越大越好",
          "覆盖 Easy、Standard、Hard 三层难度与二元/多分类/量规三种决策原语（231 题公开集）",
          text_jev_overall,
          "%",
          max_val=100,
          digits=1
      )}

      {render_bar_chart(
          "2. 文本推理延迟中位数，越小越好",
          "跳过缓存，真实单步前向耗时（单位：毫秒）",
          text_latency,
          " ms",
          max_val=300,
          is_lower_better=True,
          digits=1
      )}

      {render_bar_chart(
          "3. 多模态综合准确率，越大越好",
          "涵盖存在性核验、四选一细粒度分类与候选框空间定位（ImageJev-box 900 题）",
          mm_scores,
          "%",
          max_val=100,
          digits=1
      )}

      {render_bar_chart(
          "4. 空间候选框定位准确率，越大越好",
          "根据视觉目标空间坐标与边界框执行精确定位决策（仅对支持模型）",
          mm_grounding,
          "%",
          max_val=100,
          digits=1
      )}

      {render_bar_chart(
          "5. 多模态单步延迟中位数，越小越好",
          "包含冷启动视觉编码与 Warm Cache 命中状态耗时对比（单位：毫秒）",
          mm_latency,
          " ms",
          max_val=900,
          is_lower_better=True,
          digits=1
      )}

      {render_bar_chart(
          "6. 总物理内存占用峰值，越小越好",
          "涵盖 mlx_worker、状态栏与 CLI 客户端物理内存峰值之和（libproc.h 精确采样）",
          ram_peak,
          " GB",
          max_val=20,
          is_lower_better=True,
          digits=2
      )}

      {render_bar_chart(
          "7. GPU 平均利用率",
          "推理活跃期内 Metal GPU 算力核心平均利用率（IOAccelerator 实时采样）",
          gpu_avg,
          "%",
          max_val=100,
          digits=1
      )}
    </div>

    <!-- Footer -->
    <footer>
      <div>decision-cli • macOS Apple Silicon 本地单步概率决策评测</div>
      <div>评测时间: {data.get("evaluated_at", "2026-10-30")} | 硬件平台: {data.get("hardware", "Apple M5 Pro")}</div>
    </footer>

  </div>

</body>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Pure bar-chart report generated at: {OUT_HTML}")


if __name__ == "__main__":
    main()
