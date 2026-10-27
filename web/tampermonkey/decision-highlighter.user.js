// ==UserScript==
// @name         decision-cli 网页内容判定助手
// @namespace    https://github.com/cafe3310/decision-cli
// @version      0.3.0
// @description  在任意网页上划选区域，基于本地 decision-cli 决策模型（Gemma-4 等）对文本进行毫秒级单步概率判定。支持收起/展开、连续自动判定与批量拖拽标注一键判定。
// @author       cafe3310
// @match        *://*/*
// @grant        GM_xmlhttpRequest
// @grant        GM_setValue
// @grant        GM_getValue
// @connect      127.0.0.1
// @connect      localhost
// @run-at       document-end
// ==/UserScript==

(function () {
  'use strict';

  // 避免在 iframe 中重复挂载
  if (window.top !== window.self) return;

  // -------------------------------------------------------------
  // 1. 常量与配置持久化
  // -------------------------------------------------------------
  const STORAGE_KEY_API_URL = 'dc_api_url';
  const STORAGE_KEY_RUBRIC = 'dc_current_rubric';
  const STORAGE_KEY_COLLAPSED = 'dc_is_collapsed';
  const STORAGE_KEY_AUTO_INFER = 'dc_auto_infer';

  const PRESET_RUBRICS = [
    '内容看起来似乎是广告',
    '内容看起来像是软广告或推广',
    '内容看起来具备典型的 ai 生成特征'
  ];

  function getStorage(key, def) {
    let val;
    if (typeof GM_getValue === 'function') {
      val = GM_getValue(key, def);
    } else {
      val = localStorage.getItem(key);
      if (val === null) val = def;
    }
    if (val === 'true') return true;
    if (val === 'false') return false;
    return val;
  }

  function setStorage(key, val) {
    if (typeof GM_setValue === 'function') {
      GM_setValue(key, val);
    } else {
      localStorage.setItem(key, String(val));
    }
  }

  let serverApiUrl = getStorage(STORAGE_KEY_API_URL, 'http://127.0.0.1:8080');
  let currentRubric = getStorage(STORAGE_KEY_RUBRIC, PRESET_RUBRICS[0]);
  let isDockCollapsed = getStorage(STORAGE_KEY_COLLAPSED, false);
  let autoInfer = getStorage(STORAGE_KEY_AUTO_INFER, true);

  let isSelecting = false;
  let selectionMode = 'batch'; // 'batch' (批量拖拽标注) 或 'auto' (划选即自动推理)
  let activeHighlights = [];  // 存储页面上所有高亮区域对象
  let highlightCounter = 0;   // 高亮自增序号

  // -------------------------------------------------------------
  // 2. 注入全局隔离样式
  // -------------------------------------------------------------
  const styleEl = document.createElement('style');
  styleEl.id = 'decision-cli-styles';
  styleEl.textContent = `
    /* 主容器与浮窗 */
    #decision-cli-dock {
      position: fixed;
      right: 24px;
      bottom: 24px;
      width: 340px;
      background: #ffffff;
      color: #1e293b;
      border-radius: 14px;
      box-shadow: 0 12px 30px -4px rgba(0, 0, 0, 0.18), 0 4px 12px -2px rgba(0, 0, 0, 0.08);
      border: 1px solid #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      font-size: 13px;
      line-height: 1.5;
      z-index: 2147483640;
      box-sizing: border-box;
      user-select: none;
      transition: width 0.22s cubic-bezier(0.16, 1, 0.3, 1), 
                  height 0.22s cubic-bezier(0.16, 1, 0.3, 1), 
                  box-shadow 0.2s, 
                  border-radius 0.2s;
    }
    #decision-cli-dock * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    /* 收起状态的极简药丸徽章 */
    #decision-cli-dock.dc-collapsed {
      width: auto !important;
      border-radius: 9999px;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
      border: 1px solid #cbd5e1;
    }
    #decision-cli-dock.dc-collapsed .dc-dock-header,
    #decision-cli-dock.dc-collapsed .dc-dock-body {
      display: none !important;
    }
    .dc-mini-pill {
      display: none;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      cursor: pointer;
      font-size: 12.5px;
      font-weight: 600;
      color: #0f172a;
    }
    #decision-cli-dock.dc-collapsed .dc-mini-pill {
      display: flex;
    }
    .dc-mini-pill:hover {
      background: #f8fafc;
      border-radius: 9999px;
    }
    .dc-mini-tag {
      background: #eff6ff;
      color: #2563eb;
      padding: 1px 7px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 500;
      max-width: 100px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    /* 展开状态顶部标题栏 */
    .dc-dock-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px;
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
      border-radius: 14px 14px 0 0;
      cursor: grab;
    }
    .dc-dock-title {
      font-weight: 600;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 6px;
      color: #0f172a;
    }
    .dc-status-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #94a3b8;
    }
    .dc-status-dot.dc-online {
      background: #10b981;
      box-shadow: 0 0 6px #10b981;
    }
    .dc-status-dot.dc-offline {
      background: #ef4444;
      box-shadow: 0 0 6px #ef4444;
    }
    .dc-header-tools {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .dc-icon-btn {
      background: none;
      border: none;
      color: #64748b;
      cursor: pointer;
      font-size: 12px;
      padding: 3px 6px;
      border-radius: 4px;
      transition: background 0.15s;
    }
    .dc-icon-btn:hover {
      background: #e2e8f0;
      color: #0f172a;
    }

    /* 内容主体 */
    .dc-dock-body {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: calc(100vh - 120px);
      overflow-y: auto;
    }

    /* 操作按钮 */
    .dc-btn-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .dc-primary-btn {
      width: 100%;
      color: #ffffff;
      border: none;
      padding: 9px 10px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 12.5px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
      transition: background 0.15s, transform 0.05s;
    }
    .dc-primary-btn:active {
      transform: scale(0.98);
    }
    .dc-btn-blue {
      background: #2563eb;
    }
    .dc-btn-blue:hover {
      background: #1d4ed8;
    }
    .dc-btn-sky {
      background: #0284c7;
    }
    .dc-btn-sky:hover {
      background: #0369a1;
    }
    .dc-btn-green {
      background: #059669;
    }
    .dc-btn-green:hover {
      background: #047857;
    }
    .dc-btn-active {
      background: #ea580c !important;
    }

    /* 准则配置区 */
    .dc-section-title {
      font-size: 12px;
      font-weight: 600;
      color: #475569;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .dc-preset-list {
      display: flex;
      flex-direction: column;
      gap: 5px;
      margin-bottom: 8px;
    }
    .dc-preset-pill {
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      color: #334155;
      padding: 5px 8px;
      border-radius: 6px;
      font-size: 11.5px;
      cursor: pointer;
      text-align: left;
      transition: all 0.15s;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .dc-preset-pill:hover {
      background: #e2e8f0;
      border-color: #cbd5e1;
    }
    .dc-preset-pill.dc-selected {
      background: #eff6ff;
      border-color: #3b82f6;
      color: #1d4ed8;
      font-weight: 600;
    }
    .dc-rubric-input {
      width: 100%;
      height: 52px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 12px;
      color: #1e293b;
      resize: vertical;
      outline: none;
      font-family: inherit;
    }
    .dc-rubric-input:focus {
      border-color: #2563eb;
      box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15);
    }

    /* 统计状态条 */
    .dc-stats-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #f8fafc;
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
      font-size: 11.5px;
      color: #64748b;
    }

    /* 最新选中文本概览 */
    .dc-selection-preview {
      background: #f8fafc;
      border: 1px dashed #cbd5e1;
      border-radius: 6px;
      padding: 7px 9px;
      font-size: 11px;
      color: #64748b;
      max-height: 50px;
      overflow-y: auto;
      word-break: break-all;
    }
    .dc-selection-preview.dc-has-text {
      color: #1e293b;
      border-style: solid;
    }

    /* 底部操作行 */
    .dc-bottom-actions {
      display: flex;
      gap: 8px;
    }
    .dc-sub-btn {
      flex: 1;
      background: #f1f5f9;
      color: #475569;
      border: 1px solid #e2e8f0;
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 11.5px;
      font-weight: 500;
      cursor: pointer;
      text-align: center;
    }
    .dc-sub-btn:hover {
      background: #e2e8f0;
      color: #0f172a;
    }

    /* 划选遮罩层 */
    #decision-cli-mask {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(15, 23, 42, 0.04);
      z-index: 2147483630;
      cursor: crosshair;
      user-select: none;
    }
    .dc-mask-banner {
      position: fixed;
      top: 18px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(15, 23, 42, 0.88);
      color: #ffffff;
      padding: 8px 18px;
      border-radius: 30px;
      font-size: 12.5px;
      font-weight: 500;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
      display: flex;
      align-items: center;
      gap: 10px;
      z-index: 2147483635;
      pointer-events: auto;
    }
    .dc-banner-btn {
      border: none;
      padding: 4px 12px;
      border-radius: 14px;
      font-size: 11.5px;
      cursor: pointer;
      font-weight: 600;
    }
    .dc-banner-eval-btn {
      background: #059669;
      color: #ffffff;
    }
    .dc-banner-eval-btn:hover {
      background: #047857;
    }
    .dc-banner-exit-btn {
      background: rgba(255, 255, 255, 0.2);
      color: #fff;
    }
    .dc-banner-exit-btn:hover {
      background: rgba(255, 255, 255, 0.35);
    }
    .dc-drag-rect {
      position: absolute;
      border: 2px dashed #2563eb;
      background: rgba(37, 99, 235, 0.12);
      pointer-events: none;
      z-index: 2147483632;
    }

    /* 页面上的常驻高亮框 */
    .dc-highlight-box {
      position: absolute;
      border: 2px solid #0284c7;
      background: rgba(2, 132, 199, 0.08);
      border-radius: 6px;
      z-index: 2147483620;
      pointer-events: none;
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.8);
      transition: border-color 0.2s, background 0.2s;
    }
    .dc-highlight-box.dc-res-yes {
      border-color: #ea580c;
      background: rgba(234, 88, 12, 0.09);
    }
    .dc-highlight-box.dc-res-no {
      border-color: #10b981;
      background: rgba(16, 185, 129, 0.08);
    }

    /* 紧贴高亮框顶部的结果/操作浮标 */
    .dc-highlight-badge-bar {
      position: absolute;
      right: 0;
      top: -34px;
      pointer-events: auto;
      display: flex;
      align-items: center;
      gap: 6px;
      background: #ffffff;
      padding: 4px 8px;
      border-radius: 6px;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.16);
      border: 1px solid #e2e8f0;
      font-size: 12px;
      font-weight: 600;
      white-space: nowrap;
      user-select: none;
    }
    .dc-badge-idx {
      background: #e0f2fe;
      color: #0369a1;
      padding: 1px 5px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
    }
    .dc-badge-pill {
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 11.5px;
      font-weight: 700;
    }
    .dc-badge-pill.dc-pill-pending {
      background: #f1f5f9;
      color: #64748b;
      border: 1px solid #cbd5e1;
    }
    .dc-badge-pill.dc-pill-yes {
      background: #fef2f2;
      color: #dc2626;
      border: 1px solid #fecaca;
    }
    .dc-badge-pill.dc-pill-no {
      background: #ecfdf5;
      color: #059669;
      border: 1px solid #a7f3d0;
    }
    .dc-badge-btn {
      background: #2563eb;
      color: #ffffff;
      border: none;
      padding: 2px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 11.5px;
    }
    .dc-badge-btn:hover {
      background: #1d4ed8;
    }
    .dc-badge-btn.dc-close-btn {
      background: #f1f5f9;
      color: #64748b;
      padding: 2px 6px;
    }
    .dc-badge-btn.dc-close-btn:hover {
      background: #e2e8f0;
      color: #0f172a;
    }
  `;
  document.head.appendChild(styleEl);

  // -------------------------------------------------------------
  // 3. 网络请求封装 (GM_xmlhttpRequest 优先)
  // -------------------------------------------------------------
  function makeRequest(url, method, data) {
    return new Promise((resolve, reject) => {
      if (typeof GM_xmlhttpRequest === 'function') {
        GM_xmlhttpRequest({
          method: method || 'GET',
          url: url,
          headers: { 'Content-Type': 'application/json' },
          data: data ? JSON.stringify(data) : undefined,
          timeout: 20000,
          onload: function (response) {
            try {
              const parsed = JSON.parse(response.responseText);
              if (response.status >= 200 && response.status < 300) {
                resolve(parsed);
              } else {
                reject(new Error(parsed.error || `HTTP ${response.status}`));
              }
            } catch (err) {
              reject(new Error(`响应解析失败: ${response.responseText}`));
            }
          },
          onerror: function () {
            reject(new Error('网络连接失败，请确认 decision-cli serve 正在运行'));
          },
          ontimeout: function () {
            reject(new Error('推理请求超时'));
          }
        });
      } else {
        fetch(url, {
          method: method || 'GET',
          headers: { 'Content-Type': 'application/json' },
          body: data ? JSON.stringify(data) : undefined
        })
          .then(async (res) => {
            const parsed = await res.json();
            if (!res.ok) throw new Error(parsed.error || `HTTP ${res.status}`);
            return parsed;
          })
          .then(resolve)
          .catch(reject);
      }
    });
  }

  // -------------------------------------------------------------
  // 4. 构建浮窗 UI (含收起与展开状态)
  // -------------------------------------------------------------
  const dock = document.createElement('div');
  dock.id = 'decision-cli-dock';
  if (isDockCollapsed) {
    dock.classList.add('dc-collapsed');
  }

  dock.innerHTML = `
    <!-- 收起状态极简微型药丸 -->
    <div class="dc-mini-pill" id="dc-mini-pill" title="点击展开 Decision 判定面板">
      <span class="dc-status-dot" id="dc-mini-status-dot"></span>
      <span>🎯 Decision</span>
      <span class="dc-mini-tag" id="dc-mini-rubric-tag">${currentRubric}</span>
      <span style="color: #94a3b8; font-size: 11px;">⛶</span>
    </div>

    <!-- 展开状态标题栏 -->
    <div class="dc-dock-header" id="dc-dock-drag-header">
      <div class="dc-dock-title">
        <span class="dc-status-dot" id="dc-status-dot" title="正在探测 serve 状态..."></span>
        <span>Decision 决策判定</span>
      </div>
      <div class="dc-header-tools">
        <button class="dc-icon-btn" id="dc-btn-config" title="配置 API 地址">⚙️</button>
        <button class="dc-icon-btn" id="dc-btn-ping" title="检查 serve 状态">🔄</button>
        <button class="dc-icon-btn" id="dc-btn-collapse" title="收起浮窗">▼</button>
      </div>
    </div>

    <!-- 展开状态内容主体 -->
    <div class="dc-dock-body">
      <!-- 可折叠的 API 配置栏 -->
      <div id="dc-config-panel" style="display: none; background: #f8fafc; padding: 8px; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 11.5px;">
        <div style="font-weight: 600; margin-bottom: 4px; color: #475569;">API 服务地址配置</div>
        <div style="display: flex; gap: 6px;">
          <input type="text" id="dc-api-input" style="flex: 1; border: 1px solid #cbd5e1; border-radius: 4px; padding: 3px 6px; font-size: 11.5px;" value="${serverApiUrl}" />
          <button id="dc-btn-save-api" class="dc-icon-btn" style="background: #2563eb; color: #fff; padding: 3px 8px; font-weight: 600;">保存</button>
        </div>
      </div>

      <!-- 核心操作按钮组（批量拖拽标注 vs 连续自动判定） -->
      <div class="dc-btn-grid">
        <button class="dc-primary-btn dc-btn-sky" id="dc-btn-batch-mark" title="连续拖拽框选多个区域，稍后一键批量判定">
          <span>📋 批量拖拽标注</span>
        </button>
        <button class="dc-primary-btn dc-btn-blue" id="dc-btn-auto-mark" title="连续拖拽框选，松开鼠标立即自动触发模型推理">
          <span>⚡️ 连续自动判定</span>
        </button>
      </div>

      <!-- 醒目的批量判定执行大按钮 -->
      <button class="dc-primary-btn dc-btn-green" id="dc-btn-batch-eval" style="display: none; padding: 10px 12px; font-size: 13.5px;">
        <span>⚡️ 立即批量判定 (<span id="dc-pending-badge">0</span> 处待判)</span>
      </button>

      <!-- 高亮与待判定统计条 -->
      <div class="dc-stats-row">
        <span>已标注: <b id="dc-stat-total" style="color: #0f172a;">0</b> 处</span>
        <span>待判定: <b id="dc-stat-pending" style="color: #0284c7;">0</b></span>
        <span>已完成: <b id="dc-stat-done" style="color: #059669;">0</b></span>
      </div>

      <!-- 判定标准编辑 -->
      <div>
        <div class="dc-section-title">
          <span>判定标准 (Rubrics)</span>
        </div>
        <div class="dc-preset-list" id="dc-preset-list"></div>
        <textarea class="dc-rubric-input" id="dc-rubric-input" placeholder="输入自定义判定准则..."></textarea>
      </div>

      <!-- 最新选区概览 -->
      <div>
        <div class="dc-section-title">
          <span>最新选区概览</span>
          <span id="dc-text-count" style="color: #94a3b8; font-weight: normal;">未划选</span>
        </div>
        <div class="dc-selection-preview" id="dc-selection-preview">点击「批量拖拽标注」在网页上连续拉框，随后一键批量判定。</div>
      </div>

      <!-- 底部管理按钮 -->
      <div class="dc-bottom-actions">
        <button class="dc-sub-btn" id="dc-btn-clear-all">清空所有高亮 (0)</button>
        <button class="dc-sub-btn" id="dc-btn-reeval-all">全部重新判定</button>
      </div>

      <div id="dc-status-msg" style="font-size: 11px; color: #64748b; text-align: center;"></div>
    </div>
  `;
  document.body.appendChild(dock);

  // 绑定元素引用
  const miniPill = dock.querySelector('#dc-mini-pill');
  const miniStatusDot = dock.querySelector('#dc-mini-status-dot');
  const miniRubricTag = dock.querySelector('#dc-mini-rubric-tag');
  const statusDot = dock.querySelector('#dc-status-dot');
  const btnPing = dock.querySelector('#dc-btn-ping');
  const btnConfig = dock.querySelector('#dc-btn-config');
  const btnCollapse = dock.querySelector('#dc-btn-collapse');
  const configPanel = dock.querySelector('#dc-config-panel');
  const apiInput = dock.querySelector('#dc-api-input');
  const btnSaveApi = dock.querySelector('#dc-btn-save-api');
  const btnBatchMark = dock.querySelector('#dc-btn-batch-mark');
  const btnAutoMark = dock.querySelector('#dc-btn-auto-mark');
  const btnBatchEval = dock.querySelector('#dc-btn-batch-eval');
  const pendingBadge = dock.querySelector('#dc-pending-badge');
  const statTotal = dock.querySelector('#dc-stat-total');
  const statPending = dock.querySelector('#dc-stat-pending');
  const statDone = dock.querySelector('#dc-stat-done');
  const rubricInput = dock.querySelector('#dc-rubric-input');
  const presetList = dock.querySelector('#dc-preset-list');
  const previewBox = dock.querySelector('#dc-selection-preview');
  const textCount = dock.querySelector('#dc-text-count');
  const btnClearAll = dock.querySelector('#dc-btn-clear-all');
  const btnReevalAll = dock.querySelector('#dc-btn-reeval-all');
  const statusMsg = dock.querySelector('#dc-status-msg');

  // 展开与收起状态切换
  function toggleCollapse(collapsed) {
    isDockCollapsed = collapsed;
    dock.classList.toggle('dc-collapsed', isDockCollapsed);
    setStorage(STORAGE_KEY_COLLAPSED, isDockCollapsed);
  }

  btnCollapse.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleCollapse(true);
  });

  miniPill.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleCollapse(false);
  });

  // 渲染预设准则标签
  function renderPresets() {
    presetList.innerHTML = '';
    PRESET_RUBRICS.forEach((preset) => {
      const pill = document.createElement('div');
      pill.className = 'dc-preset-pill' + (preset === currentRubric ? ' dc-selected' : '');
      pill.textContent = preset;
      pill.title = preset;
      pill.addEventListener('click', () => {
        currentRubric = preset;
        rubricInput.value = preset;
        miniRubricTag.textContent = preset;
        setStorage(STORAGE_KEY_RUBRIC, preset);
        renderPresets();
        updateBannerContent();
      });
      presetList.appendChild(pill);
    });
  }
  rubricInput.value = currentRubric;
  miniRubricTag.textContent = currentRubric;
  renderPresets();

  rubricInput.addEventListener('input', () => {
    currentRubric = rubricInput.value.trim();
    miniRubricTag.textContent = currentRubric || '无准则';
    setStorage(STORAGE_KEY_RUBRIC, currentRubric);
    const pills = presetList.querySelectorAll('.dc-preset-pill');
    pills.forEach((p) => {
      p.classList.toggle('dc-selected', p.textContent === currentRubric);
    });
    updateBannerContent();
  });

  // 配置面板切换与保存
  btnConfig.addEventListener('click', (e) => {
    e.stopPropagation();
    const isShown = configPanel.style.display !== 'none';
    configPanel.style.display = isShown ? 'none' : 'block';
  });

  btnSaveApi.addEventListener('click', (e) => {
    e.stopPropagation();
    const val = (apiInput.value || '').trim().replace(/\/+$/, '');
    if (val) {
      serverApiUrl = val;
      setStorage(STORAGE_KEY_API_URL, serverApiUrl);
      configPanel.style.display = 'none';
      checkServerHealth();
    }
  });

  // 检查 decision-cli serve 连通状态
  async function checkServerHealth() {
    statusDot.className = 'dc-status-dot';
    miniStatusDot.className = 'dc-status-dot';
    statusDot.title = '正在探测服务...';
    try {
      const res = await makeRequest(`${serverApiUrl}/health`, 'GET');
      if (res && res.status === 'ok') {
        statusDot.className = 'dc-status-dot dc-online';
        miniStatusDot.className = 'dc-status-dot dc-online';
        statusDot.title = `decision-cli serve 在线 (${res.default_model})`;
        miniPill.title = `serve 在线 (${res.default_model}) · 点击展开面板`;
        statusMsg.textContent = `服务已就绪 (${res.default_model})`;
        statusMsg.style.color = '#10b981';
      } else {
        throw new Error('服务返回异常');
      }
    } catch (err) {
      statusDot.className = 'dc-status-dot dc-offline';
      miniStatusDot.className = 'dc-status-dot dc-offline';
      statusDot.title = `未连接到 serve: ${err.message}。请运行 decision-cli serve`;
      miniPill.title = '未连接服务，请运行 decision-cli serve · 点击展开面板';
      statusMsg.textContent = '未连接服务，请运行 decision-cli serve';
      statusMsg.style.color = '#ef4444';
    }
  }
  btnPing.addEventListener('click', checkServerHealth);
  checkServerHealth();

  // -------------------------------------------------------------
  // 5. 网页区域划选与多模式调度 (批量标注 vs 连续自动判定)
  // -------------------------------------------------------------
  btnBatchMark.addEventListener('click', () => {
    if (isSelecting && selectionMode === 'batch') {
      exitSelectionMode();
    } else {
      enterSelectionMode('batch');
    }
  });

  btnAutoMark.addEventListener('click', () => {
    if (isSelecting && selectionMode === 'auto') {
      exitSelectionMode();
    } else {
      enterSelectionMode('auto');
    }
  });

  function updateBannerContent() {
    const mask = document.getElementById('decision-cli-mask');
    if (!mask) return;
    const banner = mask.querySelector('.dc-mask-banner');
    if (!banner) return;

    const pendingCount = activeHighlights.filter((h) => h.status === 'pending').length;

    if (selectionMode === 'batch') {
      banner.innerHTML = `
        <span>📋 <b>批量标注模式</b> · 已标注 <b>${activeHighlights.length}</b> 处 (${pendingCount} 待判) · 准则: <b>${currentRubric}</b></span>
        ${pendingCount > 0 ? `<button class="dc-banner-btn dc-banner-eval-btn" id="dc-btn-banner-batch-eval">⚡️ 立即批量判定 (${pendingCount})</button>` : ''}
        <button class="dc-banner-btn dc-banner-exit-btn" id="dc-btn-exit-banner">退出选框 (ESC)</button>
      `;
    } else {
      banner.innerHTML = `
        <span>⚡️ <b>连续自动判定</b> · 选定即推理 · 准则: <b>${currentRubric}</b></span>
        <button class="dc-banner-btn dc-banner-exit-btn" id="dc-btn-exit-banner">退出选框 (ESC)</button>
      `;
    }

    const bannerEvalBtn = banner.querySelector('#dc-btn-banner-batch-eval');
    if (bannerEvalBtn) {
      bannerEvalBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        runBatchDecision();
      });
    }

    const bannerExitBtn = banner.querySelector('#dc-btn-exit-banner');
    if (bannerExitBtn) {
      bannerExitBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        exitSelectionMode();
      });
    }
  }

  function enterSelectionMode(mode) {
    isSelecting = true;
    selectionMode = mode;

    btnBatchMark.classList.toggle('dc-btn-active', mode === 'batch');
    btnAutoMark.classList.toggle('dc-btn-active', mode === 'auto');
    if (mode === 'batch') {
      btnBatchMark.innerHTML = '<span>⏹ 停止标注 (ESC)</span>';
      btnAutoMark.innerHTML = '<span>⚡️ 连续自动判定</span>';
    } else {
      btnAutoMark.innerHTML = '<span>⏹ 停止自动判定</span>';
      btnBatchMark.innerHTML = '<span>📋 批量拖拽标注</span>';
    }

    let mask = document.getElementById('decision-cli-mask');
    if (!mask) {
      mask = document.createElement('div');
      mask.id = 'decision-cli-mask';
      mask.innerHTML = `<div class="dc-mask-banner"></div>`;
      document.body.appendChild(mask);
    }
    updateBannerContent();

    let startX = 0, startY = 0;
    let dragRect = null;
    let isMouseDown = false;

    function onMouseDown(e) {
      if (e.target.closest('#decision-cli-dock') || e.target.closest('.dc-highlight-badge-bar') || e.target.closest('.dc-mask-banner')) return;
      isMouseDown = true;
      startX = e.pageX;
      startY = e.pageY;

      dragRect = document.createElement('div');
      dragRect.className = 'dc-drag-rect';
      dragRect.style.left = `${startX}px`;
      dragRect.style.top = `${startY}px`;
      dragRect.style.width = '0px';
      dragRect.style.height = '0px';
      document.body.appendChild(dragRect);

      e.preventDefault();
    }

    function onMouseMove(e) {
      if (!isMouseDown || !dragRect) return;
      const curX = e.pageX;
      const curY = e.pageY;

      const left = Math.min(startX, curX);
      const top = Math.min(startY, curY);
      const width = Math.abs(curX - startX);
      const height = Math.abs(curY - startY);

      dragRect.style.left = `${left}px`;
      dragRect.style.top = `${top}px`;
      dragRect.style.width = `${width}px`;
      dragRect.style.height = `${height}px`;
    }

    function onMouseUp(e) {
      if (!isMouseDown) return;
      isMouseDown = false;

      const endX = e.pageX;
      const endY = e.pageY;
      const left = Math.min(startX, endX);
      const top = Math.min(startY, endY);
      const width = Math.abs(endX - startX);
      const height = Math.abs(endY - startY);

      if (dragRect) {
        dragRect.remove();
        dragRect = null;
      }

      // 如果拖拽范围极小（< 12px），智能选取点击位置下方的最近非浮窗元素
      let targetRect = { left, top, right: left + width, bottom: top + height, width, height };
      if (width < 12 && height < 12) {
        const elem = document.elementFromPoint(e.clientX, e.clientY);
        if (elem && !elem.closest('#decision-cli-dock') && elem !== document.body && elem !== document.documentElement) {
          const b = elem.getBoundingClientRect();
          targetRect = {
            left: b.left + window.scrollX,
            top: b.top + window.scrollY,
            right: b.right + window.scrollX,
            bottom: b.bottom + window.scrollY,
            width: b.width,
            height: b.height,
            targetElement: elem
          };
        }
      }

      // 生成高亮卡片
      const hlObj = applyHighlightAndExtract(targetRect);

      // 更新顶部 Banner 统计
      updateBannerContent();

      // 如果是「连续自动判定」模式，松开即刻触发单项推理
      if (hlObj && selectionMode === 'auto') {
        runDecisionForHighlight(hlObj);
      }
    }

    function onKeyDown(e) {
      if (e.key === 'Escape') {
        if (dragRect) dragRect.remove();
        exitSelectionMode();
      } else if (e.key === 'Enter') {
        // 回车键快捷触发批量判定
        runBatchDecision();
      }
    }

    mask.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('keydown', onKeyDown);

    mask._cleanup = () => {
      mask.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('keydown', onKeyDown);
      mask.remove();
    };
  }

  function exitSelectionMode() {
    isSelecting = false;
    btnBatchMark.classList.remove('dc-btn-active');
    btnAutoMark.classList.remove('dc-btn-active');
    btnBatchMark.innerHTML = '<span>📋 批量拖拽标注</span>';
    btnAutoMark.innerHTML = '<span>⚡️ 连续自动判定</span>';
    const mask = document.getElementById('decision-cli-mask');
    if (mask && mask._cleanup) {
      mask._cleanup();
    }
  }

  // -------------------------------------------------------------
  // 6. 文本提取与多高亮卡片管理
  // -------------------------------------------------------------
  function extractTextFromBoundingBox(rect) {
    if (rect.targetElement) {
      return (rect.targetElement.innerText || rect.targetElement.textContent || '').trim();
    }

    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        if (!node.textContent || !node.textContent.trim()) return NodeFilter.FILTER_REJECT;
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;
        const tag = parent.tagName.toLowerCase();
        if (['script', 'style', 'noscript', 'textarea', 'input', 'svg'].includes(tag)) {
          return NodeFilter.FILTER_REJECT;
        }
        if (parent.closest('#decision-cli-dock') || parent.closest('.dc-highlight-box')) {
          return NodeFilter.FILTER_REJECT;
        }
        const style = window.getComputedStyle(parent);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });

    const fragments = [];
    let currentNode;
    const range = document.createRange();

    while ((currentNode = walker.nextNode())) {
      try {
        range.selectNodeContents(currentNode);
        const rects = range.getClientRects();
        for (let i = 0; i < rects.length; i++) {
          const r = rects[i];
          const pageLeft = r.left + window.scrollX;
          const pageTop = r.top + window.scrollY;
          const pageRight = pageLeft + r.width;
          const pageBottom = pageTop + r.height;

          const intersects = !(
            pageLeft > rect.right ||
            pageRight < rect.left ||
            pageTop > rect.bottom ||
            pageBottom < rect.top
          );

          if (intersects) {
            const txt = currentNode.textContent.trim();
            if (txt && !fragments.includes(txt)) {
              fragments.push(txt);
            }
            break;
          }
        }
      } catch (e) {}
    }

    return fragments.join('\n');
  }

  function updateHighlightsSummary() {
    const total = activeHighlights.length;
    const pending = activeHighlights.filter((h) => h.status === 'pending').length;
    const done = activeHighlights.filter((h) => h.status === 'done').length;

    statTotal.textContent = total;
    statPending.textContent = pending;
    statDone.textContent = done;
    btnClearAll.textContent = `清空所有高亮 (${total})`;

    // 更新批量判定按钮状态
    if (pending > 0) {
      btnBatchEval.style.display = 'flex';
      pendingBadge.textContent = pending;
    } else {
      btnBatchEval.style.display = 'none';
    }

    updateBannerContent();
  }

  function removeHighlight(hlObj) {
    if (hlObj && hlObj.element) {
      hlObj.element.remove();
    }
    activeHighlights = activeHighlights.filter((h) => h !== hlObj);
    updateHighlightsSummary();
  }

  function clearAllHighlights() {
    activeHighlights.forEach((h) => {
      if (h.element) h.element.remove();
    });
    activeHighlights = [];
    updateHighlightsSummary();
    previewBox.textContent = '已清空所有高亮选区。';
    previewBox.classList.remove('dc-has-text');
    textCount.textContent = '未划选';
  }
  btnClearAll.addEventListener('click', clearAllHighlights);

  function applyHighlightAndExtract(rect) {
    const extractedText = extractTextFromBoundingBox(rect);
    if (!extractedText) {
      statusMsg.textContent = '所选区域内未检测到可读文本，请重新划选。';
      statusMsg.style.color = '#eab308';
      return null;
    }

    highlightCounter++;
    const curIdx = highlightCounter;

    // 创建高亮覆盖层
    const hl = document.createElement('div');
    hl.className = 'dc-highlight-box';
    hl.style.left = `${rect.left}px`;
    hl.style.top = `${rect.top}px`;
    hl.style.width = `${Math.max(rect.width, 120)}px`;
    hl.style.height = `${Math.max(rect.height, 40)}px`;

    // 附带的操作徽章条
    const badgeBar = document.createElement('div');
    badgeBar.className = 'dc-highlight-badge-bar';
    badgeBar.innerHTML = `
      <span class="dc-badge-idx">#${curIdx}</span>
      <span class="dc-badge-len" style="color: #64748b; font-weight: normal;">${extractedText.length} 字</span>
      <span class="dc-badge-pill dc-pill-pending">待判定</span>
      <button class="dc-badge-btn" id="dc-badge-eval-btn">⚡️ 判定</button>
      <button class="dc-badge-btn dc-close-btn" id="dc-badge-close-btn" title="删除该高亮">✕</button>
    `;
    hl.appendChild(badgeBar);
    document.body.appendChild(hl);

    const hlObj = {
      id: 'hl_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
      index: curIdx,
      element: hl,
      badgeBar: badgeBar,
      text: extractedText,
      rect: rect,
      status: 'pending',
      result: null
    };
    activeHighlights.push(hlObj);
    updateHighlightsSummary();

    // 同步到浮窗最新信息
    previewBox.textContent = `#${curIdx}: ` + extractedText;
    previewBox.classList.add('dc-has-text');
    textCount.textContent = `${extractedText.length} 字符`;

    // 绑定高亮浮标按钮
    badgeBar.querySelector('#dc-badge-close-btn').addEventListener('click', () => removeHighlight(hlObj));
    badgeBar.querySelector('#dc-badge-eval-btn').addEventListener('click', () => runDecisionForHighlight(hlObj));

    return hlObj;
  }

  // -------------------------------------------------------------
  // 7. 单项判定与批量判定调度逻辑
  // -------------------------------------------------------------
  btnBatchEval.addEventListener('click', runBatchDecision);

  btnReevalAll.addEventListener('click', () => {
    if (activeHighlights.length === 0) {
      statusMsg.textContent = '暂无高亮区域可重判';
      statusMsg.style.color = '#ef4444';
      return;
    }
    // 将所有高亮标记为待判状态并全量重判
    activeHighlights.forEach((h) => {
      h.status = 'pending';
    });
    runBatchDecision();
  });

  // 批量判定入口
  async function runBatchDecision() {
    const pendingTargets = activeHighlights.filter((h) => h.status === 'pending');
    if (pendingTargets.length === 0) {
      statusMsg.textContent = '当前没有待判定的高亮区域';
      statusMsg.style.color = '#64748b';
      return;
    }

    const rubric = (rubricInput.value || '').trim();
    if (!rubric) {
      statusMsg.textContent = '请先输入或选择判定准则';
      statusMsg.style.color = '#ef4444';
      return;
    }

    btnBatchEval.disabled = true;
    btnBatchEval.innerHTML = `<span>⏳ 正在批量判定 (${pendingTargets.length} 处)...</span>`;
    statusMsg.textContent = `开始批量判定 ${pendingTargets.length} 处文本...`;
    statusMsg.style.color = '#0284c7';

    let successCount = 0;
    let yesCount = 0;

    // 并发发起所有待判定项的推理请求
    const promises = pendingTargets.map(async (hlObj) => {
      const res = await runDecisionForHighlight(hlObj);
      if (res) {
        successCount++;
        if (res.decision === '是' || res.passed) yesCount++;
      }
    });

    await Promise.allSettled(promises);

    btnBatchEval.disabled = false;
    updateHighlightsSummary();

    statusMsg.textContent = `批量判定完成: 共 ${successCount}/${pendingTargets.length} 处 (是: ${yesCount}, 否: ${successCount - yesCount})`;
    statusMsg.style.color = '#059669';
  }

  // 单项判定实现
  async function runDecisionForHighlight(hlObj) {
    if (!hlObj || !hlObj.text) return null;
    const rubric = (rubricInput.value || '').trim();
    if (!rubric) {
      statusMsg.textContent = '请输入或选择判定准则';
      statusMsg.style.color = '#ef4444';
      return null;
    }

    hlObj.status = 'evaluating';
    const badgeBar = hlObj.badgeBar;
    badgeBar.innerHTML = `
      <span class="dc-badge-idx">#${hlObj.index}</span>
      <span style="color: #64748b; font-size: 11px;">⏳ 判定中...</span>
      <button class="dc-badge-btn dc-close-btn" id="dc-badge-close-btn">✕</button>
    `;
    badgeBar.querySelector('#dc-badge-close-btn').addEventListener('click', () => removeHighlight(hlObj));

    try {
      const payload = {
        rubrics: rubric,
        context: hlObj.text
      };

      const res = await makeRequest(`${serverApiUrl}/v1/noun`, 'POST', payload);

      const isYes = res.decision === '是' || res.passed === true;
      const probPercent = (res.probability * 100).toFixed(1) + '%';
      const latencyStr = res.latency_ms ? `${res.latency_ms.toFixed(0)}ms` : '';
      const cachedTag = res.cached ? ' [已缓存]' : '';

      hlObj.status = 'done';
      hlObj.result = res;

      // 更新高亮框边框与浮标结果
      hlObj.element.classList.remove('dc-res-yes', 'dc-res-no');
      hlObj.element.classList.add(isYes ? 'dc-res-yes' : 'dc-res-no');

      badgeBar.innerHTML = `
        <span class="dc-badge-idx">#${hlObj.index}</span>
        <span class="dc-badge-pill ${isYes ? 'dc-pill-yes' : 'dc-pill-no'}">
          ${res.decision || (isYes ? '是' : '否')} ${probPercent}
        </span>
        <button class="dc-badge-btn" id="dc-badge-reeval-btn" title="重新判定">↺</button>
        <button class="dc-badge-btn dc-close-btn" id="dc-badge-close-btn" title="删除该高亮">✕</button>
      `;

      badgeBar.title = `选区 #${hlObj.index}\n准则: ${rubric}\n置信概率: ${probPercent}\n耗时: ${latencyStr}${cachedTag}\n模型: ${res.model || 'gemma-4-26b-a4b'}`;

      badgeBar.querySelector('#dc-badge-close-btn').addEventListener('click', () => removeHighlight(hlObj));
      badgeBar.querySelector('#dc-badge-reeval-btn').addEventListener('click', () => runDecisionForHighlight(hlObj));

      updateHighlightsSummary();
      return res;
    } catch (err) {
      hlObj.status = 'error';
      badgeBar.innerHTML = `
        <span class="dc-badge-idx">#${hlObj.index}</span>
        <span style="color: #ef4444; font-size: 11px;">⚠️ 失败</span>
        <button class="dc-badge-btn" id="dc-badge-retry-btn">重试</button>
        <button class="dc-badge-btn dc-close-btn" id="dc-badge-close-btn">✕</button>
      `;
      badgeBar.querySelector('#dc-badge-close-btn').addEventListener('click', () => removeHighlight(hlObj));
      badgeBar.querySelector('#dc-badge-retry-btn').addEventListener('click', () => runDecisionForHighlight(hlObj));

      updateHighlightsSummary();
      return null;
    }
  }

  // -------------------------------------------------------------
  // 8. 浮窗自由拖拽定位 (展开与收起状态均支持)
  // -------------------------------------------------------------
  let isDraggingDock = false;
  let dockOffsetX = 0, dockOffsetY = 0;

  function handleDragStart(e) {
    if (e.target.closest('button') || e.target.closest('input')) return;
    isDraggingDock = true;
    const rect = dock.getBoundingClientRect();
    dockOffsetX = e.clientX - rect.left;
    dockOffsetY = e.clientY - rect.top;
    e.preventDefault();
  }

  const dragHeader = dock.querySelector('#dc-dock-drag-header');
  dragHeader.addEventListener('mousedown', handleDragStart);
  miniPill.addEventListener('mousedown', handleDragStart);

  window.addEventListener('mousemove', (e) => {
    if (!isDraggingDock) return;
    let newLeft = e.clientX - dockOffsetX;
    let newTop = e.clientY - dockOffsetY;

    const maxLeft = window.innerWidth - dock.offsetWidth - 10;
    const maxTop = window.innerHeight - dock.offsetHeight - 10;
    newLeft = Math.max(10, Math.min(newLeft, maxLeft));
    newTop = Math.max(10, Math.min(newTop, maxTop));

    dock.style.left = `${newLeft}px`;
    dock.style.top = `${newTop}px`;
    dock.style.right = 'auto';
    dock.style.bottom = 'auto';
  });

  window.addEventListener('mouseup', () => {
    if (isDraggingDock) {
      isDraggingDock = false;
    }
  });

})();
