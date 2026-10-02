/* River Model Lab: static, dependency-free and compatible with file://. */
(() => {
  "use strict";
  const data = window.RIVER_LAB_DATA;
  const $ = (id) => document.getElementById(id);
  const FEATURES = [
    { key: "water_depth", name: "水深", unit: "m", digits: 3 },
    { key: "discharge", name: "流量", unit: "m³/s", digits: 2 },
    { key: "sediment", name: "含沙浓度", unit: "kg/m³", digits: 4 }
  ];
  const MODELS = [
    { key: "physics_guided", name: "物理引导模型", color: "#132b3b", cls: "plot-physics" },
    { key: "data_only", name: "纯数据模型", color: "#497e72", cls: "plot-data" },
    { key: "persistence", name: "持续性基线", color: "#b38447", cls: "plot-persistence" }
  ];
  const METHODS = { target_only: "目标河少样本训练", zero_shot: "零目标监督迁移", fine_tuned: "预训练后微调" };
  const SVG_NS = "http://www.w3.org/2000/svg";
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const fmt = (value, digits = 3) => finite(value) ? value.toLocaleString("zh-CN", { minimumFractionDigits: 0, maximumFractionDigits: digits }) : "—";
  const fixed = (value, digits = 4) => finite(value) ? value.toFixed(digits) : "—";
  const setText = (id, value) => { $(id).textContent = value ?? ""; };
  const make = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const sv = (tag, attrs = {}, text) => {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const time = (value, short = false) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value ? String(value) : "未提供";
    const iso = date.toISOString();
    return short ? iso.slice(5, 10) + " " + iso.slice(11, 16) : iso.slice(0, 10) + " " + iso.slice(11, 16) + " UTC";
  };
  const localPath = (value) => typeof value === "string" && /^(?:\.\.?\/|[a-zA-Z0-9_-])/.test(value) && !/^[a-z]+:/i.test(value) && !value.startsWith("//");
  const resourceLink = (title, path, className = "", download = true) => {
    if (!localPath(path)) return null;
    const link = make("a", undefined, className);
    link.href = path;
    if (download) link.setAttribute("download", "");
    link.append(make("span", title), make("span", "↗"));
    return link;
  };
  const validRiver = (river) => river && typeof river.id === "string" && river.forecast && Array.isArray(river.forecast.timestamps) && river.forecast.timestamps.length > 0 && Array.isArray(river.forecast.physics_guided);
  if (!data || data.schema_version !== 1 || !Array.isArray(data.rivers) || !data.rivers.length || !data.rivers.every(validRiver)) {
    $("data-error").hidden = false;
    if (data) $("data-error").append(make("p", "数据结构不完整或版本不受支持，请重新生成网站展示数据。"));
    return;
  }
  const state = { river: 0, feature: 1, node: 0, view: "forecast" };
  const synthetic = data.data_status !== "observed";
  const observationLabel = synthetic ? "合成观测" : "观测参考";
  const historyLabel = synthetic ? "合成历史" : "观测历史";
  if (!synthetic) {
    document.querySelector(".research-notice p").textContent = "本次模型使用用户提供的观测记录训练与评估；河道参数仍是未校准的情景设定。请结合测站质量、单位和独立测试结果阅读预测。";
    document.querySelector(".status-pill").textContent = "观测数据";
    setText("river-data-state", "观测输入研究");
    document.querySelector(".chart-legend span").lastChild.textContent = observationLabel;
  }
  const getRiver = () => data.rivers[state.river];
  const getFeature = () => FEATURES[state.feature];
  const valueAt = (rows, index, node = state.node, feature = state.feature) => rows?.[index]?.[node]?.[feature];
  $("research-content").hidden = false;
  setText("river-count", data.rivers.length);
  $("forecast-hours").replaceChildren(document.createTextNode(String(data.protocol?.horizon_hours ?? 6)), make("span", "h"));
  setText("generated-at", "实验生成时间\n" + time(data.created_at));
  $("generated-at").style.whiteSpace = "pre-line";
  setText("model-label", data.model_label || "物理引导图时序网络");

  // WAI-ARIA tab keyboard behavior: arrow keys, Home and End activate tabs.
  const wireTabKeys = (container) => {
    container.addEventListener("keydown", (event) => {
      const tabs = [...container.querySelectorAll('[role="tab"]')];
      const current = tabs.indexOf(document.activeElement);
      if (current < 0) return;
      let next;
      if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (current + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      tabs[next].focus();
      tabs[next].click();
      tabs[next].scrollIntoView({ block: "nearest", inline: "nearest" });
    });
  };
  data.rivers.forEach((river, index) => {
    const button = make("button", undefined, "river-tab");
    button.type = "button";
    button.id = "river-tab-" + index;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", "river-panel");
    button.append(make("span", river.name_zh), make("small", river.name_en));
    button.addEventListener("click", () => { state.river = index; state.node = 0; renderRiver(); });
    $("river-tabs").append(button);
  });
  wireTabKeys($("river-tabs"));
  document.querySelectorAll(".view-tabs [data-view]").forEach((button) => {
    button.addEventListener("click", () => { state.view = button.dataset.view; renderView(); });
  });
  wireTabKeys(document.querySelector(".view-tabs"));
  document.querySelectorAll(".feature-tabs [data-feature]").forEach((button) => {
    button.addEventListener("click", () => { state.feature = Number(button.dataset.feature); renderView(); });
  });
  wireTabKeys(document.querySelector(".feature-tabs"));
  $("node-select").setAttribute("aria-label", "河段");
  $("node-select").addEventListener("change", (event) => { state.node = Number(event.target.value); renderView(); });

  function renderRiver() {
    const river = getRiver();
    [...$("river-tabs").children].forEach((button, index) => {
      const selected = index === state.river;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    $("river-panel").setAttribute("aria-labelledby", "river-tab-" + state.river);
    setText("river-region", river.region);
    setText("river-name", river.name_zh);
    setText("river-name-en", river.name_en);
    setText("river-summary", river.summary);
    setText("river-hydrology", river.hydrology);
    const references = (river.profile?.source_ids || []).map((id) => (data.sources || []).findIndex((source) => source.id === id)).filter((index) => index >= 0);
    if (references.length) {
      const links = make("span", "背景资料 ", "river-source-links");
      references.forEach((index) => {
        const link = make("a", "[" + (index + 1) + "]");
        link.href = "#source-" + index;
        link.title = data.sources[index].title;
        links.append(link, document.createTextNode(" "));
      });
      $("river-summary").append(document.createTextNode(" "), links);
    }
    setText("river-number", String(state.river + 1).padStart(2, "0") + " / " + String(data.rivers.length).padStart(2, "0"));
    const reaches = Array.isArray(river.reaches) && river.reaches.length ? river.reaches : ["示意河段 1", "示意河段 2", "示意河段 3", "示意河段 4", "示意河段 5"];
    $("node-select").replaceChildren(...reaches.map((name, index) => {
      const option = make("option", name);
      option.value = String(index);
      return option;
    }));
    $("node-select").value = String(state.node);
    const profile = river.profile || {};
    const parameters = [
      ["河宽", fmt(profile.width_m, 0), "m"],
      ["情景坡度", finite(profile.slope) ? profile.slope.toFixed(5) : "—", ""],
      ["曼宁糙率", fixed(profile.manning_n, 3), "s·m⁻¹⁄³"],
      ["基准流量", fmt(profile.base_discharge, 0), "m³/s"]
    ];
    $("river-parameters").replaceChildren(...parameters.map(([label, value, unit]) => {
      const entry = make("div");
      const dd = make("dd", value);
      if (unit) dd.append(make("span", unit));
      entry.append(make("dt", label), dd);
      return entry;
    }));
    const protocol = river.protocol || data.protocol || {};
    setText("training-summary", `${fmt(protocol.hours, 0)} 小时序列 · ${protocol.epochs ?? "—"} 轮训练`);
    const downloads = river.downloads || {};
    const labels = { predictions: "预测 CSV", backtest: "回测 CSV", checkpoint: "物理引导权重 .pt", data_checkpoint: "纯数据权重 .pt", metrics: "评估指标 JSON" };
    $("river-downloads").replaceChildren();
    Object.entries(labels).forEach(([key, label]) => {
      const link = resourceLink(label, downloads[key]);
      if (link) $("river-downloads").append(link);
    });
    renderView();
  }

  function renderView() {
    const river = getRiver();
    const feature = getFeature();
    document.querySelectorAll("[data-feature]").forEach((button) => {
      const selected = Number(button.dataset.feature) === state.feature;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll("[data-view]").forEach((button) => {
      const selected = button.dataset.view === state.view;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    $("chart-view").setAttribute("aria-labelledby", "view-" + state.view);
    setText("chart-kicker", state.view === "forecast" ? "水沙过程 · 历史与预测" : "水沙过程 · 留出集回测");
    setText("chart-title", feature.name + (state.view === "forecast" ? "预测" : "回测") + " / " + feature.unit);
    const origin = state.view === "backtest" ? river.backtest.issued_at : river.forecast.issued_at;
    setText("issued-at", "预测起点 " + time(origin) + (state.view === "backtest" ? " · 留出测试窗口" : " · 序列末时刻"));
    $("interval-legend").hidden = state.view !== "forecast";
    const coverage = river.forecast.interval_coverage?.[feature.key];
    const interval = river.forecast.interval_label || "验证集绝对残差 90% 经验区间";
    setText("chart-note", state.view === "forecast"
      ? `实线历史为${observationLabel}；阴影区为序列末时刻之后的推演，无对应后续观测。带状范围：${interval}。${finite(coverage) ? `独立测试集覆盖率 ${fmt(coverage * 100, 1)}%。` : ""}这不是当前时刻的实时预报。`
      : `本图展示一个留出测试窗口的${observationLabel}与六小时模型输出。下方指标覆盖全部测试窗口与全部河段，不仅是当前显示的这一个窗口。`);
    setText("mae-label", "MAE (" + feature.unit + ")");
    setText("rmse-label", "RMSE (" + feature.unit + ")");
    setText("metrics-note", `${feature.name} · 全部测试窗口和 ${river.reaches?.length || 5} 个河段汇总 · nRMSE = RMSE / 测试观测极差`);
    setText("metric-caption", `${river.name_zh}${feature.name}测试集误差比较`);
    $("metric-rows").replaceChildren(...MODELS.map((model) => {
      const metrics = river.metrics?.[model.key]?.[feature.key] || {};
      const tr = make("tr");
      const th = make("th");
      th.scope = "row";
      const dot = make("i", undefined, "model-dot");
      dot.style.background = model.color;
      dot.setAttribute("aria-hidden", "true");
      th.append(dot, document.createTextNode(model.name));
      tr.append(th, make("td", fmt(metrics.mae, feature.digits)), make("td", fmt(metrics.rmse, feature.digits)), make("td", fixed(metrics.nse, 3)), make("td", fixed(metrics.nrmse, 4)));
      return tr;
    }));
    renderChart();
    renderChartTable();
  }

  function renderChart() {
    const river = getRiver();
    const feature = getFeature();
    const forecastView = state.view === "forecast";
    const record = forecastView ? river.forecast : river.backtest;
    const svg = $("forecast-chart");
    const historyTimes = forecastView ? (river.history?.timestamps || []) : [];
    const currentTimes = record?.timestamps || [];
    const timestamps = [...historyTimes, ...currentTimes];
    const historyLength = historyTimes.length;
    const width = Math.max(300, Math.round(svg.parentElement.clientWidth || 840));
    const compact = width < 500;
    const height = compact ? 280 : 340;
    const margin = { left: compact ? 55 : 66, right: 12, top: 31, bottom: 42 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren(sv("desc", { id: "chart-description" }, `${river.name_zh}，${river.reaches?.[state.node] || "河段 " + (state.node + 1)}，${feature.name}，单位${feature.unit}。${forecastView ? `24 小时${historyLabel}与 6 小时未来推演，包括三种模型和物理引导模型的经验误差范围。` : `6 小时留出集回测，对比${observationLabel}和三种模型。`}下方提供图表数值表。`));
    if (!record || !timestamps.length) {
      svg.append(sv("text", { x: 15, y: 60, fill: "#64716f", "font-size": 12 }, "当前河流没有可展示的回测数据。"));
      return;
    }
    const lines = [];
    if (forecastView && historyLength) lines.push({ key: "observed", cls: "plot-observed", values: historyTimes.map((_, index) => valueAt(river.history.values, index)), offset: 0 });
    if (!forecastView) lines.push({ key: "observed", cls: "plot-observed", values: currentTimes.map((_, index) => valueAt(record.observed, index)), offset: 0 });
    MODELS.forEach((model) => lines.push({ key: model.key, cls: model.cls, values: currentTimes.map((_, index) => valueAt(record[model.key], index)), offset: historyLength }));
    const lowers = forecastView ? currentTimes.map((_, index) => valueAt(record.lower, index)) : [];
    const uppers = forecastView ? currentTimes.map((_, index) => valueAt(record.upper, index)) : [];
    const values = [...lines.flatMap((line) => line.values), ...lowers, ...uppers].filter(finite);
    if (!values.length) {
      svg.append(sv("text", { x: 15, y: 60, fill: "#64716f", "font-size": 12 }, "当前变量没有有效数值。"));
      return;
    }
    const rawMin = Math.min(...values), rawMax = Math.max(...values);
    const span = Math.max(rawMax - rawMin, Math.abs(rawMax) * 0.08, 0.01);
    const minimum = Math.max(0, rawMin - span * 0.15);
    const maximum = rawMax + span * 0.16;
    const x = (index) => margin.left + index / Math.max(1, timestamps.length - 1) * plotWidth;
    const y = (value) => margin.top + (maximum - value) / (maximum - minimum) * plotHeight;
    const path = (points) => points.map(([px, py], index) => (index ? "L" : "M") + px.toFixed(2) + "," + py.toFixed(2)).join(" ");
    const axis = (value) => {
      if (Math.abs(value) >= 100000) return fmt(value / 1000, 0) + "k";
      if (Math.abs(value) >= 1000) return fmt(value, 0);
      if (Math.abs(value) >= 10) return fmt(value, 1);
      return fmt(value, feature.key === "sediment" ? 3 : 2);
    };
    if (forecastView && historyLength) {
      const divider = x(historyLength - 0.5);
      svg.append(sv("rect", { x: divider, y: margin.top, width: width - margin.right - divider, height: plotHeight, class: "plot-forecast-zone" }));
      svg.append(sv("line", { x1: divider, x2: divider, y1: margin.top, y2: margin.top + plotHeight, class: "plot-divider" }));
      svg.append(sv("text", { x: divider + 8, y: margin.top - 11, class: "plot-area-label" }, "预测"));
      svg.append(sv("text", { x: margin.left, y: margin.top - 11, class: "plot-area-label" }, historyLabel));
    }
    for (let tick = 0; tick <= 4; tick++) {
      const value = minimum + tick / 4 * (maximum - minimum);
      const py = y(value);
      svg.append(sv("line", { x1: margin.left, x2: width - margin.right, y1: py, y2: py, class: "plot-grid" }));
      svg.append(sv("text", { x: margin.left - 10, y: py + 4, "text-anchor": "end", class: "plot-axis-label" }, axis(value)));
    }
    svg.append(sv("text", { x: margin.left - 10, y: 15, "text-anchor": "end", class: "plot-axis-label" }, feature.unit));
    if (forecastView && lowers.every(finite) && uppers.every(finite)) {
      const points = [...lowers.map((value, index) => [x(historyLength + index), y(value)]), ...uppers.map((value, index) => [x(historyLength + index), y(value)]).reverse()];
      svg.append(sv("path", { d: path(points) + " Z", class: "plot-band" }));
    }
    lines.forEach((line) => {
      let segments = [], current = [];
      if (forecastView && line.key !== "observed" && historyLength) {
        const anchor = valueAt(river.history.values, historyLength - 1);
        if (finite(anchor)) current.push([x(historyLength - 1), y(anchor)]);
      }
      line.values.forEach((value, index) => {
        if (finite(value)) current.push([x(line.offset + index), y(value)]);
        else if (current.length) { segments.push(current); current = []; }
      });
      if (current.length) segments.push(current);
      segments.forEach((points) => svg.append(sv("path", { d: path(points), class: "plot-line " + line.cls })));
      if (!forecastView) line.values.forEach((value, index) => {
        if (finite(value)) svg.append(sv("circle", { cx: x(index), cy: y(value), r: line.key === "observed" ? 3 : 2.2, fill: line.key === "observed" ? "#999e8a" : MODELS.find((model) => model.key === line.key).color }));
      });
    });
    const labelCount = compact ? 3 : 5;
    const indices = [...new Set(Array.from({ length: labelCount }, (_, index) => Math.round(index / (labelCount - 1) * (timestamps.length - 1))))];
    indices.forEach((index) => {
      const iso = time(timestamps[index], true);
      const label = sv("text", { x: x(index), y: height - 23, "text-anchor": index === 0 ? "start" : index === timestamps.length - 1 ? "end" : "middle", class: "plot-time-label" });
      label.append(sv("tspan", { x: x(index) }, iso.slice(0, 5)), sv("tspan", { x: x(index), dy: 13 }, iso.slice(6)));
      svg.append(label);
    });
  }

  function renderChartTable() {
    const river = getRiver();
    const forecastView = state.view === "forecast";
    const record = forecastView ? river.forecast : river.backtest;
    const feature = getFeature();
    const table = $("chart-table");
    const caption = make("caption", `${river.name_zh} · ${river.reaches?.[state.node] || "河段 " + (state.node + 1)} · ${feature.name} (${feature.unit})`);
    const head = make("thead"), headRow = make("tr");
    const labels = ["时间 (UTC)", ...(forecastView ? [] : [observationLabel]), ...MODELS.map((model) => model.name), ...(forecastView ? ["区间下限", "区间上限"] : [])];
    labels.forEach((label) => { const th = make("th", label); th.scope = "col"; headRow.append(th); });
    head.append(headRow);
    const body = make("tbody");
    (record?.timestamps || []).forEach((stamp, index) => {
      const tr = make("tr");
      const th = make("th", time(stamp)); th.scope = "row"; tr.append(th);
      if (!forecastView) tr.append(make("td", fmt(valueAt(record.observed, index), feature.digits)));
      MODELS.forEach((model) => tr.append(make("td", fmt(valueAt(record[model.key], index), feature.digits))));
      if (forecastView) ["lower", "upper"].forEach((key) => tr.append(make("td", fmt(valueAt(record[key], index), feature.digits))));
      body.append(tr);
    });
    table.replaceChildren(caption, head, body);
  }

  function renderComparison() {
    $("comparison-dimensions").replaceChildren(...(data.comparison?.dimensions || []).map((dimension) => {
      const tr = make("tr");
      const th = make("th", dimension.label); th.scope = "row";
      tr.append(th, make("td", dimension.yellow_river), make("td", dimension.mississippi));
      return tr;
    }));
    const directions = data.transfer?.directions || [];
    const name = (id) => data.rivers.find((river) => river.id === id)?.name_zh || id;
    $("transfer-results").replaceChildren(...directions.map((direction) => {
      const card = make("article", undefined, "transfer-card");
      card.append(make("p", "跨流域迁移 / " + fmt(direction.few_shot_fraction * 100, 1) + "% 目标训练窗口", "eyebrow"));
      const heading = make("h4");
      heading.append(document.createTextNode(name(direction.source)), make("span", "→"), document.createTextNode(name(direction.target)));
      card.append(heading);
      const table = make("table"), caption = make("caption", `${name(direction.source)}至${name(direction.target)}的平均归一化误差比较`, "visually-hidden");
      const head = make("thead"), row = make("tr");
      ["训练方式", "相对误差", "nRMSE 均值 ± 标准差"].forEach((label) => { const th = make("th", label); th.scope = "col"; row.append(th); });
      head.append(row); const body = make("tbody");
      const summaries = direction.summary || [];
      const maximum = Math.max(...summaries.map((item) => finite(item.nrmse_mean) ? item.nrmse_mean : 0), 0.0001);
      ["target_only", "zero_shot", "fine_tuned"].forEach((method) => {
        const summary = summaries.find((item) => item.method === method) || {};
        const tr = make("tr"), th = make("th", METHODS[method]); th.scope = "row";
        const barCell = make("td", undefined, "transfer-bar-cell"), bar = make("span", undefined, "transfer-bar"), fill = make("i");
        barCell.setAttribute("aria-hidden", "true");
        fill.style.width = (finite(summary.nrmse_mean) ? Math.max(0, summary.nrmse_mean) / maximum * 100 : 0) + "%";
        bar.append(fill); barCell.append(bar);
        tr.append(th, barCell, make("td", fixed(summary.nrmse_mean, 4) + " ± " + fixed(summary.nrmse_std, 4))); body.append(tr);
      });
      table.append(caption, head, body); card.append(table);
      card.append(make("p", direction.conclusion || "尚无可比较的迁移结果。"));
      return card;
    }));
    if (!directions.length) $("transfer-results").append(make("p", "本次实验未运行比较学习。运行包含迁移阶段的实验后，双向结果会显示在这里。"));
    const seeds = data.transfer?.seeds || [...new Set(directions.flatMap((direction) => direction.rows?.map((row) => row.seed) || []))];
    const scaling = data.transfer?.scaling || "零目标监督迁移仍使用目标训练时段的无标签输入校准缩放器。";
    const validation = data.transfer?.validation || "少样本比例仅指监督训练窗口预算；模型选择使用完整目标验证集，验证标签另计。";
    setText("transfer-note", `nRMSE 为 RMSE 除以目标测试观测极差；表中汇总三个变量的 nRMSE，越小越好。${seeds.length ? `随机种子：${seeds.join("、")}；± 为种子间标准差。` : ""}${scaling}${validation}迁移预训练额外使用源河监督数据与计算。`);
    $("lessons").replaceChildren(...(data.comparison?.lessons || []).map((lesson, index) => {
      const article = make("article");
      article.append(make("span", String(index + 1).padStart(2, "0"), "lesson-number"), make("h4", lesson.title), make("p", lesson.body));
      return article;
    }));
  }

  function renderMethod() {
    const protocol = data.protocol || {};
    const items = [
      ["历史窗口 → 预测窗口", `${protocol.history_hours ?? 24} h → ${protocol.horizon_hours ?? 6} h`],
      ["数据切分", "70% / 15% / 15% · 时间顺序"],
      ["每河序列长度", fmt(protocol.hours, 0) + " 小时"],
      ["训练轮数 / 随机种子", `${protocol.epochs ?? "—"} 轮 / ${protocol.seed ?? "—"}`],
      ["输入时间范围 (UTC)", time(protocol.start).replace(" UTC", "") + " 至 " + time(protocol.end).replace(" UTC", "")]
    ];
    $("protocol").replaceChildren(...items.map(([label, value]) => {
      const block = make("div"), dl = make("dl"); dl.append(make("dt", label), make("dd", value)); block.append(dl); return block;
    }));
    $("limitations").replaceChildren(...(data.limitations || []).map((limitation) => make("li", limitation)));
    const docs = data.docs || data.downloads || {};
    const labels = { study: "多河流研究说明", comparison: "黄河与密西西比河比较", results: "实验结果与误差报告", publishing: "网站发布指南" };
    $("project-downloads").replaceChildren();
    Object.entries(labels).forEach(([key, label]) => {
      const link = resourceLink(label, docs[key], "resource-link", false);
      if (link) $("project-downloads").append(link);
    });
    $("sources").replaceChildren(...(data.sources || []).map((source, index) => {
      const li = make("li"), wrapper = make("div");
      li.id = "source-" + index;
      try {
        const url = new URL(source.url);
        if (!/^https?:$/.test(url.protocol)) throw new Error("Unsupported source URL");
        const link = make("a", source.title || url.hostname);
        link.href = url.href; link.target = "_blank"; link.rel = "noopener noreferrer";
        link.append(make("span", "↗", "external"));
        const span = make("span", url.hostname, "source-domain");
        wrapper.append(link, span);
      } catch (_) {
        wrapper.append(make("span", source.title || "资料来源"));
      }
      if (source.supports) wrapper.append(make("span", source.supports, "source-support"));
      li.append(wrapper); return li;
    }));
  }
  renderComparison();
  renderMethod();
  renderRiver();
  let resizeTimer;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(renderChart, 120);
  });
})();
