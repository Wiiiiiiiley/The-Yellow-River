/* The scenario plot reads the displayed table, keeping one source of values. */
(() => {
  "use strict";
  const svg = document.getElementById("forecast-chart");
  const rows = [...document.querySelectorAll("#forecast-results .forecast-table tbody tr")];
  if (!svg || rows.length !== 6) return;
  const rivers = [
    {id:"yellow_river", name:"Yellow River", station:"Lijin", shape:"circle"},
    {id:"mississippi", name:"Mississippi River", station:"St. Louis", shape:"rect"}
  ];
  const dates = rows.map((row) => row.dataset.scenarioDate);
  if (dates.some((date) => !/^\d{4}-\d{2}-\d{2}$/.test(date || ""))) return;
  const series = rivers.map((river) => ({...river, values:rows.map((row) => {
    const cell = row.querySelector(`td[data-river="${river.id}"]`);
    return cell ? Number(cell.textContent.trim().replace(/,/g, "")) : NaN;
  })}));
  if (series.some((river) => river.values.some((value) => !Number.isFinite(value) || value < 0))) return;
  const node = (tag, attrs = {}, text) => {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, String(value)));
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const format = (value) => value.toLocaleString("en-US", {minimumFractionDigits:1, maximumFractionDigits:1});
  const description = svg.querySelector("desc").textContent;
  function drawForecast() {
    const width = Math.max(240, Math.round(svg.clientWidth || svg.parentElement.clientWidth || 480));
    const height = 310, compact = width < 420;
    const margin = {left:48, right:64, top:23, bottom:42};
    const plotWidth = width - margin.left - margin.right, plotHeight = height - margin.top - margin.bottom;
    const maximum = Math.max(1000, Math.ceil(Math.max(...series.flatMap((river) => river.values)) / 1000) * 1000);
    const x = (day) => margin.left + day / (dates.length - 1) * plotWidth;
    const y = (value) => margin.top + (1 - value / maximum) * plotHeight;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren(node("desc", {id:"forecast-chart-description"}, description));
    for (let tick = 0; tick <= 4; tick++) {
      const value = maximum * tick / 4;
      svg.append(node("line", {x1:margin.left, x2:width-margin.right, y1:y(value), y2:y(value), class:"forecast-chart-grid"}),
        node("text", {x:margin.left-9, y:y(value)+4, "text-anchor":"end", class:"forecast-chart-tick"}, value.toLocaleString("en-US")));
    }
    const ticks = compact ? [0, 2, 5] : [0, 1, 2, 3, 4, 5];
    ticks.forEach((day) => svg.append(node("text", {x:x(day), y:height-22, "text-anchor":day===0?"start":day===5?"end":"middle", class:"forecast-chart-tick"}, `${Number(dates[day].slice(8, 10))} Jan`)));
    svg.append(node("text", {x:margin.left, y:height-3, class:"forecast-chart-tick"}, "Scenario date · 2027"));
    series.forEach((river) => {
      const path = river.values.map((value, day) => `${day ? "L" : "M"}${x(day).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
      svg.append(node("path", {d:path, class:"forecast-chart-line", "data-forecast-series":river.id}));
      river.values.forEach((value, day) => {
        const position = river.shape === "circle" ? {cx:x(day), cy:y(value), r:3.5} : {x:x(day)-3.5, y:y(value)-3.5, width:7, height:7};
        const point = node(river.shape, {...position, "data-forecast-river":river.id, "data-scenario-date":dates[day], "data-discharge":value});
        point.append(node("title", {}, `${river.name} · ${river.station} · ${dates[day]}: ${format(value)} m³/s (illustrative scenario)`));
        svg.append(point);
      });
      svg.append(node("text", {x:x(5)+9, y:y(river.values[5])+4, class:"forecast-chart-value"}, format(river.values[5])));
    });
  }
  drawForecast();
  if (typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(drawForecast);
    observer.observe(svg.parentElement);
  } else {
    window.addEventListener("resize", drawForecast);
  }
})();

/* A descriptive seasonal-flow chart. The written study stands on its own. */
(() => {
  "use strict";
  const data = window.OBSERVED_RIVERS_DATA;
  const ids = ["yellow_river", "mississippi"];
  const labels = {
    yellow_river: {name: "Yellow River", station: "Lijin"},
    mississippi: {name: "Mississippi River", station: "St. Louis"}
  };
  const monthLabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const $ = (id) => document.getElementById(id);
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const valid = data?.schema_version === 2 && data.data_status === "observed" && ids.every((id) => {
    const river = data.rivers?.[id];
    return river?.frequency === "daily" && river.unit === "m³/s" && Array.isArray(river.observations)
      && river.observations.length > 0 && river.observations.every((row) => /^\d{4}-\d{2}-\d{2}$/.test(row.date) && finite(row.discharge) && row.discharge >= 0);
  });
  if (!valid) return;
  const make = (tag, text, cls) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; };
  const svgNode = (tag, attrs = {}, text) => { const node = document.createElementNS("http://www.w3.org/2000/svg", tag); Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value))); if (text !== undefined) node.textContent = text; return node; };
  const series = ids.map((id) => {
    const river = data.rivers[id], rows = river.observations;
    const mean = rows.reduce((sum, row) => sum + row.discharge, 0) / rows.length;
    const months = Array.from({length:12}, () => ({sum:0, count:0}));
    rows.forEach((row) => { const month = Number(row.date.slice(5, 7)) - 1; if (month >= 0 && month < 12) { months[month].sum += row.discharge; months[month].count++; } });
    return {id, ...labels[id], mean, start:rows[0].date, end:rows.at(-1).date,
      color:id === "yellow_river" ? "#e4bf86" : "#89bdac", values:months.map((month) => month.count && mean > 0 ? month.sum / month.count / mean : null)};
  });
  if (series.some((river) => !(river.mean > 0) || river.values.some((value) => !finite(value)))) return;
  $("flow-error").hidden = true; $("flow-content").hidden = false;
  const period = series[0].start === series[1].start && series[0].end === series[1].end ? `${series[0].start} — ${series[0].end}` : "Station-specific observation periods";
  $("flow-period").textContent = `Lijin and St. Louis stations · ${period}`;
  $("flow-stats").replaceChildren(...series.map((river) => {
    const block = make("div", undefined, "flow-stat");
    block.append(make("h4", river.name + " · " + river.station), make("strong", river.mean.toLocaleString("en-US", {maximumFractionDigits:0}) + " m³/s"), make("p", river.start + " — " + river.end));
    return block;
  }));
  function draw() {
    const svg = $("seasonal-chart"), width = Math.max(300, Math.round(svg.parentElement.clientWidth || 760)), compact = width < 500;
    const height = compact ? 250 : 285, margin = {left:43, right:14, top:28, bottom:33};
    const w = width - margin.left - margin.right, h = height - margin.top - margin.bottom;
    const maximum = Math.max(...series.flatMap((river) => river.values)) * 1.1;
    const x = (month) => margin.left + month / 11 * w, y = (value) => margin.top + (maximum - value) / maximum * h;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren(svgNode("desc", {id:"seasonal-description"}, `Yellow River at Lijin and Mississippi River at St. Louis, ${period}. Each month's mean discharge is divided by that station's mean discharge over the full study period.`));
    for (let tick = 0; tick <= 4; tick++) {
      const value = maximum * tick / 4;
      svg.append(svgNode("line", {x1:margin.left, x2:width-margin.right, y1:y(value), y2:y(value), stroke:"#ffffff1a"}), svgNode("text", {x:margin.left-8, y:y(value)+4, "text-anchor":"end", fill:"#b5c7c0", "font-size":10}, value.toFixed(1)));
    }
    svg.append(svgNode("line", {x1:margin.left, x2:width-margin.right, y1:y(1), y2:y(1), stroke:"#d7ded43f", "stroke-dasharray":"4 5"}));
    const ticks = compact ? [0, 2, 5, 8, 11] : Array.from({length:12}, (_, index) => index);
    ticks.forEach((month) => svg.append(svgNode("text", {x:x(month), y:height-10, "text-anchor":month===0?"start":month===11?"end":"middle", fill:"#b5c7c0", "font-size":10}, monthLabels[month])));
    series.forEach((river) => {
      const path = river.values.map((value, month) => `${month ? "L" : "M"}${x(month).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
      svg.append(svgNode("path", {d:path, fill:"none", stroke:river.color, "stroke-width":2.5, "stroke-linejoin":"round", "stroke-linecap":"round"}));
      river.values.forEach((value, month) => svg.append(svgNode("circle", {cx:x(month), cy:y(value), r:2.6, fill:river.color})));
    });
    svg.append(svgNode("text", {x:margin.left, y:13, fill:"#b5c7c0", "font-size":10}, "Relative to each station's mean discharge"));
  }
  draw();
  let timer;
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(draw, 120); });
})();
