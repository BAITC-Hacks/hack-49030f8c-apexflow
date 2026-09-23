import { roles, canonicalGid, rankNodes, csvFor, selectGraph, edgePath, clusterColor } from "./model.mjs";

(async () => {
  "use strict";

  const response = await fetch(new URL("./demo-data.json", import.meta.url));
  if (!response.ok) throw new Error("Не удалось загрузить синтетический набор.");
  const sample = await response.json();
  if (sample.synthetic !== true) throw new Error("Ожидался явно синтетический набор.");
  const { nodes, edges, clusters } = sample;
  const top = rankNodes(nodes);
  let selectedGid = top[0].gid;

  const $ = (selector) => document.querySelector(selector);
  const gidSearch = $("#gid-search");
  const message = $("#search-message");
  const roleFilter = $("#role-filter");
  const clusterFilter = $("#cluster-filter");
  const number = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });

  const formatKzt = (value) => `${number.format(value)} KZT`;
  const nodeFor = (gid) => nodes.find((node) => node.gid === gid);
  const create = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  };

  function populateFilters() {
    roleFilter.append(new Option("Все роли", "all"));
    Object.entries(roles).forEach(([code, role]) => roleFilter.append(new Option(role.label, code)));
    clusterFilter.append(new Option("Все кластеры", "all"));
    clusters.forEach((cluster) => clusterFilter.append(new Option(`Кластер ${cluster.id}`, String(cluster.id))));
  }

  function renderStats() {
    $("#node-count").textContent = nodes.length;
    $("#edge-count").textContent = edges.length;
    $("#seed-count").textContent = nodes.filter((node) => node.isSeed).length;
    $("#cluster-count").textContent = clusters.length;
  }

  function renderTop() {
    const filtered = top.filter((node) =>
      (roleFilter.value === "all" || node.role === roleFilter.value) &&
      (clusterFilter.value === "all" || String(node.cluster) === clusterFilter.value),
    );
    $("#top-count").textContent = `${filtered.length} из ${top.length}`;
    const table = $("#top-table");
    table.replaceChildren();
    if (!filtered.length) {
      const empty = create("tr");
      const cell = create("td", "По текущему фильтру строк нет. Поиск по полному набору доступен.");
      cell.colSpan = 5; empty.append(cell); table.append(empty);
    }
    filtered.forEach((node) => {
      const row = create("tr");
      if (node.gid === selectedGid) row.classList.add("selected");
      row.tabIndex = 0;
      row.setAttribute("aria-label", `Открыть gid ${node.gid}`);
      row.append(create("td", String(node.rank)));
      row.append(create("td", node.gid));
      const roleCell = create("td");
      const roleLabel = create("span", undefined, "role-cell");
      const dot = create("i", undefined, "role-dot");
      dot.style.background = roles[node.role].color;
      roleLabel.append(dot, document.createTextNode(roles[node.role].label));
      roleCell.append(roleLabel);
      row.append(roleCell);
      row.append(create("td", node.priority.toFixed(2)));
      row.append(create("td", node.evidence));
      row.addEventListener("click", () => selectNode(node.gid, "Узел выбран из глобального топа."));
      row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(node.gid, "Узел выбран из глобального топа."); } });
      table.append(row);
    });
    const hidden = !filtered.some(node => node.gid === selectedGid);
    $("#top-count").textContent += hidden ? " · выбранный gid вне фильтра; карточка доступна" : "";
  }

  function renderCase() {
    const node = nodeFor(selectedGid);
    const incoming = edges.filter((edge) => edge.dst === selectedGid);
    const outgoing = edges.filter((edge) => edge.src === selectedGid);
    $("#selected-gid").textContent = node.gid;
    $("#role-badge").textContent = roles[node.role].label;
    $("#role-badge").style.color = roles[node.role].color;
    $("#evidence").textContent = node.evidence;
    $("#role-score").textContent = node.roleScore.toFixed(3);
    $("#priority-score").textContent = node.priority.toFixed(3);
    $("#cluster-id").textContent = String(node.cluster);
    $("#depth-seed").textContent = `${node.depth} / ${node.isSeed ? "да" : "нет"}`;
    const restrictions = [];
    const nextSteps = [];
    if (node.depth === 4) { restrictions.push("Выборка обрывается на четвёртом колене."); nextSteps.push("Запросить последующие исходящие операции за пределами глубины текущей выгрузки."); }
    if (node.isSeed) { restrictions.push("Входящие операции извне наблюдаемого графа могут быть неполными даже за тот же период."); nextSteps.push("Запросить полные входящие операции за наблюдаемый период, включая плательщиков вне выгруженного графа."); }
    if (!incoming.length && !outgoing.length) {
      restrictions.push("Наблюдаемых связей нет.");
      nextSteps.push("Уточнить полноту выгрузки и наличие операций вне наблюдаемого периода.");
    }
    if (node.role === "transit") nextSteps.push("Сопоставить даты доступных входящих и исходящих операций до вывода о последовательности потоков.");
    if (!nextSteps.length) nextSteps.push("Проверить показанные операции и принять решение об углублённой проверке у специалиста.");
    $("#next-step").textContent = `${restrictions.join(" ")} ${nextSteps.join(" ")}`.trim();
    renderEdges(incoming, outgoing);
    renderCluster(node.cluster);
  }

  function renderEdges(incoming, outgoing) {
    const renderTable = (holder, rows, counterpart, heading) => {
      holder.replaceChildren();
      if (!rows.length) { holder.append(create("p", "Наблюдаемых связей нет.", "empty-state")); return; }
      const table = create("table", undefined, "mini-table");
      const head = document.createElement("thead");
      const headRow = document.createElement("tr");
      [heading, "Сумма, KZT", "Операций"].forEach((value) => headRow.append(create("th", value)));
      head.append(headRow); table.append(head);
      const body = document.createElement("tbody");
      [...rows].sort((a, b) => b.sum - a.sum).forEach((edge) => {
        const row = document.createElement("tr");
        row.append(create("td", edge[counterpart]));
        row.append(create("td", number.format(edge.sum)));
        row.append(create("td", String(edge.nTx)));
        body.append(row);
      });
      table.append(body); holder.append(table);
    };
    renderTable($("#incoming"), incoming, "src", "src");
    renderTable($("#outgoing"), outgoing, "dst", "dst");
  }

  function renderCluster(clusterId) {
    const cluster = clusters.find((item) => item.id === clusterId);
    const holder = $("#cluster-summary");
    holder.replaceChildren();
    holder.append(create("strong", `Кластер ${cluster.id}`));
    holder.append(create("p", `${cluster.nNodes} узлов · ${cluster.nSeed} seed · внутренний оборот ${formatKzt(cluster.sum)}`));
    holder.append(create("p", cluster.hypothesis));
    cluster.topGids.forEach(gid => {
      const button = create("button", gid, "button button-secondary");
      button.addEventListener("click", () => selectNode(gid, `Открыт gid ${gid} из кластера.`));
      holder.append(button);
    });
  }

  function renderGraph() {
    const { nodes: graphNodes, edges: visibleEdges } = selectGraph(nodes, edges, selectedGid, $("#graph-mode").value);
    const colorBy = $("#graph-color").value;
    const color = node => colorBy === "role" ? roles[node.role].color : clusterColor(node.cluster);
    const svg = $("#network");
    svg.replaceChildren();
    const ns = "http://www.w3.org/2000/svg";
    const svgElement = (tag, attributes = {}) => {
      const element = document.createElementNS(ns, tag);
      Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
      return element;
    };
    const defs = svgElement("defs");
    const marker = svgElement("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
    marker.append(svgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#627278" })); defs.append(marker); svg.append(defs);
    const positions = new Map();
    if (graphNodes.length === 1) positions.set(selectedGid, [480, 210]);
    else graphNodes.forEach((node, index) => {
      const angle = -Math.PI / 2 + index * (2 * Math.PI / graphNodes.length);
      positions.set(node.gid, [480 + 285 * Math.cos(angle), 210 + 130 * Math.sin(angle)]);
    });
    visibleEdges.forEach((edge) => {
      const [x1, y1] = positions.get(edge.src); const [x2, y2] = positions.get(edge.dst);
      const reciprocal = visibleEdges.some(other => other.src === edge.dst && other.dst === edge.src);
      const line = svgElement("path", { d: edgePath(edge.src, edge.dst, [x1,y1], [x2,y2], reciprocal), fill: "none", stroke: "#627278", "stroke-width": 2, "marker-end": "url(#arrow)" });
      line.append(createSvgTitle(`${edge.src} → ${edge.dst}; ${formatKzt(edge.sum)}; операций: ${edge.nTx}`)); svg.append(line);
    });
    graphNodes.forEach((node) => {
      const [x, y] = positions.get(node.gid); const group = svgElement("g");
      const circle = svgElement("circle", { cx: x, cy: y, r: node.gid === selectedGid ? 32 : 29, fill: color(node), stroke: node.gid === selectedGid ? "#142229" : "#fff", "stroke-width": node.gid === selectedGid ? 5 : 3 });
      if (node.isSeed) group.append(svgElement("circle", { cx:x, cy:y, r:39, fill:"none", stroke:"#b8860b", "stroke-width":2, "stroke-dasharray":"5 3" }));
      group.append(createSvgTitle(`gid: ${node.gid}\nРоль: ${roles[node.role].label}\nКластер: ${node.cluster}\ndepth: ${node.depth}; seed: ${node.isSeed ? "да" : "нет"}`));
      group.append(circle);
      const label = svgElement("text", { x, y: y + 55, "text-anchor": "middle", fill: "#142229", "font-size": 12, "font-family": "system-ui, sans-serif" });
      label.textContent = node.gid;
      group.setAttribute("role", "button"); group.setAttribute("tabindex", "0");
      group.setAttribute("aria-label", `Открыть gid ${node.gid}`);
      group.addEventListener("click", () => selectNode(node.gid, "Узел выбран на графе."));
      group.addEventListener("keydown", event => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); selectNode(node.gid, "Узел выбран на графе."); } });
      group.append(label); svg.append(group);
    });
    $("#graph-caption").textContent = `Показано ${graphNodes.length} узлов и ${visibleEdges.length} наблюдаемых рёбер. Полные gid доступны в карточке и таблицах.`;
    const legend = $("#color-legend"); legend.replaceChildren();
    const keys = new Set();
    graphNodes.forEach(node => {
      const key = colorBy === "role" ? node.role : node.cluster;
      if (keys.has(key)) return; keys.add(key);
      const item = create("span");
      const dot = create("i"); dot.style.background = color(node);
      item.append(dot, document.createTextNode(colorBy === "role" ? roles[node.role].label : `Кластер ${node.cluster}`));
      legend.append(item);
    });
  }

  function createSvgTitle(text) { const title = document.createElementNS("http://www.w3.org/2000/svg", "title"); title.textContent = text; return title; }

  function selectNode(gid, successMessage) {
    selectedGid = gid;
    message.textContent = successMessage;
    message.classList.remove("error");
    renderAll();
  }

  function download(kind) {
    const blob = new Blob([csvFor(kind, nodes, clusters)], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = `${kind}.csv`; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }

  function renderAll() { renderTop(); renderCase(); renderGraph(); }
  $("#search-button").addEventListener("click", () => {
    const gid = canonicalGid(gidSearch.value);
    if (!gid) { message.textContent = "gid должен быть целым числом без дробной части или экспоненты."; message.classList.add("error"); return; }
    if (!nodeFor(gid)) { message.textContent = "Такого gid нет в загруженной выборке."; message.classList.add("error"); return; }
    selectNode(gid, `Найден gid ${gid}.`);
  });
  gidSearch.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); $("#search-button").click(); } });
  roleFilter.addEventListener("change", renderTop);
  clusterFilter.addEventListener("change", renderTop);
  $("#graph-mode").addEventListener("change", renderGraph);
  $("#graph-color").addEventListener("change", renderGraph);
  document.querySelectorAll("[data-download]").forEach((button) => button.addEventListener("click", () => download(button.dataset.download)));

  populateFilters(); renderStats(); renderAll();
})().catch((error) => {
  const status = document.querySelector("#search-message");
  status.textContent = error.message + " Обновите страницу после проверки файлов сайта.";
  status.classList.add("error");
  document.querySelector("#search-button").disabled = true;
});
