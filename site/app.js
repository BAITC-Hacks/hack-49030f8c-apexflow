(() => {
  "use strict";

  const roles = {
    consolidator: { label: "Признаки консолидации", color: "#2563eb" },
    transit: { label: "Признаки транзита", color: "#7c3aed" },
    distributor: { label: "Признаки распределения", color: "#ea580c" },
    terminal: { label: "Предполагаемый конечный получатель", color: "#059669" },
    coordinator: { label: "Структурный кандидат на координацию", color: "#be123c" },
    peripheral: { label: "Недостаточно выраженные признаки", color: "#64748b" },
  };

  const nodes = [
    { gid: "9007199254740993", depth: 0, isSeed: true, role: "coordinator", roleScore: .73, priority: .82, cluster: 1, evidence: "Синтетический пример: стартовый узел с наблюдаемыми исходящими связями." },
    { gid: "10000000000000002", depth: 1, isSeed: false, role: "consolidator", roleScore: .86, priority: .94, cluster: 1, evidence: "Синтетический пример: несколько входящих и исходящих контрагентов." },
    { gid: "10000000000000003", depth: 2, isSeed: false, role: "transit", roleScore: .77, priority: .88, cluster: 1, evidence: "Синтетический пример: входящий и исходящий поток в окружении." },
    { gid: "10000000000000004", depth: 3, isSeed: false, role: "distributor", roleScore: .68, priority: .74, cluster: 1, evidence: "Синтетический пример: один источник и несколько направлений проверки." },
    { gid: "10000000000000005", depth: 4, isSeed: false, role: "terminal", roleScore: .81, priority: .79, cluster: 1, evidence: "Синтетический пример: узел на границе глубины выборки." },
    { gid: "10000000000000006", depth: 2, isSeed: false, role: "peripheral", roleScore: .22, priority: .31, cluster: 1, evidence: "Синтетический пример: малое наблюдаемое окружение." },
    { gid: "10000000000000007", depth: 1, isSeed: false, role: "peripheral", roleScore: .05, priority: .08, cluster: 2, evidence: "Синтетический пример: изолированный узел без наблюдаемых связей." },
  ];
  const edges = [
    { src: "9007199254740993", dst: "10000000000000002", sum: 125000, nTx: 4 },
    { src: "10000000000000002", dst: "10000000000000003", sum: 82000, nTx: 2 },
    { src: "10000000000000002", dst: "10000000000000004", sum: 39000, nTx: 1 },
    { src: "10000000000000003", dst: "10000000000000005", sum: 76000, nTx: 3 },
    { src: "10000000000000006", dst: "10000000000000002", sum: 15000, nTx: 1 },
  ];
  const clusters = [
    { id: 1, nNodes: 6, nSeed: 1, sum: 337000, topGids: ["10000000000000002", "10000000000000003", "9007199254740993"], hypothesis: "Синтетическая группа со связанными переводами; не является результатом анализа." },
    { id: 2, nNodes: 1, nSeed: 0, sum: 0, topGids: ["10000000000000007"], hypothesis: "Синтетический изолят без наблюдаемых связей." },
  ];
  const top = [...nodes].sort((a, b) => b.priority - a.priority || a.gid.localeCompare(b.gid)).map((node, index) => ({ ...node, rank: index + 1 }));
  let selectedGid = top[0].gid;

  const $ = (selector) => document.querySelector(selector);
  const gidSearch = $("#gid-search");
  const message = $("#search-message");
  const roleFilter = $("#role-filter");
  const clusterFilter = $("#cluster-filter");
  const number = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });

  const canonicalGid = (value) => {
    const trimmed = String(value).trim();
    if (!/^[+-]?\d+$/.test(trimmed)) return null;
    return BigInt(trimmed).toString();
  };
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
      row.addEventListener("click", () => selectNode(node.gid, "Узел выбран из глобального топа."));
      row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(node.gid, "Узел выбран из глобального топа."); } });
      table.append(row);
    });
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
    if (node.isSeed) { restrictions.push("Входящие операции до начала выборки могут быть неполными."); nextSteps.push("Уточнить входящие операции, предшествующие текущей выборке."); }
    if (!incoming.length && !outgoing.length) restrictions.push("Наблюдаемых связей нет; это не ошибка визуализации.");
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
    holder.append(create("p", `top_gids: ${cluster.topGids.join(", ")}`));
  }

  function renderGraph() {
    const selected = nodeFor(selectedGid);
    const visibleEdges = edges.filter((edge) => edge.src === selectedGid || edge.dst === selectedGid);
    const visibleIds = [...new Set([selectedGid, ...visibleEdges.flatMap((edge) => [edge.src, edge.dst])])];
    const graphNodes = visibleIds.map(nodeFor).filter(Boolean);
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
      const distance = Math.hypot(x2 - x1, y2 - y1) || 1; const radius = 31;
      const line = svgElement("line", { x1: x1 + radius * (x2 - x1) / distance, y1: y1 + radius * (y2 - y1) / distance, x2: x2 - (radius + 7) * (x2 - x1) / distance, y2: y2 - (radius + 7) * (y2 - y1) / distance, stroke: "#627278", "stroke-width": 2, "marker-end": "url(#arrow)" });
      line.append(createSvgTitle(`${edge.src} → ${edge.dst}; ${formatKzt(edge.sum)}; операций: ${edge.nTx}`)); svg.append(line);
    });
    graphNodes.forEach((node) => {
      const [x, y] = positions.get(node.gid); const group = svgElement("g");
      const circle = svgElement("circle", { cx: x, cy: y, r: node.gid === selectedGid ? 32 : 29, fill: roles[node.role].color, stroke: node.gid === selectedGid ? "#142229" : node.isSeed ? "#f7bc22" : "#fff", "stroke-width": node.gid === selectedGid ? 5 : 3 });
      if (node.isSeed && node.gid !== selectedGid) circle.setAttribute("stroke-dasharray", "5 3");
      group.append(createSvgTitle(`gid: ${node.gid}\nРоль: ${roles[node.role].label}\nКластер: ${node.cluster}\ndepth: ${node.depth}; seed: ${node.isSeed ? "да" : "нет"}`));
      group.append(circle);
      const label = svgElement("text", { x, y: y + 4, "text-anchor": "middle", fill: "#fff", "font-size": 10, "font-family": "system-ui, sans-serif" });
      label.textContent = node.gid.length > 16 ? `${node.gid.slice(0, 7)}…${node.gid.slice(-6)}` : node.gid;
      group.append(label); svg.append(group);
    });
    $("#graph-caption").textContent = visibleEdges.length ? `Показано ${graphNodes.length} узлов и ${visibleEdges.length} наблюдаемых рёбер. Полные gid доступны в карточке и таблицах.` : "Наблюдаемых связей нет: показан сам найденный узел.";
  }

  function createSvgTitle(text) { const title = document.createElementNS("http://www.w3.org/2000/svg", "title"); title.textContent = text; return title; }

  function selectNode(gid, successMessage) {
    selectedGid = gid;
    message.textContent = successMessage;
    message.classList.remove("error");
    renderAll();
  }

  function csvFor(kind) {
    if (kind === "nodes_roles") return ["gid,role,role_score,cluster_id,priority_score,evidence", ...nodes.map((node) => [node.gid, node.role, node.roleScore, node.cluster, node.priority, `"${node.evidence}"`].join(","))].join("\n");
    if (kind === "clusters") return ["cluster_id,n_nodes,n_seed,sum_kzt_internal,top_gids,hypothesis", ...clusters.map((cluster) => [cluster.id, cluster.nNodes, cluster.nSeed, cluster.sum, `"${JSON.stringify(cluster.topGids).replaceAll('"', '""')}"`, `"${cluster.hypothesis}"`].join(","))].join("\n");
    return ["rank,gid,role,priority_score,why", ...top.map((node) => [node.rank, node.gid, node.role, node.priority, `"${node.evidence}"`].join(","))].join("\n");
  }

  function download(kind) {
    const blob = new Blob([csvFor(kind)], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = `${kind}.csv`; link.click();
    URL.revokeObjectURL(link.href);
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
  document.querySelectorAll("[data-download]").forEach((button) => button.addEventListener("click", () => download(button.dataset.download)));

  populateFilters(); renderStats(); renderAll();
})();
