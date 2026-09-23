// Pure view logic shared by the website and its dependency-free Node tests.
export const roles = {
  consolidator: { label: "Признаки консолидации", color: "#2563eb" },
  transit: { label: "Признаки транзита", color: "#7c3aed" },
  distributor: { label: "Признаки распределения", color: "#ea580c" },
  terminal: { label: "Предполагаемый конечный получатель", color: "#059669" },
  coordinator: { label: "Структурный кандидат на координацию", color: "#be123c" },
  peripheral: { label: "Недостаточно выраженных признаков / периферия", color: "#64748b" },
};
export function canonicalGid(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (trimmed.length > 64 || !/^[+-]?[0-9]+$/.test(trimmed)) return null;
  const gid = BigInt(trimmed);
  return gid >= -(2n ** 63n) && gid < 2n ** 63n ? gid.toString() : null;
}
export function compareGids(a, b) {
  return BigInt(a) < BigInt(b) ? -1 : BigInt(a) > BigInt(b) ? 1 : 0;
}
export function rankNodes(nodes) {
  return [...nodes].sort((a, b) => b.priority - a.priority || compareGids(a.gid, b.gid))
    .map((node, i) => ({ ...node, rank: i + 1 }));
}
export function clusterColor(id) {
  const colors = ["#2563eb", "#7c3aed", "#059669", "#ea580c", "#be123c", "#0f766e", "#a16207"];
  return colors[((id % colors.length) + colors.length) % colors.length];
}
export function selectGraph(nodes, edges, gid, mode = "neighborhood") {
  const selected = nodes.find((node) => node.gid === gid);
  if (!selected) throw new Error("Unknown gid");
  const ids = new Set(mode === "cluster" ? nodes.filter((n) => n.cluster === selected.cluster).map((n) => n.gid) : [gid]);
  const visibleEdges = edges.filter((edge) => mode === "cluster"
    ? ids.has(edge.src) && ids.has(edge.dst) : edge.src === gid || edge.dst === gid);
  visibleEdges.forEach((edge) => { ids.add(edge.src); ids.add(edge.dst); });
  return { nodes: nodes.filter((n) => ids.has(n.gid)), edges: visibleEdges };
}
export function edgePath(source, target, from, to, reciprocal) {
  const [x1, y1] = from, [x2, y2] = to;
  if (source === target) return `M ${x1-18} ${y1-23} C ${x1-80} ${y1-100}, ${x1+80} ${y1-100}, ${x1+20} ${y1-28}`;
  const distance = Math.hypot(x2-x1, y2-y1);
  const start = `${x1+33*(x2-x1)/distance} ${y1+33*(y2-y1)/distance}`;
  const end = `${x2-40*(x2-x1)/distance} ${y2-40*(y2-y1)/distance}`;
  return reciprocal
    ? `M ${start} Q ${(x1+x2)/2-38*(y2-y1)/distance} ${(y1+y2)/2+38*(x2-x1)/distance}, ${end}`
    : `M ${start} L ${end}`;
}
export function csvCell(value) {
  return '"' + String(value).replaceAll('"', '""') + '"';
}
export function csvFor(kind, nodes, clusters) {
  let header, rows;
  if (kind === "nodes_roles") {
    header = ["gid","role","role_score","cluster_id","priority_score","evidence"];
    rows = [...nodes].sort((a,b) => compareGids(a.gid,b.gid)).map(n => [n.gid,n.role,n.roleScore,n.cluster,n.priority,n.evidence]);
  } else if (kind === "clusters") {
    header = ["cluster_id","n_nodes","n_seed","sum_kzt_internal","top_gids","hypothesis"];
    rows = [...clusters].sort((a,b) => a.id-b.id).map(c => {
      // JSON integer tokens constructed from validated strings avoid JS rounding.
      if (c.topGids.some(gid => canonicalGid(gid) !== gid)) throw new Error("Invalid top_gids");
      return [c.id,c.nNodes,c.nSeed,c.sum,"[" + c.topGids.join(",") + "]",c.hypothesis];
    });
  } else if (kind === "top_nodes") {
    header = ["rank","gid","role","priority_score","why"];
    rows = rankNodes(nodes).map(n => [n.rank,n.gid,n.role,n.priority,n.evidence]);
  } else throw new Error("Unknown CSV");
  return [header.join(","), ...rows.map(row => row.map(csvCell).join(","))].join("\r\n") + "\r\n";
}
