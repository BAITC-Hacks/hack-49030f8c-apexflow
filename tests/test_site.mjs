import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { canonicalGid, rankNodes, selectGraph, csvFor, csvCell, edgePath } from "../site/model.mjs";

const sample = JSON.parse(readFileSync(new URL("../site/demo-data.json", import.meta.url), "utf8"));
test("exact signed int64 search, no numeric round trip", () => {
  assert.equal(canonicalGid(" 9007199254740993 "), "9007199254740993");
  for (const value of ["1.0", "1e3", "9223372036854775808", true, 9007199254740993, ""]) assert.equal(canonicalGid(value), null);
});
test("tie-breaks sort gids numerically", () => {
  assert.deepEqual(rankNodes([{gid:"10", priority:1},{gid:"2", priority:1},{gid:"-1",priority:1}]).map(n=>n.gid), ["-1","2","10"]);
});
test("isolate stays searchable; cluster includes every member", () => {
  const graph = selectGraph(sample.nodes, sample.edges, "10000000000000007");
  assert.equal(graph.nodes.length, 1);
  assert.equal(graph.edges.length, 0);
  assert.equal(selectGraph(sample.nodes, sample.edges, "10000000000000002", "cluster").nodes.length, 6);
});
test("CSV escaping and JSON integer tokens preserve identifiers", () => {
  assert.equal(csvCell('one,"two"\nthree'), '"one,""two""\nthree"');
  const csv = csvFor("clusters", sample.nodes, sample.clusters);
  assert.ok(csv.includes('"[10000000000000002,10000000000000003,9007199254740993,'));
  assert.ok(!csv.includes('[""10000000000000002""'));
  assert.equal(csvFor("nodes_roles", sample.nodes, sample.clusters).split("\r\n").length, sample.nodes.length + 2);
});
test("synthetic cluster totals match observed edges", () => {
  for (const cluster of sample.clusters) {
    const members = sample.nodes.filter(n => n.cluster === cluster.id);
    const ids = new Set(members.map(n=>n.gid));
    assert.equal(cluster.nNodes, members.length);
    assert.equal(cluster.nSeed, members.filter(n=>n.isSeed).length);
    assert.equal(cluster.sum, sample.edges.filter(e=>ids.has(e.src)&&ids.has(e.dst)).reduce((sum,e)=>sum+e.sum,0));
    assert.deepEqual(cluster.topGids, rankNodes(members).slice(0,cluster.topGids.length).map(n=>n.gid));
  }
  assert.ok(sample.nodes.filter(n=>n.depth===4).every(n=>n.role!=="terminal"));
});
test("directions use separate reciprocal arcs and a nonzero loop", () => {
  assert.ok(edgePath("1","2",[0,0],[100,0],true).includes("Q"));
  assert.notEqual(edgePath("1","2",[0,0],[100,0],true), edgePath("2","1",[100,0],[0,0],true));
  assert.ok(edgePath("1","1",[20,20],[20,20],false).includes("C"));
});
