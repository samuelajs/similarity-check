
(function () {
  "use strict";

  // ---------- Python backend integration (silhouette + room/OCR detection) ----------
  // Both were previously attempted client-side (OpenCV.js distance-transform port for the
  // silhouette, Tesseract.js + a line-for-line port of the wall/geometry logic for rooms).
  // Verified against a colleague's tested Python pipeline (segment_denah.py), the JS versions
  // did not reach the same quality (OCR misreads, unaligned room boxes, silhouette points
  // following interior walls instead of the outer contour) -- root cause suspected in the
  // OpenCV.js/Tesseract.js library behavior, not the ported geometry (which matched Python
  // pixel-for-pixel in isolated tests). Both now call segment_denah.py directly via a small
  // local HTTP server (untuk-sam/demo_server.py) instead of re-implementing them in-browser.
  function getBackendUrl() {
    const input = document.getElementById("backendUrl");
    const raw = (input && input.value || "").trim();
    return (raw || "http://127.0.0.1:8765").replace(/\/+$/, "");
  }

  async function backendSegment(mode, blob, organik, gabung) {
    const url = getBackendUrl() + "/segment?mode=" + mode + (organik ? "&organik=1" : "") + (gabung ? "&gabung=1" : "");
    let response;
    try {
      response = await fetch(url, { method: "POST", body: blob });
    } catch (err) {
      const e = new Error("backend_unreachable");
      e.code = "backend_unreachable";
      throw e;
    }
    const data = await response.json().catch(() => null);
    if (!response.ok || !data) {
      const e = new Error((data && data.error) || ("http_" + response.status));
      e.code = (data && data.error) || "bad_response";
      throw e;
    }
    return data;
  }

  // ---------- geometry core (tested logic) ----------
  function polygonArea(pts) {
    let a = 0;
    for (let i = 0; i < pts.length; i++) {
      const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % pts.length];
      a += x1 * y2 - x2 * y1;
    }
    return Math.abs(a) / 2;
  }
  function centroid(pts) {
    let cx = 0, cy = 0, a = 0;
    for (let i = 0; i < pts.length; i++) {
      const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % pts.length];
      const cross = x1 * y2 - x2 * y1;
      a += cross; cx += (x1 + x2) * cross; cy += (y1 + y2) * cross;
    }
    a *= 0.5;
    if (Math.abs(a) < 1e-9) {
      const n = pts.length;
      return [pts.reduce((s, p) => s + p[0], 0) / n, pts.reduce((s, p) => s + p[1], 0) / n];
    }
    return [cx / (6 * a), cy / (6 * a)];
  }
  function normalize(pts) {
    const [cx, cy] = centroid(pts);
    const centered = pts.map(([x, y]) => [x - cx, y - cy]);
    const area = polygonArea(centered);
    const scale = Math.sqrt(area) || 1;
    return centered.map(([x, y]) => [x / scale, y / scale]);
  }
  function rotate(pts, angleRad) {
    const c = Math.cos(angleRad), s = Math.sin(angleRad);
    return pts.map(([x, y]) => [x * c - y * s, x * s + y * c]);
  }
  function pointInPolygon(pt, poly) {
    const [x, y] = pt;
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }
  function bounds(pts) {
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  }
  function overlapGrid(polyA, polyB, grid) {
    const bA = bounds(polyA), bB = bounds(polyB);
    const minX = Math.min(bA.minX, bB.minX) * 1.15, maxX = Math.max(bA.maxX, bB.maxX) * 1.15;
    const minY = Math.min(bA.minY, bB.minY) * 1.15, maxY = Math.max(bA.maxY, bB.maxY) * 1.15;
    let inBoth = 0, inEither = 0;
    for (let i = 0; i < grid; i++) {
      for (let j = 0; j < grid; j++) {
        const x = minX + (maxX - minX) * (i + 0.5) / grid;
        const y = minY + (maxY - minY) * (j + 0.5) / grid;
        const a = pointInPolygon([x, y], polyA);
        const b = pointInPolygon([x, y], polyB);
        if (a && b) inBoth++;
        if (a || b) inEither++;
      }
    }
    return inEither > 0 ? inBoth / inEither : 0;
  }
  function bestRotationOverlap(polyA, polyB, steps) {
    let best = -1, bestAngle = 0, bestPoly = polyB;
    for (let k = 0; k < steps; k++) {
      const ang = (2 * Math.PI * k) / steps;
      const rotated = rotate(polyB, ang);
      const ov = overlapGrid(polyA, rotated, 55);
      if (ov > best) { best = ov; bestAngle = ang; bestPoly = rotated; }
    }
    const refined = overlapGrid(polyA, bestPoly, 150);
    return { overlap: refined, angleDeg: (bestAngle * 180) / Math.PI, poly: bestPoly };
  }

  // ---------- Elliptical Fourier Descriptors (Kuhl & Giardina, 1982) ----------
  // Direct polygon formula, normalized for size, rotation (1st harmonic axis) and start point.
  function ellipticFourierRaw(pts, order) {
    const K = pts.length;
    const dx = [], dy = [], dt = [];
    for (let p = 0; p < K; p++) {
      const [x0, y0] = pts[p];
      const [x1, y1] = pts[(p + 1) % K];
      dx.push(x1 - x0); dy.push(y1 - y0);
      dt.push(Math.hypot(x1 - x0, y1 - y0) || 1e-9);
    }
    const xi = [0];
    for (let p = 0; p < K; p++) xi.push(xi[p] + dt[p]);
    const T = xi[K];
    const coeffs = [];
    for (let n = 1; n <= order; n++) {
      let a = 0, b = 0, c = 0, d = 0;
      const k = T / (2 * n * n * Math.PI * Math.PI);
      for (let p = 0; p < K; p++) {
        const t1 = 2 * n * Math.PI * xi[p + 1] / T;
        const t0 = 2 * n * Math.PI * xi[p] / T;
        a += (dx[p] / dt[p]) * (Math.cos(t1) - Math.cos(t0));
        b += (dx[p] / dt[p]) * (Math.sin(t1) - Math.sin(t0));
        c += (dy[p] / dt[p]) * (Math.cos(t1) - Math.cos(t0));
        d += (dy[p] / dt[p]) * (Math.sin(t1) - Math.sin(t0));
      }
      coeffs.push([k * a, k * b, k * c, k * d]);
    }
    return coeffs;
  }

  function normalizeEFD(coeffs) {
    const [a1, b1, c1, d1] = coeffs[0];
    const theta1 = 0.5 * Math.atan2(2 * (a1 * b1 + c1 * d1), a1 * a1 - b1 * b1 + c1 * c1 - d1 * d1);
    const rotated = coeffs.map(([a, b, c, d], i) => {
      const n = i + 1;
      const ct = Math.cos(n * theta1), st = Math.sin(n * theta1);
      return [a * ct + b * st, -a * st + b * ct, c * ct + d * st, -c * st + d * ct];
    });
    const [ra1, , rc1] = rotated[0];
    const psi = Math.atan2(rc1, ra1);
    const cp = Math.cos(psi), sp = Math.sin(psi);
    const phased = rotated.map(([a, b, c, d]) => [
      a * cp + c * sp, b * cp + d * sp, -a * sp + c * cp, -b * sp + d * cp,
    ]);
    const scale = Math.hypot(phased[0][0], phased[0][2]) || 1e-9;
    return phased.map(([a, b, c, d]) => [a / scale, b / scale, c / scale, d / scale]);
  }

  function efdDistance(ptsA, ptsB, order) {
    order = order || 10;
    const nA = normalizeEFD(ellipticFourierRaw(ptsA, order));
    const nB = normalizeEFD(ellipticFourierRaw(ptsB, order));
    let sum = 0;
    for (let i = 0; i < order; i++) {
      for (let k = 0; k < 4; k++) sum += (nA[i][k] - nB[i][k]) ** 2;
    }
    return Math.sqrt(sum);
  }

  // Rotation that brings a shape's ORIGINAL points into EFD's own canonical pose.
  // Note: subject to a well-known 180deg ambiguity in first-harmonic EFD normalization
  // (the major axis has no built-in "front"/"back") -- documented, not a bug.
  function efdCanonicalAngle(pts, order) {
    order = order || 10;
    const [a1, b1, c1, d1] = ellipticFourierRaw(pts, order)[0];
    const theta1 = 0.5 * Math.atan2(2 * (a1 * b1 + c1 * d1), a1 * a1 - b1 * b1 + c1 * c1 - d1 * d1);
    const ra1 = a1 * Math.cos(theta1) + b1 * Math.sin(theta1);
    const rc1 = c1 * Math.cos(theta1) + d1 * Math.sin(theta1);
    const psi = Math.atan2(rc1, ra1);
    return -psi;
  }

  // ---------- Component 2: room-topology similarity via Graph Edit Distance ----------
  // Approximate GED by bipartite matching (Riesen & Bunke 2009) -- standard in floor-plan
  // graph comparison literature, always polynomial-time (unlike exact GED, which is
  // NP-hard), at the cost of being an upper bound on the true minimum GED rather than the
  // exact value. Runs entirely client-side on whatever room+adjacency data each panel
  // already has (from Deteksi Ruang or a loaded case) -- no backend round-trip needed,
  // same philosophy as the EFD/overlap computation above.

  // Kuhn-Munkres minimum-cost perfect assignment, O(n^3), square cost matrix.
  function hungarianAssignment(costMatrix) {
    const n = costMatrix.length;
    const INF = Infinity;
    const u = new Array(n + 1).fill(0);
    const v = new Array(n + 1).fill(0);
    const p = new Array(n + 1).fill(0); // p[j] = row (1-indexed) matched to column j
    const way = new Array(n + 1).fill(0);

    for (let i = 1; i <= n; i++) {
      p[0] = i;
      let j0 = 0;
      const minv = new Array(n + 1).fill(INF);
      const used = new Array(n + 1).fill(false);
      do {
        used[j0] = true;
        const i0 = p[j0];
        let delta = INF, j1 = -1;
        for (let j = 1; j <= n; j++) {
          if (!used[j]) {
            const cur = costMatrix[i0 - 1][j - 1] - u[i0] - v[j];
            if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
            if (minv[j] < delta) { delta = minv[j]; j1 = j; }
          }
        }
        for (let j = 0; j <= n; j++) {
          if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
          else { minv[j] -= delta; }
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do {
        const j1 = way[j0];
        p[j0] = p[j1];
        j0 = j1;
      } while (j0);
    }

    const rowToCol = new Array(n).fill(-1);
    for (let j = 1; j <= n; j++) {
      if (p[j] > 0) rowToCol[p[j] - 1] = j - 1;
    }
    let total = 0;
    for (let i = 0; i < n; i++) total += costMatrix[i][rowToCol[i]];
    return { assignment: rowToCol, total };
  }

  function normRoomName(name) { return (name || "").trim().toUpperCase(); }

  // From a room editor's getState() ({rooms, adjacency}) build {nodes, edges} for GED.
  // normArea is each room's polygon-pixel area divided by the SUM OF ALL ROOM POLYGONS
  // CURRENTLY IN THIS LIST (not the building's silhouette/total floor area, which this
  // function never sees, and not a physical unit -- no real-world scale calibration here).
  // This is scale-invariant (no need to know physical dimensions) but relative to the room
  // *list*: deleting a room from the list changes the survivors' fractions too, since the
  // denominator shrinks. That ripple is intentional (see footer note), not a bug.
  function buildRoomGraph(roomState) {
    const rooms = (roomState && roomState.rooms) || [];
    const adjacency = (roomState && roomState.adjacency) || [];
    const areas = rooms.map((r) => polygonArea(r.points || []));
    const totalArea = areas.reduce((s, a) => s + a, 0) || 1;
    const nodes = rooms.map((r, i) => {
      const pts = r.points || [];
      const [cx, cy] = pts.length >= 3 ? centroid(pts) : (pts[0] || [0, 0]);
      return { id: r.id, name: normRoomName(r.name), normArea: areas[i] / totalArea, cx, cy };
    });
    const idSet = new Set(nodes.map((n) => n.id));
    const edges = adjacency.filter((e) => idSet.has(e.a) && idSet.has(e.b)).map((e) => [e.a, e.b]);
    return { nodes, edges };
  }

  function graphEditDistance(graphA, graphB, weights) {
    weights = weights || { name: 0.7, area: 0.3 };
    const n = graphA.nodes.length, m = graphB.nodes.length;
    const size = n + m;
    const BIG = 1e6;
    const matrix = Array.from({ length: size }, () => new Array(size).fill(BIG));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const a = graphA.nodes[i], b = graphB.nodes[j];
        const nameCost = a.name === b.name ? 0 : 1;
        const areaCost = Math.abs(a.normArea - b.normArea);
        matrix[i][j] = weights.name * nameCost + weights.area * areaCost;
      }
    }
    for (let i = 0; i < n; i++) matrix[i][m + i] = 1; // delete A_i
    for (let j = 0; j < m; j++) matrix[n + j][j] = 1; // insert B_j
    for (let j = 0; j < m; j++) for (let i = 0; i < n; i++) matrix[n + j][m + i] = 0; // eps-eps

    const { assignment: rowToCol, total: nodeCost } = size > 0 ? hungarianAssignment(matrix) : { assignment: [], total: 0 };
    const colToRow = new Array(size).fill(-1);
    rowToCol.forEach((c, r) => { colToRow[c] = r; });

    const matches = [], deleted = [], inserted = [];
    const aToB = new Map();
    for (let i = 0; i < n; i++) {
      const c = rowToCol[i];
      if (c < m) { matches.push({ a: graphA.nodes[i].id, b: graphB.nodes[c].id, cost: matrix[i][c] }); aToB.set(i, c); }
      else deleted.push(graphA.nodes[i].id);
    }
    for (let j = 0; j < m; j++) {
      const r = colToRow[j];
      if (r >= n) inserted.push(graphB.nodes[j].id);
    }

    const edgeKey = (x, y) => [x, y].sort().join("|");
    const bIndexOf = new Map(graphB.nodes.map((node, idx) => [node.id, idx]));
    const bEdgesByIndex = graphB.edges.map(([x, y]) => [bIndexOf.get(x), bIndexOf.get(y)]).filter(([x, y]) => x !== undefined && y !== undefined);
    const bEdgeIndexSet = new Set(bEdgesByIndex.map(([x, y]) => edgeKey(x, y)));

    const aIndexOf = new Map(graphA.nodes.map((node, idx) => [node.id, idx]));
    const aEdgesByIndex = graphA.edges.map(([x, y]) => [aIndexOf.get(x), aIndexOf.get(y)]).filter(([x, y]) => x !== undefined && y !== undefined);

    let kept = 0, removed = 0;
    const coveredBEdgeKeys = new Set();
    aEdgesByIndex.forEach(([x, y]) => {
      const bx = aToB.get(x), by = aToB.get(y);
      if (bx !== undefined && by !== undefined && bEdgeIndexSet.has(edgeKey(bx, by))) { kept++; coveredBEdgeKeys.add(edgeKey(bx, by)); }
      else removed++;
    });
    let added = 0;
    bEdgesByIndex.forEach(([x, y]) => { if (!coveredBEdgeKeys.has(edgeKey(x, y))) added++; });

    const edgeCost = removed + added;
    const ged = nodeCost + edgeCost;
    const maxCost = n + m + graphA.edges.length + graphB.edges.length;
    const similarity = maxCost > 0 ? Math.max(0, 1 - ged / maxCost) : 1;

    return { ged, similarity, matches, deleted, inserted, edgeChanges: { kept, removed, added } };
  }

  // ---------- Component 3: space hierarchy via graph centrality ----------
  // Three centrality measures per room, all normalized to [0,1]. These correspond to three
  // DISTINCT measures from Space Syntax theory (Hillier & Hanson) -- an earlier round of this
  // tool incorrectly lumped two of them (Integration and Choice) into one option when asking
  // Sam to choose a methodology; this is the corrected, properly separated version:
  // - "Konektivitas" (Connectivity) = degree centrality -- share of all other rooms this one
  //   connects to directly.
  // - "Integrasi" (Integration) = closeness centrality, Wasserman & Faust's variant for
  //   disconnected graphs (scales down for rooms that can't reach everyone, instead of just
  //   erroring/ignoring them) -- how FEW STEPS on average it takes to reach every other room.
  //   Inspired by Space Syntax's Integration (which is depth/closeness-based) but NOT the
  //   canonical RRA formula, which needs a justified graph rooted at the building's
  //   exterior/entrance -- not available in this tool's data model.
  // - "Choice" = betweenness centrality (Brandes' algorithm, normalized by (n-1)(n-2)/2) --
  //   how often this room lies ON THE SHORTEST PATH between two OTHER rooms, i.e. how much
  //   it functions as a through-route/transit space rather than a destination. Inspired by
  //   Space Syntax's Choice (also betweenness-based) but again the plain graph-theory version,
  //   not Space Syntax software's angular/metric-weighted variants.
  // Integration and Choice are NOT interchangeable: a room can be highly reachable (high
  // Integration) without being a through-route for others (low Choice), and vice versa.
  function buildAdjList(nodes, edges) {
    const adj = new Map(nodes.map((n) => [n.id, []]));
    edges.forEach(([a, b]) => {
      if (adj.has(a) && adj.has(b)) { adj.get(a).push(b); adj.get(b).push(a); }
    });
    return adj;
  }

  function bfsDistances(startId, adj) {
    const dist = new Map([[startId, 0]]);
    const queue = [startId];
    let head = 0;
    while (head < queue.length) {
      const cur = queue[head++];
      const d = dist.get(cur);
      for (const nb of adj.get(cur) || []) {
        if (!dist.has(nb)) { dist.set(nb, d + 1); queue.push(nb); }
      }
    }
    return dist;
  }

  // Brandes' algorithm for betweenness centrality (unweighted, undirected). O(V*(V+E)),
  // exact (not sampled) -- fine at this tool's scale (tens of rooms per plan).
  function betweennessRaw(nodes, adj) {
    const betweenness = new Map(nodes.map((n) => [n.id, 0]));
    nodes.forEach((s) => {
      const S = [];
      const P = new Map(nodes.map((n) => [n.id, []]));
      const sigma = new Map(nodes.map((n) => [n.id, 0]));
      sigma.set(s.id, 1);
      const d = new Map(nodes.map((n) => [n.id, -1]));
      d.set(s.id, 0);
      const queue = [s.id];
      let head = 0;
      while (head < queue.length) {
        const v = queue[head++];
        S.push(v);
        for (const w of adj.get(v) || []) {
          if (d.get(w) < 0) { queue.push(w); d.set(w, d.get(v) + 1); }
          if (d.get(w) === d.get(v) + 1) {
            sigma.set(w, sigma.get(w) + sigma.get(v));
            P.get(w).push(v);
          }
        }
      }
      const delta = new Map(nodes.map((n) => [n.id, 0]));
      while (S.length) {
        const w = S.pop();
        for (const v of P.get(w)) {
          delta.set(v, delta.get(v) + (sigma.get(v) / sigma.get(w)) * (1 + delta.get(w)));
        }
        if (w !== s.id) betweenness.set(w, betweenness.get(w) + delta.get(w));
      }
    });
    nodes.forEach((n) => betweenness.set(n.id, betweenness.get(n.id) / 2)); // undirected: each pair counted from both endpoints
    return betweenness;
  }

  function computeCentralities(nodes, edges) {
    const adj = buildAdjList(nodes, edges);
    const n = nodes.length;
    const betweenness = betweennessRaw(nodes, adj);
    const choiceDenom = (n - 1) * (n - 2) / 2;
    const out = new Map();
    nodes.forEach((node) => {
      const degree = (adj.get(node.id) || []).length;
      const connectivity = n > 1 ? degree / (n - 1) : 0;
      let integration = 0;
      if (n > 1) {
        const dist = bfsDistances(node.id, adj);
        let sumDist = 0, reachable = 0;
        dist.forEach((d, id) => { if (id !== node.id) { sumDist += d; reachable++; } });
        if (reachable > 0 && sumDist > 0) integration = (reachable / (n - 1)) * (reachable / sumDist);
      }
      const choice = choiceDenom > 0 ? betweenness.get(node.id) / choiceDenom : 0;
      out.set(node.id, { connectivity, integration, choice });
    });
    return out;
  }

  // Independent from Component 2's GED matching (own Hungarian run, own cost weights) --
  // per Sam's explicit choice, so hierarchy similarity isn't just riding on the topology
  // pairing. Substitution cost blends room-name identity with all three centrality measures;
  // delete/insert stay at unit cost 1, same convention as Component 2.
  function hierarchySimilarity(graphA, graphB, weights) {
    weights = weights || { name: 0.4, connectivity: 0.2, integration: 0.2, choice: 0.2 };
    const centA = computeCentralities(graphA.nodes, graphA.edges);
    const centB = computeCentralities(graphB.nodes, graphB.edges);
    const n = graphA.nodes.length, m = graphB.nodes.length;
    const size = n + m;
    const BIG = 1e6;
    const matrix = Array.from({ length: size }, () => new Array(size).fill(BIG));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const a = graphA.nodes[i], b = graphB.nodes[j];
        const ca = centA.get(a.id), cb = centB.get(b.id);
        const nameCost = a.name === b.name ? 0 : 1;
        const connCost = Math.abs(ca.connectivity - cb.connectivity);
        const integCost = Math.abs(ca.integration - cb.integration);
        const choiceCost = Math.abs(ca.choice - cb.choice);
        matrix[i][j] = weights.name * nameCost + weights.connectivity * connCost + weights.integration * integCost + weights.choice * choiceCost;
      }
    }
    for (let i = 0; i < n; i++) matrix[i][m + i] = 1;
    for (let j = 0; j < m; j++) matrix[n + j][j] = 1;
    for (let j = 0; j < m; j++) for (let i = 0; i < n; i++) matrix[n + j][m + i] = 0;

    const { assignment: rowToCol, total: cost } = size > 0 ? hungarianAssignment(matrix) : { assignment: [], total: 0 };
    const colToRow = new Array(size).fill(-1);
    rowToCol.forEach((c, r) => { colToRow[c] = r; });

    const matches = [], deleted = [], inserted = [];
    for (let i = 0; i < n; i++) {
      const c = rowToCol[i];
      if (c < m) matches.push({ a: graphA.nodes[i].id, b: graphB.nodes[c].id, cost: matrix[i][c] });
      else deleted.push(graphA.nodes[i].id);
    }
    for (let j = 0; j < m; j++) {
      const r = colToRow[j];
      if (r === -1 || r >= n) inserted.push(graphB.nodes[j].id);
    }

    const maxCost = n + m;
    const similarity = maxCost > 0 ? Math.max(0, 1 - cost / maxCost) : 1;
    return { similarity, cost, matches, deleted, inserted, centralitiesA: centA, centralitiesB: centB };
  }

  // ---------- bubble diagram: room graph laid out to roughly match each room's REAL
  // position in its floor plan (centroid of its edited polygon), scaled/fit into the SVG
  // canvas, then nudged apart just enough that overlapping bubbles stay readable. Standard
  // architectural relationship-diagram convention: bubble size ~ room's share of its own
  // room list, lines = door connections, layout position ~ real spatial arrangement.
  function radiusOfNode(n) {
    return 14 + 46 * Math.sqrt(Math.max(0, n.normArea));
  }

  function geometryAnchoredLayout(nodes, width, height, padding) {
    padding = padding == null ? 50 : padding;
    const pos = new Map();
    if (!nodes.length) return pos;
    const xs = nodes.map((n) => n.cx || 0), ys = nodes.map((n) => n.cy || 0);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
    const availW = Math.max(1, width - padding * 2), availH = Math.max(1, height - padding * 2);
    const scale = Math.min(availW / spanX, availH / spanY);
    const offX = padding + (availW - spanX * scale) / 2;
    const offY = padding + (availH - spanY * scale) / 2;
    nodes.forEach((n) => {
      pos.set(n.id, { x: offX + ((n.cx || 0) - minX) * scale, y: offY + ((n.cy || 0) - minY) * scale });
    });
    return pos;
  }

  // Local-only collision resolution (not a global spring simulation): pushes directly
  // overlapping bubble pairs apart just enough to clear each other, keeping the overall
  // arrangement close to the real floor-plan layout ("kurang lebih" -- roughly, not exactly).
  function relaxOverlaps(nodes, pos, radiusOf, iterations) {
    iterations = iterations || 300;
    for (let iter = 0; iter < iterations; iter++) {
      let moved = false;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = pos.get(nodes[i].id), b = pos.get(nodes[j].id);
          const minDist = radiusOf(nodes[i]) + radiusOf(nodes[j]) + 6;
          let dx = b.x - a.x, dy = b.y - a.y;
          let dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 0.001) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; dist = Math.sqrt(dx * dx + dy * dy) || 0.001; }
          if (dist < minDist) {
            const push = (minDist - dist) / 2;
            dx /= dist; dy /= dist;
            a.x -= dx * push; a.y -= dy * push;
            b.x += dx * push; b.y += dy * push;
            moved = true;
          }
        }
      }
      if (!moved) break;
    }
  }

  // SVG, not canvas: a bitmap canvas gets blurry/pixelated ("pecah") once CSS scales it
  // past its intrinsic resolution -- SVG is native vector markup, so it stays crisp at
  // any panel width or browser zoom level.
  const SVG_NS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  function drawBubbleDiagram(svgRoot, graph) {
    const vb = svgRoot.viewBox && svgRoot.viewBox.baseVal;
    const w = (vb && vb.width) || 480, h = (vb && vb.height) || 420;
    while (svgRoot.firstChild) svgRoot.removeChild(svgRoot.firstChild);
    if (!graph.nodes.length) {
      const t = svgEl("text", { x: w / 2, y: h / 2, "text-anchor": "middle", "font-family": "Space Grotesk, sans-serif", "font-size": "13", fill: "#9aa7ae" });
      t.textContent = "Tidak ada data ruang";
      svgRoot.appendChild(t);
      return;
    }
    const pos = geometryAnchoredLayout(graph.nodes, w, h, 50);
    relaxOverlaps(graph.nodes, pos, radiusOfNode, 300);
    graph.nodes.forEach((n) => {
      const p = pos.get(n.id);
      const r = radiusOfNode(n);
      p.x = Math.max(r, Math.min(w - r, p.x));
      p.y = Math.max(r, Math.min(h - r, p.y));
    });

    const edgeGroup = svgEl("g", { stroke: "rgba(91,106,118,0.5)", "stroke-width": "1.6" });
    graph.edges.forEach(([a, b]) => {
      const pa = pos.get(a), pb = pos.get(b);
      if (!pa || !pb) return;
      edgeGroup.appendChild(svgEl("line", { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y }));
    });
    svgRoot.appendChild(edgeGroup);

    const nodeGroups = new Map();
    graph.nodes.forEach((n, i) => {
      const p = pos.get(n.id);
      const radius = radiusOfNode(n);
      const color = ROOM_PALETTE[i % ROOM_PALETTE.length];
      const g = svgEl("g", { class: "bubble-node" });
      g.appendChild(svgEl("circle", { cx: p.x, cy: p.y, r: radius, fill: color + "55", stroke: color, "stroke-width": "2" }));
      const label = n.name || "(tanpa nama)";
      const text = svgEl("text", {
        x: p.x, y: p.y, "text-anchor": "middle", "dominant-baseline": "middle",
        "font-family": "Space Grotesk, sans-serif", "font-size": "11", fill: "#1c2530",
      });
      text.textContent = label.length > 16 ? label.slice(0, 15) + "…" : label;
      g.appendChild(text);
      svgRoot.appendChild(g);
      nodeGroups.set(n.id, g);
    });
    svgRoot._nodeGroups = nodeGroups;
  }

  // Called from the topology breakdown table on row hover -- lets a user see exactly
  // which two bubbles a "Ruang A / Ruang B" row refers to, across both diagrams at once.
  function setBubbleHighlight(svgRoot, id, on) {
    if (!svgRoot || !svgRoot._nodeGroups || !id) return;
    const g = svgRoot._nodeGroups.get(id);
    if (g) g.classList.toggle("bubble-hl", on);
  }

  // ---------- drawable canvas ----------
  function makeDrawer(canvasId, countId, strokeColor, fillColor, removeBtnId) {
    const canvas = document.getElementById(canvasId);
    const removeBtn = document.getElementById(removeBtnId);
    const ctx = canvas.getContext("2d");
    let pts = [];
    let closed = false;
    let bgImage = null; // HTMLImageElement or null
    let bgBlob = null; // original uploaded file bytes, when available -- sent to the backend
                        // as-is instead of a canvas re-encode, since re-decoding a JPEG through
                        // <canvas> and re-encoding as PNG was observed to shift RapidOCR's reading
                        // on some labels (e.g. "PANTRY" -> "OOOZ PANTRY") vs. the original bytes.

    function resize() {
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function drawBackground(w, h) {
      if (!bgImage) return;
      const ir = bgImage.width / bgImage.height;
      const cr = w / h;
      let dw, dh, dx, dy;
      if (ir > cr) { dw = w; dh = w / ir; dx = 0; dy = (h - dh) / 2; }
      else { dh = h; dw = h * ir; dy = 0; dx = (w - dw) / 2; }
      ctx.save();
      ctx.globalAlpha = 0.55;
      ctx.drawImage(bgImage, dx, dy, dw, dh);
      ctx.restore();
    }

    function setBackground(img, originalBlob) { bgImage = img; bgBlob = originalBlob || null; removeBtn.classList.add("visible"); draw(); updateCount(); }

    function draw() {
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;
      ctx.clearRect(0, 0, w, h);
      drawBackground(w, h);
      if (pts.length === 0) return;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      if (closed) ctx.closePath();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.stroke();
      if (closed) { ctx.fillStyle = fillColor; ctx.fill(); }
      pts.forEach(([x, y], i) => {
        ctx.beginPath();
        ctx.arc(x, y, i === 0 ? 5 : 3.5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? "#c48a3a" : strokeColor;
        ctx.fill();
      });
    }

    function updateCount() {
      document.getElementById(countId).textContent =
        pts.length + " titik" + (closed ? " · tertutup" : "") + (bgImage ? " · ada gambar" : "");
    }

    function clear() { pts = []; closed = false; draw(); updateCount(); }
    function removeBackground() { bgImage = null; bgBlob = null; removeBtn.classList.remove("visible"); draw(); updateCount(); }
    function getRawState() { return { points: pts.map((p) => p.slice()), closed }; }
    function setRawState(state) {
      pts = (state.points || []).map((p) => p.slice());
      closed = !!state.closed;
      draw();
      updateCount();
    }
    // pts are stored in this panel's own display pixel space (see computeFitRect below),
    // which is tied to the current window layout -- not portable to disk. This inverts
    // that transform back to image-relative pixel coordinates (matching how the backend
    // already reports room/silhouette points), for saving.
    function getImagePixelPoints() {
      if (!bgImage || !closed || pts.length < 3) return null;
      const { dw, dh, dx, dy } = computeFitRect();
      const iw = bgImage.width, ih = bgImage.height;
      return pts.map(([x, y]) => [((x - dx) / dw) * iw, ((y - dy) / dh) * ih]);
    }
    function loadPreset(preset) {
      const rect = canvas.getBoundingClientRect();
      const cx = rect.width / 2, cy = rect.height / 2, s = Math.min(rect.width, rect.height) * 0.32;
      pts = preset.map(([px, py]) => [cx + px * s, cy + py * s]);
      closed = true;
      draw();
      updateCount();
    }
    function getNormalized() {
      if (!closed || pts.length < 3) return null;
      return normalize(pts.map(([x, y]) => [x, -y])); // flip Y: screen-down -> math-up
    }

    function computeFitRect() {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width, h = rect.height;
      const ir = bgImage.width / bgImage.height, cr = w / h;
      let dw, dh, dx, dy;
      if (ir > cr) { dw = w; dh = w / ir; dx = 0; dy = (h - dh) / 2; }
      else { dh = h; dw = h * ir; dy = 0; dx = (w - dw) / 2; }
      return { dw, dh, dx, dy };
    }

    async function runAutoTraceBackend() {
      if (!bgImage) return { error: "no_image" };
      const blob = await getBackgroundBlob();
      if (!blob) return { error: "no_image" };
      const suffix = canvasId.slice(-1);
      const organikEl = document.getElementById("organik" + suffix);
      const gabungEl = document.getElementById("gabung" + suffix);
      let data;
      try {
        data = await backendSegment("siluet", blob, organikEl && organikEl.checked, gabungEl && gabungEl.checked);
      } catch (err) {
        return { error: err.code || "backend_error" };
      }
      const room = data.rooms && data.rooms[0];
      if (!room || !room.points || room.points.length < 3) return { error: "no_closed_region" };
      const fracPts = room.points.map(([x, y]) => [x / data.width, y / data.height]);
      const applied = applyFractionalPoints(fracPts);
      if (applied.error) return applied;
      return { ok: true, points: applied.points, method: "python_backend" };
    }

    function applyFractionalPoints(fracPts) {
      if (!bgImage) return { error: "no_image" };
      if (!Array.isArray(fracPts) || fracPts.length < 3) return { error: "invalid" };
      const { dw, dh, dx, dy } = computeFitRect();
      let newPts = fracPts.map((p) => {
        if (!Array.isArray(p) || p.length < 2) return null;
        const [fx, fy] = p;
        if (typeof fx !== "number" || typeof fy !== "number") return null;
        return [dx + Math.max(0, Math.min(1, fx)) * dw, dy + Math.max(0, Math.min(1, fy)) * dh];
      });
      if (newPts.some((p) => p === null)) return { error: "invalid" };
      if (newPts.length > 1) {
        const [x0, y0] = newPts[0], [xl, yl] = newPts[newPts.length - 1];
        if (Math.hypot(x0 - xl, y0 - yl) < 3) newPts = newPts.slice(0, -1);
      }
      if (newPts.length < 3) return { error: "too_few_points" };
      pts = newPts;
      closed = true;
      draw();
      updateCount();
      return { ok: true, points: pts.length };
    }

    function getBackgroundBlob() {
      return new Promise((resolve) => {
        if (!bgImage) { resolve(null); return; }
        if (bgBlob) { resolve(bgBlob); return; } // original file bytes, unaltered
        const off = document.createElement("canvas");
        off.width = bgImage.width; off.height = bgImage.height;
        off.getContext("2d").drawImage(bgImage, 0, 0);
        off.toBlob((blob) => resolve(blob), "image/png");
      });
    }

    function getWorkingCanvas(maxDim) {
      if (!bgImage) return null;
      maxDim = maxDim || 1000; // matches the tested-safe resolution for the wall-detection logic shared with the silhouette method
      const scale = Math.min(1, maxDim / Math.max(bgImage.width, bgImage.height));
      const ww = Math.max(1, Math.round(bgImage.width * scale));
      const wh = Math.max(1, Math.round(bgImage.height * scale));
      const off = document.createElement("canvas");
      off.width = ww; off.height = wh;
      const octx = off.getContext("2d");
      octx.fillStyle = "#ffffff"; octx.fillRect(0, 0, ww, wh);
      octx.drawImage(bgImage, 0, 0, ww, wh);
      return { canvas: off, ww, wh };
    }

    const ro = new ResizeObserver(() => resize());
    ro.observe(canvas);
    resize();
    return {
      clear, loadPreset, getNormalized, isClosed: () => closed,
      setBackground, removeBackground, runAutoTraceBackend, hasBackground: () => !!bgImage,
      applyFractionalPoints, getBackgroundBlob, getWorkingCanvas,
      getBgImage: () => bgImage, getRawState, setRawState, getImagePixelPoints,
    };
  }

  const presetHouse = [[-1,-1],[1,-1],[1,0.2],[0.3,0.2],[0.3,1],[-1,1]]; // L-shape
  const presetHouseB = [[-1.1,-0.9],[1.1,-0.9],[1.1,0.15],[0.25,0.15],[0.25,1.1],[-1.1,1.1]]; // similar L, slightly diff proportions
  const presetStar = [[0,-1],[0.22,-0.28],[0.95,-0.31],[0.36,0.11],[0.59,0.81],[0,0.38],[-0.59,0.81],[-0.36,0.11],[-0.95,-0.31],[-0.22,-0.28]];

  const drawerA = makeDrawer("canvasA", "countA", "#2e4c7a", "rgba(46,76,122,0.28)", "removeImgA");
  const drawerB = makeDrawer("canvasB", "countB", "#a8462f", "rgba(168,70,47,0.28)", "removeImgB");

  function panelPixelSize(canvasEl) {
    const rect = canvasEl.getBoundingClientRect();
    return { w: Math.max(1, rect.width), h: Math.max(1, rect.height) };
  }

  function wireEditPoints(wrapId, drawer, canvasEl, label) {
    document.getElementById(wrapId).addEventListener("click", (ev) => {
      if (ev.target.closest(".remove-img-btn, .reset-pts-btn")) return; // overlay buttons handle their own clicks
      const state = drawer.getRawState();
      const bgImg = drawer.getBgImage();
      polyModal.open({
        title: "Edit Titik — " + label,
        hint: "Klik untuk menambah titik; klik titik awal (oranye) untuk menutup. Seret titik atau garis untuk mengoreksi. Klik-dua-kali pada garis menambah titik, pada titik menghapusnya.",
        backgroundImage: bgImg,
        bgSize: bgImg ? null : panelPixelSize(canvasEl),
        polyMode: "single",
        initialPolys: state.closed && state.points.length >= 3 ? [{ points: state.points }] : [],
        onDone: (result) => {
          if (result === null) return; // Batal
          const polys = result.polys || [];
          if (polys.length && polys[0].closed) drawer.setRawState({ points: polys[0].points, closed: true });
          else drawer.setRawState({ points: [], closed: false });
          checkReady();
        },
      });
    });
  }
  wireEditPoints("wrapA", drawerA, document.getElementById("canvasA"), "Bentuk A");
  wireEditPoints("wrapB", drawerB, document.getElementById("canvasB"), "Bentuk B");

  document.getElementById("clearA").onclick = (ev) => { ev.stopPropagation(); drawerA.clear(); hideResults(); };
  document.getElementById("clearB").onclick = (ev) => { ev.stopPropagation(); drawerB.clear(); hideResults(); };
  document.getElementById("exampleA").onclick = () => { drawerA.loadPreset(presetHouse); checkReady(); };
  document.getElementById("exampleB").onclick = () => { drawerB.loadPreset(presetHouseB); checkReady(); };

  function wireUpload(uploadBtnId, fileInputId, drawer) {
    const btn = document.getElementById(uploadBtnId);
    const input = document.getElementById(fileInputId);
    btn.onclick = () => input.click();
    input.onchange = () => {
      const file = input.files && input.files[0];
      if (!file) return;
      const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name || "");
      if (isPdf) {
        openPdfPagePicker(file, drawer);
        input.value = "";
        return;
      }
      const img = new Image();
      img.onload = () => { drawer.setBackground(img, file); input.value = ""; };
      img.src = URL.createObjectURL(file);
    };
  }
  wireUpload("uploadA", "fileA", drawerA);
  wireUpload("uploadB", "fileB", drawerB);

  function resetPanel(drawer, editor) {
    drawer.removeBackground();
    drawer.clear();
    editor.reset();
    hideResults();
    checkReady();
  }
  document.getElementById("removeImgA").onclick = (ev) => { ev.stopPropagation(); resetPanel(drawerA, roomEditorA); };
  document.getElementById("removeImgB").onclick = (ev) => { ev.stopPropagation(); resetPanel(drawerB, roomEditorB); };

  const BACKEND_ERROR_MESSAGES = {
    backend_unreachable: "Server Python tidak terjangkau. Jalankan untuk-sam/jalankan.bat (atau `python demo_server.py`) dulu, cek alamat server di atas, lalu coba lagi.",
    no_closed_region: "Deteksi otomatis tidak menemukan bentuk tertutup yang jelas pada gambar ini. Coba gambar dengan garis kontur yang lebih tegas, atau telusuri manual dengan klik.",
    too_few_points: "Kontur yang ditemukan terlalu sedikit titiknya. Coba gambar lain atau telusuri manual dengan klik.",
    no_labels_found: "Tidak ada teks label ruangan yang terbaca pada gambar ini.",
  };
  function backendErrorMessage(code) {
    return BACKEND_ERROR_MESSAGES[code] || ("Deteksi gagal: " + code);
  }

  function wireAutoTrace(btnId, drawer) {
    const btn = document.getElementById(btnId);
    btn.onclick = async () => {
      if (!drawer.hasBackground()) {
        alert("Unggah gambar dulu sebelum memakai deteksi otomatis.");
        return;
      }
      const original = btn.textContent;
      btn.disabled = true; btn.textContent = "Menghubungi server\u2026";
      const res = await drawer.runAutoTraceBackend();
      btn.disabled = false; btn.textContent = original;
      if (res.error) {
        alert(backendErrorMessage(res.error));
      }
      checkReady();
    };
  }
  wireAutoTrace("autoTraceA", drawerA);
  wireAutoTrace("autoTraceB", drawerB);

  const ROOM_PALETTE = ["#a8462f", "#2e4c7a", "#4a8f5c", "#c48a3a", "#6a4a86", "#3a8f8f", "#a83a6a", "#7a7a2e"];

  // ---------- shared fullscreen point-editor: one polygon (siluet A/B) or many
  // named, non-overlapping polygons (room editor). Geometry editing only lives
  // here now -- panels outside show a static, non-interactive preview.
  function makePolygonModal() {
    const overlay = document.getElementById("polyModal");
    const canvas = document.getElementById("polyModalCanvas");
    const ctx = canvas.getContext("2d");
    const titleEl = document.getElementById("polyModalTitle");
    const hintEl = document.getElementById("polyModalHint");
    const sideEl = document.getElementById("polyModalSide");
    const btnNew = document.getElementById("polyModalNew");
    const btnUndo = document.getElementById("polyModalUndo");
    const btnCancel = document.getElementById("polyModalCancel");
    const btnDone = document.getElementById("polyModalDone");
    const modeToggleEl = document.getElementById("polyModalModeToggle");
    const btnModeShape = document.getElementById("polyModalModeShape");
    const btnModeAdjacency = document.getElementById("polyModalModeAdjacency");

    let mode = "single";
    let bgImage = null, bgW = 1, bgH = 1;
    let polys = [];
    let adjacency = []; // [{a: roomId, b: roomId}] -- manual room-to-room connections
    let onDoneCb = null;
    let interaction = null, downPos = null, moved = false;
    let hoveredIndex = -1;
    let history = [];
    let editMode = "shape"; // "shape" | "adjacency" (adjacency only meaningful in multi mode)
    let selectedForEdge = null; // room id picked as the first end of a new connection
    let localIdSeq = 0;
    let baseHint = "";
    const DRAG_THRESHOLD = 4;
    const MAX_HISTORY = 50;

    function ensureId(p) {
      if (!p.id) p.id = "local" + (localIdSeq++);
      return p.id;
    }

    function snapshot() {
      history.push({ polys: JSON.parse(JSON.stringify(polys)), adjacency: JSON.parse(JSON.stringify(adjacency)) });
      if (history.length > MAX_HISTORY) history.shift();
      updateButtons();
    }
    function undo() {
      if (!history.length) return;
      const prev = history.pop();
      polys = prev.polys;
      adjacency = prev.adjacency;
      selectedForEdge = null;
      render();
      updateButtons();
    }

    function openIndex() { return polys.findIndex((p) => !p.closed); }

    function fitRect() {
      const rect = canvas.getBoundingClientRect();
      const ir = bgW / bgH, cr = rect.width / rect.height;
      let dw, dh, dx, dy;
      if (ir > cr) { dw = rect.width; dh = rect.width / ir; dx = 0; dy = (rect.height - dh) / 2; }
      else { dh = rect.height; dw = rect.height * ir; dy = 0; dx = (rect.width - dw) / 2; }
      return { rect, dw, dh, dx, dy, scale: dw / bgW };
    }
    function toCanvas(pt, f) { return [f.dx + pt[0] * f.scale, f.dy + pt[1] * f.scale]; }
    function toImage(cx, cy, f) { return [(cx - f.dx) / f.scale, (cy - f.dy) / f.scale]; }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      render();
    }
    new ResizeObserver(resize).observe(canvas);

    // ported from untuk-sam/demo.html's "mode ruang" overlap guard
    function strictlyInside(x, y, poly) {
      let inside = false;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const yi = poly[i][1], yj = poly[j][1];
        if ((yi > y) === (yj > y)) continue;
        const xCross = (poly[j][0] - poly[i][0]) * (y - yi) / ((yj - yi) || 1e-9) + poly[i][0];
        if (x < xCross) inside = !inside;
      }
      return inside;
    }
    function segCrosses(a, b, c, d) {
      function side(p, q, r) { return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0]); }
      return side(a, c, d) * side(b, c, d) < 0 && side(a, b, c) * side(a, b, d) < 0;
    }
    function hitsOthers(points, skipIndex) {
      if (mode !== "multi") return false;
      return polys.some((other, index) => {
        if (index === skipIndex || other.points.length < 3) return false;
        if (points.some((p) => strictlyInside(p[0], p[1], other.points))) return true;
        if (other.points.some((p) => strictlyInside(p[0], p[1], points))) return true;
        for (let i = 0; i < points.length; i++) {
          const a = points[i], b = points[(i + 1) % points.length];
          for (let j = 0; j < other.points.length; j++) {
            const c = other.points[j], d = other.points[(j + 1) % other.points.length];
            if (segCrosses(a, b, c, d)) return true;
          }
        }
        return false;
      });
    }

    function centroid(points) {
      const n = points.length;
      return [points.reduce((s, p) => s + p[0], 0) / n, points.reduce((s, p) => s + p[1], 0) / n];
    }

    function renderCanvasOnly() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, rect.width, rect.height);
      const f = fitRect();
      if (bgImage) ctx.drawImage(bgImage, f.dx, f.dy, f.dw, f.dh);
      polys.forEach((p, i) => {
        const highlighted = editMode === "adjacency" ? p.id === selectedForEdge : i === hoveredIndex;
        drawPoly(p.points.map((pt) => toCanvas(pt, f)), p.color, p.closed, highlighted);
      });
      if (editMode === "adjacency") {
        const byId = {};
        polys.forEach((p) => { if (p.closed && p.id) byId[p.id] = toCanvas(centroid(p.points), f); });
        ctx.strokeStyle = "rgba(168,70,47,0.9)"; ctx.lineWidth = 2.4;
        adjacency.forEach((edge) => {
          const pa = byId[edge.a], pb = byId[edge.b];
          if (!pa || !pb) return;
          ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
        });
        ctx.fillStyle = "#1c2530";
        Object.values(byId).forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); });
      }
    }

    function render() {
      renderCanvasOnly();
      if (mode === "multi") renderSide();
    }

    function drawPoly(pts, color, closed, highlighted) {
      if (pts.length === 0) return;
      ctx.beginPath();
      pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
      if (closed) ctx.closePath();
      ctx.strokeStyle = color; ctx.lineWidth = highlighted ? 4.5 : 2.5;
      ctx.stroke();
      if (closed) { ctx.fillStyle = color + (highlighted ? "77" : "33"); ctx.fill(); }
      pts.forEach(([x, y], i) => {
        ctx.beginPath();
        ctx.arc(x, y, i === 0 ? 6 : 4.5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? "#c48a3a" : color;
        ctx.fill();
      });
    }

    function renderSide() {
      if (editMode === "adjacency") { renderAdjacencyList(); return; }
      sideEl.innerHTML = "";
      polys.forEach((p, i) => {
        const row = document.createElement("div");
        row.className = "room-row";
        row.onmouseenter = () => { hoveredIndex = i; renderCanvasOnly(); };
        row.onmouseleave = () => { hoveredIndex = -1; renderCanvasOnly(); };
        const sw = document.createElement("span");
        sw.className = "swatch-sm"; sw.style.background = p.color;
        const label = document.createElement("span");
        label.style.flex = "1"; label.style.fontSize = "12px"; label.style.color = "var(--ink)";
        label.textContent = (p.name || "(baru)") + " · " + p.points.length + " titik" + (p.closed ? "" : " · belum ditutup");
        const del = document.createElement("button");
        del.className = "del"; del.textContent = "Hapus";
        del.onclick = () => {
          snapshot();
          const removedId = p.id;
          polys.splice(i, 1);
          adjacency = adjacency.filter((e) => e.a !== removedId && e.b !== removedId);
          hoveredIndex = -1;
          render();
        };
        row.append(sw, label, del);
        sideEl.appendChild(row);
      });
    }

    function renderAdjacencyList() {
      sideEl.innerHTML = "";
      const nameOf = (id) => { const p = polys.find((q) => q.id === id); return p ? (p.name || "(tanpa nama)") : id; };
      if (adjacency.length === 0) {
        const note = document.createElement("p");
        note.style.fontSize = "12px"; note.style.color = "var(--ink-soft)";
        note.textContent = "Belum ada hubungan. Klik satu ruang lalu ruang lain di gambar untuk menghubungkan.";
        sideEl.appendChild(note);
      }
      adjacency.forEach((edge, i) => {
        const row = document.createElement("div");
        row.className = "room-row";
        const label = document.createElement("span");
        label.style.flex = "1"; label.style.fontSize = "12px"; label.style.color = "var(--ink)";
        label.textContent = nameOf(edge.a) + " — " + nameOf(edge.b);
        const del = document.createElement("button");
        del.className = "del"; del.textContent = "Hapus";
        del.onclick = () => { snapshot(); adjacency.splice(i, 1); render(); };
        row.append(label, del);
        sideEl.appendChild(row);
      });
    }

    function distToSegmentInfo(p, a, b) {
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const lenSq = dx * dx + dy * dy;
      let t = lenSq > 0 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lenSq : 0;
      t = Math.max(0, Math.min(1, t));
      const cx = a[0] + t * dx, cy = a[1] + t * dy;
      return { dist: Math.hypot(p[0] - cx, p[1] - cy), t };
    }

    canvas.addEventListener("pointerdown", (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      downPos = [x, y]; moved = false;
      const f = fitRect();

      if (editMode === "adjacency") {
        const [ix, iy] = toImage(x, y, f);
        const hit = polys.find((p) => p.closed && strictlyInside(ix, iy, p.points));
        if (!hit) { selectedForEdge = null; renderCanvasOnly(); return; }
        ensureId(hit);
        if (selectedForEdge === null) {
          selectedForEdge = hit.id;
        } else if (selectedForEdge === hit.id) {
          selectedForEdge = null;
        } else {
          snapshot();
          const existingIdx = adjacency.findIndex((e2) =>
            (e2.a === selectedForEdge && e2.b === hit.id) || (e2.a === hit.id && e2.b === selectedForEdge));
          if (existingIdx >= 0) adjacency.splice(existingIdx, 1);
          else adjacency.push({ a: selectedForEdge, b: hit.id });
          selectedForEdge = null;
        }
        render();
        return;
      }

      for (let ri = 0; ri < polys.length; ri++) {
        const pts = polys[ri].points;
        for (let pi = 0; pi < pts.length; pi++) {
          const [px, py] = toCanvas(pts[pi], f);
          if (Math.hypot(x - px, y - py) < 11) {
            snapshot();
            interaction = { type: "point", ri, pi };
            canvas.setPointerCapture(e.pointerId);
            return;
          }
        }
      }
      for (let ri = 0; ri < polys.length; ri++) {
        const poly = polys[ri];
        if (!poly.closed || poly.points.length < 2) continue;
        for (let i = 0; i < poly.points.length; i++) {
          const a = toCanvas(poly.points[i], f), b = toCanvas(poly.points[(i + 1) % poly.points.length], f);
          const { dist, t } = distToSegmentInfo([x, y], a, b);
          if (dist < 8 && t > 0.05 && t < 0.95) {
            snapshot();
            interaction = {
              type: "segment", ri, i0: i, i1: (i + 1) % poly.points.length,
              origA: poly.points[i].slice(), origB: poly.points[(i + 1) % poly.points.length].slice(),
            };
            canvas.setPointerCapture(e.pointerId);
            return;
          }
        }
      }
      const oi = openIndex();
      if (oi >= 0) {
        const poly = polys[oi];
        if (poly.points.length > 2) {
          const [sx, sy] = toCanvas(poly.points[0], f);
          if (Math.hypot(x - sx, y - sy) < 22) { closeOpenPolygon(oi); return; }
        }
        snapshot();
        poly.points.push(toImage(x, y, f));
        render();
        return;
      }
      if (mode === "single" && polys.length === 0) {
        snapshot();
        polys.push({ name: "", points: [toImage(x, y, f)], closed: false, color: "#a8462f" });
        render();
      }
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!interaction) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      if (!moved && Math.hypot(x - downPos[0], y - downPos[1]) > DRAG_THRESHOLD) moved = true;
      if (!moved) return;
      const f = fitRect();
      const poly = polys[interaction.ri];
      if (interaction.type === "point") {
        const pt = toImage(x, y, f);
        const next = poly.points.map((p, i) => i === interaction.pi ? pt : p);
        if (poly.closed && hitsOthers(next, interaction.ri)) return;
        poly.points[interaction.pi] = pt;
      } else if (interaction.type === "segment") {
        const [ix, iy] = toImage(x, y, f);
        const [dx0, dy0] = toImage(downPos[0], downPos[1], f);
        const ddx = ix - dx0, ddy = iy - dy0;
        const newA = [interaction.origA[0] + ddx, interaction.origA[1] + ddy];
        const newB = [interaction.origB[0] + ddx, interaction.origB[1] + ddy];
        const next = poly.points.map((p, i) => i === interaction.i0 ? newA : i === interaction.i1 ? newB : p);
        if (hitsOthers(next, interaction.ri)) return;
        poly.points[interaction.i0] = newA;
        poly.points[interaction.i1] = newB;
      }
      render();
    });

    function endInteraction(e) {
      interaction = null; moved = false;
      try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
    }
    canvas.addEventListener("pointerup", endInteraction);
    canvas.addEventListener("pointercancel", endInteraction);

    canvas.addEventListener("dblclick", (e) => {
      if (editMode === "adjacency") return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      const f = fitRect();
      for (let ri = 0; ri < polys.length; ri++) {
        const poly = polys[ri];
        if (!poly.closed) continue;
        for (let pi = 0; pi < poly.points.length; pi++) {
          const [px, py] = toCanvas(poly.points[pi], f);
          if (Math.hypot(x - px, y - py) < 11) {
            if (poly.points.length > 3) { snapshot(); poly.points.splice(pi, 1); render(); }
            return;
          }
        }
        for (let i = 0; i < poly.points.length; i++) {
          const a = toCanvas(poly.points[i], f), b = toCanvas(poly.points[(i + 1) % poly.points.length], f);
          const { dist, t } = distToSegmentInfo([x, y], a, b);
          if (dist < 8 && t > 0.02 && t < 0.98) {
            const next = poly.points.slice();
            next.splice(i + 1, 0, toImage(x, y, f));
            if (!hitsOthers(next, ri)) { snapshot(); poly.points = next; render(); }
            return;
          }
        }
      }
    });

    function closeOpenPolygon(oi) {
      const poly = polys[oi];
      if (poly.points.length < 3) { hintEl.textContent = "Perlu sedikitnya tiga titik sebelum ditutup."; return; }
      if (hitsOthers(poly.points, oi)) { hintEl.textContent = "Poligon ini menimpa ruang lain. Geser titiknya atau tekan Batal, lalu coba lagi."; return; }
      snapshot();
      poly.closed = true;
      if (mode === "multi" && !poly.name) poly.name = "Ruangan baru";
      ensureId(poly);
      render();
      updateButtons();
    }

    function setEditMode(next) {
      editMode = next;
      selectedForEdge = null;
      hintEl.textContent = editMode === "adjacency"
        ? "Klik satu ruang, lalu ruang lain untuk menghubungkan atau memutus hubungan pintu di antaranya."
        : baseHint;
      updateButtons();
      render();
    }

    function updateButtons() {
      btnNew.style.display = mode === "multi" && editMode === "shape" ? "" : "none";
      btnNew.disabled = mode === "multi" && openIndex() >= 0;
      btnUndo.disabled = history.length === 0;
      modeToggleEl.style.display = mode === "multi" ? "" : "none";
      btnModeShape.classList.toggle("active", editMode === "shape");
      btnModeAdjacency.classList.toggle("active", editMode === "adjacency");
    }

    btnNew.onclick = () => {
      if (mode !== "multi" || openIndex() >= 0) return;
      snapshot();
      polys.push({ id: "local" + (localIdSeq++), name: "", points: [], closed: false, color: ROOM_PALETTE[polys.length % ROOM_PALETTE.length] });
      hintEl.textContent = "Klik di gambar untuk menambah titik ruangan baru, lalu klik titik awal untuk menutup.";
      updateButtons();
      render();
    };
    btnUndo.onclick = () => undo();
    btnModeShape.onclick = () => setEditMode("shape");
    btnModeAdjacency.onclick = () => setEditMode("adjacency");
    btnCancel.onclick = () => finish(null);
    btnDone.onclick = () => {
      const oi = openIndex();
      if (oi >= 0) {
        if (polys[oi].points.length < 3) polys.splice(oi, 1);
        else { hintEl.textContent = "Tutup poligon yang sedang digambar dulu (klik titik awalnya), atau Batal."; return; }
      }
      finish({ polys, adjacency });
    };

    function finish(result) {
      overlay.classList.remove("show");
      const cb = onDoneCb;
      onDoneCb = null; polys = []; adjacency = []; bgImage = null; interaction = null; history = [];
      hoveredIndex = -1; selectedForEdge = null; editMode = "shape";
      document.removeEventListener("keydown", onKeydown);
      if (cb) cb(result);
    }

    function onKeydown(e) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        undo();
      } else if (e.key === "Escape") {
        finish(null);
      }
    }

    return {
      open({ title, hint, backgroundImage, bgSize, polyMode, initialPolys, initialAdjacency, onDone }) {
        mode = polyMode || "single";
        bgImage = backgroundImage || null;
        if (bgImage) { bgW = bgImage.width || bgImage.naturalWidth || 1; bgH = bgImage.height || bgImage.naturalHeight || 1; }
        else if (bgSize) { bgW = Math.max(1, bgSize.w); bgH = Math.max(1, bgSize.h); }
        else { bgW = 1; bgH = 1; }
        localIdSeq = 0;
        polys = (initialPolys || []).map((p, i) => ({
          id: p.id || ("local" + (localIdSeq++)),
          name: p.name || "",
          points: p.points.map((pt) => pt.slice()),
          closed: true,
          color: p.color || ROOM_PALETTE[i % ROOM_PALETTE.length],
        }));
        adjacency = (initialAdjacency || [])
          .map((e) => ({ a: e.a, b: e.b }))
          .filter((e) => polys.some((p) => p.id === e.a) && polys.some((p) => p.id === e.b));
        editMode = "shape";
        selectedForEdge = null;
        onDoneCb = onDone;
        history = [];
        hoveredIndex = -1;
        baseHint = hint || "";
        titleEl.textContent = title || "";
        hintEl.textContent = baseHint;
        sideEl.classList.toggle("show", mode === "multi");
        overlay.classList.add("show");
        updateButtons();
        document.addEventListener("keydown", onKeydown);
        requestAnimationFrame(() => { resize(); render(); });
      },
    };
  }

  const polyModal = makePolygonModal();

  function makeRoomEditor(ids) {
    // ids: {canvasId, listId, summaryId, addBtnId}
    const canvas = document.getElementById(ids.canvasId);
    let state = null; // { ww, wh, sourceCanvas, rooms, adjacency }

    function fitRect() {
      const rect = canvas.getBoundingClientRect();
      const ir = state.ww / state.wh, cr = rect.width / rect.height;
      let dw, dh, dx, dy;
      if (ir > cr) { dw = rect.width; dh = rect.width / ir; dx = 0; dy = (rect.height - dh) / 2; }
      else { dh = rect.height; dw = rect.height * ir; dy = 0; dx = (rect.width - dw) / 2; }
      return { rect, dw, dh, dx, dy, scale: dw / state.ww };
    }

    function centroid(points) {
      const n = points.length;
      return [points.reduce((s, p) => s + p[0], 0) / n, points.reduce((s, p) => s + p[1], 0) / n];
    }

    let hoveredIndex = -1;

    function renderCanvas() {
      if (!state) return;
      const dpr = window.devicePixelRatio || 1;
      const { rect, dw, dh, dx, dy, scale } = fitRect();
      canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#f4f7f8"; ctx.fillRect(0, 0, rect.width, rect.height);
      ctx.drawImage(state.sourceCanvas, dx, dy, dw, dh);
      const toCanvas = ([x, y]) => [dx + x * scale, dy + y * scale];
      state.rooms.forEach((r, i) => {
        const pts = r.points.map(toCanvas);
        const color = ROOM_PALETTE[i % ROOM_PALETTE.length];
        const hi = i === hoveredIndex;
        ctx.beginPath();
        pts.forEach((p, j) => j === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
        ctx.closePath();
        ctx.fillStyle = color + (hi ? "77" : "33"); ctx.fill();
        ctx.strokeStyle = color; ctx.lineWidth = hi ? 3.5 : 2; ctx.stroke();
        const minX = Math.min(...pts.map((p) => p[0])), minY = Math.min(...pts.map((p) => p[1]));
        ctx.fillStyle = color; ctx.font = "bold 11px Space Grotesk, sans-serif";
        ctx.fillText(r.name || "(tanpa nama)", minX + 4, minY + 14);
      });
      if (state.adjacency && state.adjacency.length) {
        const byId = {};
        state.rooms.forEach((r) => { if (r.id) byId[r.id] = toCanvas(centroid(r.points)); });
        ctx.strokeStyle = "rgba(168,70,47,0.85)"; ctx.lineWidth = 1.6;
        state.adjacency.forEach((edge) => {
          const pa = byId[edge.a], pb = byId[edge.b];
          if (!pa || !pb) return;
          ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
        });
        ctx.fillStyle = "#1c2530";
        Object.values(byId).forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill(); });
      }
    }

    function render() {
      renderCanvas();
      renderList();
    }

    function renderList() {
      const listEl = document.getElementById(ids.listId);
      listEl.innerHTML = "";
      state.rooms.forEach((r, i) => {
        const row = document.createElement("div");
        row.className = "room-row";
        row.onmouseenter = () => { hoveredIndex = i; renderCanvas(); };
        row.onmouseleave = () => { hoveredIndex = -1; renderCanvas(); };
        const sw = document.createElement("span");
        sw.className = "swatch-sm";
        sw.style.background = ROOM_PALETTE[i % ROOM_PALETTE.length];
        const input = document.createElement("input");
        input.type = "text"; input.value = r.name;
        input.oninput = () => { r.name = input.value; };
        input.onblur = render;
        const del = document.createElement("button");
        del.className = "del"; del.textContent = "Hapus";
        del.onclick = () => { state.rooms.splice(i, 1); hoveredIndex = -1; render(); checkReady(); };
        row.append(sw, input, del);
        listEl.appendChild(row);
      });
      const doorNote = state.adjacency && state.adjacency.length ? " Garis merah tipis menandai sambungan pintu antar-ruang." : "";
      document.getElementById(ids.summaryId).textContent =
        state.rooms.length + " ruangan terdaftar. Nama bisa diedit langsung; \"Hapus\" untuk membuang; \"Edit Ruang\" untuk mengoreksi titik atau menandai ruangan yang belum terdeteksi." + doorNote;
    }

    document.getElementById(ids.addBtnId).onclick = () => {
      if (!state) return;
      polyModal.open({
        title: "Edit Ruang",
        hint: "Seret titik/garis untuk mengoreksi. \"+ Poligon Baru\" lalu klik di gambar untuk menandai ruangan yang belum terdeteksi — larangan tumpang-tindih otomatis berlaku. Pakai \"Edit Hubungan\" untuk mengoreksi sambungan pintu antar-ruang secara manual.",
        backgroundImage: state.sourceCanvas,
        polyMode: "multi",
        initialPolys: state.rooms.map((r, i) => ({ id: r.id, name: r.name, points: r.points, color: ROOM_PALETTE[i % ROOM_PALETTE.length] })),
        initialAdjacency: state.adjacency || [],
        onDone: (result) => {
          if (result === null) return; // Batal
          state.rooms = result.polys.filter((p) => p.closed && p.points.length >= 3).map((p) => ({
            id: p.id, name: p.name || "", source: p.name ? "tulisan" : "kosong", points: p.points,
          }));
          const keptIds = new Set(state.rooms.map((r) => r.id));
          state.adjacency = (result.adjacency || []).filter((e) => keptIds.has(e.a) && keptIds.has(e.b));
          render();
          checkReady();
        },
      });
    };

    return {
      setResult(ww, wh, sourceCanvas, rooms, adjacency) {
        state = { ww, wh, sourceCanvas, rooms, adjacency: adjacency || [] };
        document.getElementById(ids.canvasId.replace("Canvas", "Results")).style.display = "block";
        render();
      },
      getState: () => state,
      reset() {
        state = null;
        document.getElementById(ids.canvasId.replace("Canvas", "Results")).style.display = "none";
        document.getElementById(ids.listId).innerHTML = "";
        document.getElementById(ids.summaryId).textContent = "";
      },
    };
  }

  const roomEditorA = makeRoomEditor({ canvasId: "roomCanvasA", listId: "roomListA", summaryId: "roomSummaryA", addBtnId: "roomAddA" });
  const roomEditorB = makeRoomEditor({ canvasId: "roomCanvasB", listId: "roomListB", summaryId: "roomSummaryB", addBtnId: "roomAddB" });

  function wireRoomTrace(btnId, drawer, editor, progressId) {
    const btn = document.getElementById(btnId);
    const progressEl = document.getElementById(progressId);
    btn.onclick = async () => {
      if (!drawer.hasBackground()) { alert("Unggah gambar dulu sebelum memakai deteksi ruang."); return; }
      const blob = await drawer.getBackgroundBlob();
      if (!blob) return;
      const original = btn.textContent;
      btn.disabled = true; btn.textContent = "Menghubungi server\u2026";
      progressEl.textContent = "Membaca denah di server Python\u2026";
      const organikEl = document.getElementById("organik" + btnId.slice(-1));
      const organik = !!(organikEl && organikEl.checked);
      try {
        const data = await backendSegment("ruang", blob, organik, false);
        const rooms = (data.rooms || [])
          .filter((r) => r.points && r.points.length >= 3)
          .map((r) => ({ id: r.id || "", name: r.name || "", source: r.name ? (r.source || "tulisan") : "kosong", points: r.points }));
        if (rooms.length === 0) { alert(backendErrorMessage("no_labels_found")); return; }
        if (organik) alert("Dinding Melengkung: bentuk ruang belum digambar otomatis (celah pintu pada dinding melengkung belum bisa dijembatani secara andal). Tiap ruang jadi kotak PERKIRAAN luas (mengikuti dinding terdekat ke segala arah dari posisi namanya) -- buka \"Edit Ruang\" lalu gambar ulang tiap kotak jadi bentuk ruang yang sebenarnya.");
        const full = drawer.getWorkingCanvas(1e9); // full resolution, matches the pixel space of `data`
        editor.setResult(data.width, data.height, full.canvas, rooms, data.adjacency);
        checkReady();
      } catch (err) {
        alert(backendErrorMessage(err.code || (err && err.message) || "unknown"));
      } finally {
        btn.disabled = false; btn.textContent = original; progressEl.textContent = "";
      }
    };
  }
  wireRoomTrace("roomTraceA", drawerA, roomEditorA, "ocrProgressA");
  wireRoomTrace("roomTraceB", drawerB, roomEditorB, "ocrProgressB");

  // ---------- save/load a case: silhouette + rooms + adjacency + the image itself, all
  // in one JSON file on the Python backend (untuk-sam/hasil/<name>-case.json) -- this is
  // the persistent input the later similarity components (beyond silhouette) will read.
  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  async function refreshCaseList(selectEl) {
    try {
      const names = await (await fetch(getBackendUrl() + "/cases")).json();
      const previous = selectEl.value;
      selectEl.innerHTML = '<option value="">(pilih kasus tersimpan)</option>' +
        names.map((n) => `<option value="${n}">${n.replace(/-case\.json$/, "")}</option>`).join("");
      if (names.includes(previous)) selectEl.value = previous;
    } catch (_) {}
  }

  function wireSaveLoad(saveBtnId, loadBtnId, selectId, statusId, drawer, roomEditor, label) {
    const saveBtn = document.getElementById(saveBtnId);
    const loadBtn = document.getElementById(loadBtnId);
    const selectEl = document.getElementById(selectId);
    const statusEl = document.getElementById(statusId);
    refreshCaseList(selectEl);

    saveBtn.onclick = async () => {
      const siluetState = drawer.getRawState();
      const roomState = roomEditor.getState();
      if (!siluetState.closed && !roomState) {
        alert("Belum ada data untuk disimpan. Tutup siluet (Edit Titik) atau jalankan Deteksi Ruang dulu.");
        return;
      }
      const blob = await drawer.getBackgroundBlob();
      if (!blob) {
        alert("Unggah gambar dulu -- data disimpan bersama gambar aslinya supaya bisa dipakai lagi nanti.");
        return;
      }
      const defaultName = (blob.name || ("denah-" + label)).replace(/\.[^.]+$/, "");
      const name = prompt("Nama untuk kasus ini:", defaultName);
      if (!name) return;
      statusEl.textContent = "Menyimpan…";
      try {
        const base64 = await blobToBase64(blob);
        const bgImg = drawer.getBgImage();
        const payload = {
          name,
          image: { filename: blob.name || (name + ".png"), content_type: blob.type || "image/png", data_base64: base64 },
          width: bgImg ? (bgImg.width || bgImg.naturalWidth) : (roomState ? roomState.ww : null),
          height: bgImg ? (bgImg.height || bgImg.naturalHeight) : (roomState ? roomState.wh : null),
          silhouette: drawer.getImagePixelPoints(),
          rooms: roomState ? roomState.rooms : [],
          adjacency: roomState ? (roomState.adjacency || []) : [],
        };
        const res = await fetch(getBackendUrl() + "/save_case", { method: "POST", body: JSON.stringify(payload) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || ("http_" + res.status));
        statusEl.textContent = "Tersimpan: " + data.saved;
        refreshCaseList(selectEl);
      } catch (err) {
        statusEl.textContent = "";
        alert("Gagal menyimpan: " + (err && err.message ? err.message : err));
      }
    };

    loadBtn.onclick = async () => {
      const chosen = selectEl.value;
      if (!chosen) { alert("Pilih kasus tersimpan dulu dari daftar."); return; }
      statusEl.textContent = "Memuat…";
      try {
        const data = await (await fetch(getBackendUrl() + "/cases/" + encodeURIComponent(chosen))).json();
        if (data.image && data.image.data_base64) {
          const byteChars = atob(data.image.data_base64);
          const bytes = new Uint8Array(byteChars.length);
          for (let i = 0; i < byteChars.length; i++) bytes[i] = byteChars.charCodeAt(i);
          const imgBlob = new Blob([bytes], { type: (data.image.content_type || "image/png") });
          const img = new Image();
          await new Promise((resolve, reject) => {
            img.onload = resolve; img.onerror = reject;
            img.src = URL.createObjectURL(imgBlob);
          });
          drawer.setBackground(img, imgBlob);
        } else {
          drawer.removeBackground();
        }
        if (data.silhouette && data.silhouette.length >= 3 && data.width && data.height) {
          const fracPts = data.silhouette.map(([x, y]) => [x / data.width, y / data.height]);
          drawer.applyFractionalPoints(fracPts);
        } else {
          drawer.setRawState({ points: [], closed: false });
        }
        if (data.rooms && data.rooms.length && data.width && data.height) {
          const full = drawer.getWorkingCanvas(1e9);
          roomEditor.setResult(data.width, data.height, full.canvas, data.rooms, data.adjacency || []);
        }
        statusEl.textContent = "Dimuat: " + chosen;
        checkReady();
      } catch (err) {
        statusEl.textContent = "";
        alert("Gagal memuat: " + (err && err.message ? err.message : err));
      }
    };
  }
  wireSaveLoad("saveCaseA", "loadCaseA", "caseListA", "caseStatusA", drawerA, roomEditorA, "A");
  wireSaveLoad("saveCaseB", "loadCaseB", "caseListB", "caseStatusB", drawerB, roomEditorB, "B");

  (function wireBackendStatus() {
    const input = document.getElementById("backendUrl");
    const statusEl = document.getElementById("backendStatus");
    const checkBtn = document.getElementById("backendCheck");
    try {
      const saved = localStorage.getItem("skripsi_backend_url");
      if (saved) input.value = saved;
    } catch (_) {}
    async function check() {
      statusEl.style.color = "var(--ink-soft)";
      statusEl.textContent = "mengecek…";
      try {
        const res = await fetch(getBackendUrl() + "/hasil", { method: "GET" });
        if (!res.ok) throw new Error("http_" + res.status);
        statusEl.textContent = "terhubung";
        statusEl.style.color = "#3a8f5c";
      } catch (_) {
        statusEl.textContent = "tidak terhubung — jalankan untuk-sam/jalankan.bat dulu";
        statusEl.style.color = "var(--rust)";
      }
      try { localStorage.setItem("skripsi_backend_url", input.value.trim()); } catch (_) {}
    }
    checkBtn.onclick = check;
    check();
  })();

  const computeBtn = document.getElementById("computeBtn");

  function hasRoomData(roomEditor) {
    const st = roomEditor.getState();
    return !!(st && st.rooms && st.rooms.length > 0);
  }

  function checkReady() {
    const silhouetteReady = drawerA.isClosed() && drawerB.isClosed();
    const topoReady = hasRoomData(roomEditorA) && hasRoomData(roomEditorB);
    computeBtn.disabled = !(silhouetteReady || topoReady);
  }

  function hideResults() {
    document.getElementById("results").classList.remove("show");
    checkReady();
  }

  let lastCompute = null; // { a, bDefault, bBest, defaultOverlap, best }
  let rotationMode = "best"; // "best" | "default" | "efd"

  function setReadoutActive(mode) {
    document.getElementById("readoutBest").classList.toggle("active", mode === "best");
    document.getElementById("readoutDefault").classList.toggle("active", mode === "default");
    document.getElementById("readoutEfd").classList.toggle("active", mode === "efd");
  }

  function efdVerdict(d) {
    if (d < 0.15) return { label: "sangat dekat", verdictHigh: true };
    if (d < 0.4) return { label: "sedang", verdictHigh: null };
    return { label: "jauh", verdictHigh: false };
  }

  function roomNameById(roomState, id) {
    const r = (roomState.rooms || []).find((x) => x.id === id);
    return r ? (r.name || "(tanpa nama)") : id;
  }

  function renderTopologyResult(result, roomStateA, roomStateB, graphA, graphB) {
    const svgRootA = document.getElementById("bubbleSvgA");
    const svgRootB = document.getElementById("bubbleSvgB");
    drawBubbleDiagram(svgRootA, graphA);
    drawBubbleDiagram(svgRootB, graphB);
    document.getElementById("topoScoreBig").textContent = (result.similarity * 100).toFixed(1) + "%";
    document.getElementById("topoGedReadout").textContent = result.ged.toFixed(2);
    document.getElementById("topoNodeReadout").textContent =
      result.matches.length + " / " + result.deleted.length + " / " + result.inserted.length;
    document.getElementById("topoEdgeReadout").textContent =
      result.edgeChanges.kept + " / " + result.edgeChanges.removed + " / " + result.edgeChanges.added;

    let note;
    if (result.similarity > 0.75) note = "Topologi tinggi — susunan & hubungan antar-ruang sangat mirip.";
    else if (result.similarity > 0.45) note = "Topologi sedang — ada kemiripan susunan ruang, tapi juga perbedaan berarti.";
    else note = "Topologi rendah — susunan & hubungan antar-ruang berbeda secara mendasar.";
    document.getElementById("topoScoreNote").textContent = note;

    const table = document.getElementById("topoTable");
    table.innerHTML = "";
    function addRow(cells, isHeader, aId, bId) {
      const tr = document.createElement("tr");
      cells.forEach((c) => {
        const cell = document.createElement(isHeader ? "th" : "td");
        cell.textContent = c;
        cell.style.padding = "4px 8px";
        cell.style.borderBottom = isHeader ? "2px solid var(--paper-line)" : "1px solid var(--paper-line)";
        if (isHeader) { cell.style.textAlign = "left"; cell.style.color = "var(--ink-soft)"; cell.style.fontSize = "11px"; cell.style.textTransform = "uppercase"; }
        tr.appendChild(cell);
      });
      if (!isHeader && (aId || bId)) {
        tr.setAttribute("data-hoverable", "1");
        tr.addEventListener("mouseenter", () => { setBubbleHighlight(svgRootA, aId, true); setBubbleHighlight(svgRootB, bId, true); });
        tr.addEventListener("mouseleave", () => { setBubbleHighlight(svgRootA, aId, false); setBubbleHighlight(svgRootB, bId, false); });
      }
      table.appendChild(tr);
    }
    addRow(["Ruang A", "Ruang B", "Biaya"], true);
    result.matches.slice().sort((x, y) => y.cost - x.cost).forEach((m) => {
      addRow([roomNameById(roomStateA, m.a), roomNameById(roomStateB, m.b), m.cost.toFixed(2)], false, m.a, m.b);
    });
    result.deleted.forEach((id) => addRow([roomNameById(roomStateA, id), "(dihapus)", "1.00"], false, id, null));
    result.inserted.forEach((id) => addRow(["(disisipkan)", roomNameById(roomStateB, id), "1.00"], false, null, id));
  }

  function hieBar(className, value) {
    const track = document.createElement("div");
    track.className = "hie-bar-track";
    const fill = document.createElement("div");
    fill.className = "hie-bar-fill " + className;
    fill.style.width = (value * 100).toFixed(0) + "%";
    track.appendChild(fill);
    const val = document.createElement("span");
    val.className = "hie-bar-val";
    val.textContent = value.toFixed(2);
    return [track, val];
  }

  function renderCentralityRank(container, roomState, centralities, svgRoot) {
    container.innerHTML = "";
    const rows = (roomState.rooms || []).map((r) => {
      const c = centralities.get(r.id) || { connectivity: 0, integration: 0, choice: 0 };
      return { id: r.id, name: r.name || "(tanpa nama)", connectivity: c.connectivity, integration: c.integration, choice: c.choice };
    });
    rows.sort((x, y) => (y.connectivity + y.integration + y.choice) - (x.connectivity + x.integration + x.choice));
    rows.forEach((r) => {
      const row = document.createElement("div");
      row.className = "hie-rank-row";
      const name = document.createElement("span");
      name.className = "hie-rank-name";
      name.textContent = r.name;
      name.title = r.name;
      row.append(name, ...hieBar("conn", r.connectivity), ...hieBar("integ", r.integration), ...hieBar("choice", r.choice));
      row.addEventListener("mouseenter", () => setBubbleHighlight(svgRoot, r.id, true));
      row.addEventListener("mouseleave", () => setBubbleHighlight(svgRoot, r.id, false));
      container.appendChild(row);
    });
  }

  function renderHierarchyResult(result, roomStateA, roomStateB, svgRootA, svgRootB) {
    document.getElementById("hieScoreBig").textContent = (result.similarity * 100).toFixed(1) + "%";
    document.getElementById("hieGedReadout").textContent = result.cost.toFixed(2);
    document.getElementById("hieNodeReadout").textContent =
      result.matches.length + " / " + result.deleted.length + " / " + result.inserted.length;

    let note;
    if (result.similarity > 0.75) note = "Hierarki tinggi — ruang yang paling sentral/dominan sejalan di kedua bentuk.";
    else if (result.similarity > 0.45) note = "Hierarki sedang — ada kemiripan tingkat kepentingan ruang, tapi juga perbedaan berarti.";
    else note = "Hierarki rendah — ruang yang paling sentral/dominan berbeda secara mendasar.";
    document.getElementById("hieScoreNote").textContent = note;

    renderCentralityRank(document.getElementById("hieRankA"), roomStateA, result.centralitiesA, svgRootA);
    renderCentralityRank(document.getElementById("hieRankB"), roomStateB, result.centralitiesB, svgRootB);

    const table = document.getElementById("hieTable");
    table.innerHTML = "";
    function addRow(cells, isHeader, aId, bId) {
      const tr = document.createElement("tr");
      cells.forEach((c) => {
        const cell = document.createElement(isHeader ? "th" : "td");
        cell.textContent = c;
        cell.style.padding = "4px 8px";
        cell.style.borderBottom = isHeader ? "2px solid var(--paper-line)" : "1px solid var(--paper-line)";
        if (isHeader) { cell.style.textAlign = "left"; cell.style.color = "var(--ink-soft)"; cell.style.fontSize = "11px"; cell.style.textTransform = "uppercase"; }
        tr.appendChild(cell);
      });
      if (!isHeader && (aId || bId)) {
        tr.setAttribute("data-hoverable", "1");
        tr.addEventListener("mouseenter", () => { setBubbleHighlight(svgRootA, aId, true); setBubbleHighlight(svgRootB, bId, true); });
        tr.addEventListener("mouseleave", () => { setBubbleHighlight(svgRootA, aId, false); setBubbleHighlight(svgRootB, bId, false); });
      }
      table.appendChild(tr);
    }
    addRow(["Ruang A", "Ruang B", "Konek. A", "Konek. B", "Integ. A", "Integ. B", "Choice A", "Choice B", "Biaya"], true);
    result.matches.slice().sort((x, y) => y.cost - x.cost).forEach((m) => {
      const ca = result.centralitiesA.get(m.a), cb = result.centralitiesB.get(m.b);
      addRow([
        roomNameById(roomStateA, m.a), roomNameById(roomStateB, m.b),
        ca.connectivity.toFixed(2), cb.connectivity.toFixed(2),
        ca.integration.toFixed(2), cb.integration.toFixed(2),
        ca.choice.toFixed(2), cb.choice.toFixed(2),
        m.cost.toFixed(2),
      ], false, m.a, m.b);
    });
    result.deleted.forEach((id) => {
      const ca = result.centralitiesA.get(id);
      addRow([roomNameById(roomStateA, id), "(dihapus)", ca.connectivity.toFixed(2), "—", ca.integration.toFixed(2), "—", ca.choice.toFixed(2), "—", "1.00"], false, id, null);
    });
    result.inserted.forEach((id) => {
      const cb = result.centralitiesB.get(id);
      addRow(["(disisipkan)", roomNameById(roomStateB, id), "—", cb.connectivity.toFixed(2), "—", cb.integration.toFixed(2), "—", cb.choice.toFixed(2), "1.00"], false, null, id);
    });
  }

  computeBtn.addEventListener("click", () => {
    const a = drawerA.getNormalized();
    const b = drawerB.getNormalized();
    const silhouetteOk = !!(a && b);
    document.getElementById("silhouetteResults").style.display = silhouetteOk ? "contents" : "none";

    const roomStateA = roomEditorA.getState();
    const roomStateB = roomEditorB.getState();
    const topoOk = !!(roomStateA && roomStateA.rooms.length && roomStateB && roomStateB.rooms.length);
    document.getElementById("topologyResults").style.display = topoOk ? "" : "none";
    document.getElementById("hierarchyResults").style.display = topoOk ? "" : "none";
    if (topoOk) {
      const graphA = buildRoomGraph(roomStateA);
      const graphB = buildRoomGraph(roomStateB);
      const topoResult = graphEditDistance(graphA, graphB);
      renderTopologyResult(topoResult, roomStateA, roomStateB, graphA, graphB);
      const hieResult = hierarchySimilarity(graphA, graphB);
      renderHierarchyResult(hieResult, roomStateA, roomStateB, document.getElementById("bubbleSvgA"), document.getElementById("bubbleSvgB"));
    }

    if (!silhouetteOk) {
      if (topoOk) document.getElementById("results").classList.add("show");
      return;
    }

    const defaultOverlap = overlapGrid(a, b, 140);
    const best = bestRotationOverlap(a, b, 72);

    let efdDist = null, efdV = null, aAligned = a, bEfdAligned = b;
    if (a.length >= 4 && b.length >= 4) {
      efdDist = efdDistance(a, b, 10);
      efdV = efdVerdict(efdDist);
      aAligned = rotate(a, efdCanonicalAngle(a, 10));
      bEfdAligned = rotate(b, efdCanonicalAngle(b, 10));
    }
    lastCompute = { a, bDefault: b, bBest: best.poly, defaultOverlap, best, aAligned, bEfdAligned, efdDist };
    rotationMode = "best";

    document.getElementById("efdReadout").textContent = efdDist === null ? "n/a (min. 4 titik)" : efdDist.toFixed(3) + " (" + efdV.label + ")";
    document.getElementById("scoreBig").textContent = (best.overlap * 100).toFixed(1) + "%";
    document.getElementById("angleReadout").textContent = best.angleDeg.toFixed(1) + "\u00b0 \u2192 " + (best.overlap * 100).toFixed(1) + "%";
    document.getElementById("defaultReadout").textContent = (defaultOverlap * 100).toFixed(1) + "%";
    setReadoutActive("best");

    const overlapHigh = best.overlap > 0.75, overlapLow = best.overlap <= 0.45;
    let note;
    if (best.overlap > 0.75) note = "Tumpang-tindih tinggi — kedua bentuk secara geometris sangat mirip setelah disejajarkan.";
    else if (best.overlap > 0.45) note = "Tumpang-tindih sedang — ada kemiripan struktural, tapi juga perbedaan yang cukup berarti.";
    else note = "Tumpang-tindih rendah — kedua bentuk berbeda secara mendasar, bukan sekadar beda orientasi.";

    if (efdV) {
      const overlapVerdictHigh = overlapHigh ? true : overlapLow ? false : null;
      if (efdV.verdictHigh !== null && overlapVerdictHigh !== null && efdV.verdictHigh !== overlapVerdictHigh) {
        note += " \u26a0\ufe0f Perhatian: tumpang-tindih dan jarak EFD TIDAK SEPAKAT (satu bilang mirip, satu bilang beda) — ini sinyal untuk diperiksa manual, bukan diabaikan.";
      } else if (efdV.verdictHigh === true && overlapVerdictHigh === true) {
        note += " Kedua metrik (tumpang-tindih & EFD) SEPAKAT: bentuknya mirip.";
      } else if (efdV.verdictHigh === false && overlapVerdictHigh === false) {
        note += " Kedua metrik (tumpang-tindih & EFD) SEPAKAT: bentuknya beda.";
      }
    }
    document.getElementById("scoreNote").textContent = note;

    document.getElementById("results").classList.add("show");
    requestAnimationFrame(() => requestAnimationFrame(() => drawOverlay(a, best.poly)));
  });

  function selectRotationMode(mode) {
    if (!lastCompute || rotationMode === mode) return;
    if (mode === "efd" && lastCompute.efdDist === null) return; // not enough points to define EFD
    rotationMode = mode;
    setReadoutActive(mode);
    const polyA = mode === "efd" ? lastCompute.aAligned : lastCompute.a;
    const polyB = mode === "best" ? lastCompute.bBest : mode === "efd" ? lastCompute.bEfdAligned : lastCompute.bDefault;
    if (mode !== "efd") {
      const overlap = mode === "best" ? lastCompute.best.overlap : lastCompute.defaultOverlap;
      document.getElementById("scoreBig").textContent = (overlap * 100).toFixed(1) + "%";
    } else {
      document.getElementById("scoreBig").textContent = lastCompute.efdDist.toFixed(3);
    }
    drawOverlay(polyA, polyB);
  }
  document.getElementById("readoutBest").addEventListener("click", () => selectRotationMode("best"));
  document.getElementById("readoutDefault").addEventListener("click", () => selectRotationMode("default"));
  document.getElementById("readoutEfd").addEventListener("click", () => selectRotationMode("efd"));

  function drawOverlay(a, bAligned) {
    const canvas = document.getElementById("canvasOverlay");
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const all = a.concat(bAligned);
    const b_ = bounds(all);
    const pad = 0.9;
    const spanX = (b_.maxX - b_.minX) || 1, spanY = (b_.maxY - b_.minY) || 1;
    const scale = Math.min(rect.width, rect.height) * pad / Math.max(spanX, spanY);
    const cx = rect.width / 2, cy = rect.height / 2;
    const midX = (b_.minX + b_.maxX) / 2, midY = (b_.minY + b_.maxY) / 2;
    const toScreen = ([x, y]) => [cx + (x - midX) * scale, cy - (y - midY) * scale];

    function fillPoly(pts, color) {
      ctx.beginPath();
      pts.forEach((p, i) => { const [x, y] = toScreen(p); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = color.replace("0.55", "1").replace("0.28", "1");
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }
    ctx.globalCompositeOperation = "source-over";
    fillPoly(a, "rgba(46,76,122,0.5)");
    ctx.globalCompositeOperation = "multiply";
    fillPoly(bAligned, "rgba(168,70,47,0.5)");
    ctx.globalCompositeOperation = "source-over";
  }

  checkReady();
})();
