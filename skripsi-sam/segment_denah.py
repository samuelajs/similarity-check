"""Gambar3: one polygon per label, following the real walls (not a bounding box).

Two MEETING ROOM texts are two rooms. The dashed line between them is the
folding door. The unlabeled strip their doors open into is circulation.
"""
import base64
import json
import re
import sys
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
try:
    import pymupdf
except ImportError:
    pymupdf = None
HERE = __import__("pathlib").Path(__file__).resolve().parent
SRC = str(HERE / "contoh" / "Gambar3.jpeg")
OUT = str(HERE / "hasil")


def runs_on_line(mask_1d, bridge, min_len):
    xs = np.where(mask_1d > 0)[0]
    if len(xs) == 0:
        return []
    groups = []
    start = prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        gap = x - prev - 1
        if gap > bridge:
            if prev - start + 1 >= min_len:
                groups.append((start, prev))
            start = x
        prev = x
    if prev - start + 1 >= min_len:
        groups.append((start, prev))
    return groups


def wall_lines(wall, bridge=80, min_len=36, thickness=7):
    horiz = []
    for y in range(wall.shape[0]):
        for a, b in runs_on_line(wall[y], bridge, min_len):
            horiz.append((y, a, b))
    vert = []
    for x in range(wall.shape[1]):
        for a, b in runs_on_line(wall[:, x], bridge, min_len):
            vert.append((x, a, b))

    def cluster(segs):
        segs = sorted(segs, key=lambda s: (s[0], s[1]))
        groups = []
        for coord, a, b in segs:
            hit = None
            for g in groups:
                if abs(coord - g["c"]) <= thickness and not (b < g["a"] - bridge or a > g["b"] + bridge):
                    hit = g
                    break
            if hit is None:
                groups.append({"c": coord, "a": a, "b": b, "cs": [coord]})
            else:
                hit["cs"].append(coord)
                hit["c"] = int(np.median(hit["cs"]))
                hit["a"] = min(hit["a"], a)
                hit["b"] = max(hit["b"], b)
        return [(g["c"], g["a"], g["b"]) for g in groups if g["b"] - g["a"] >= min_len]

    return cluster(horiz), cluster(vert)



def instances(words):
    tokens = []
    for box, text, score in words or []:
        if float(score) < 0.45:
            continue
        raw = text.strip().upper().replace("−", "-")
        if raw.replace(".", "").replace("-", "").replace("+", "").isdigit():
            continue
        if len(raw) < 2:
            continue
        cx = float(np.mean([p[0] for p in box]))
        cy = float(np.mean([p[1] for p in box]))
        tokens.append([raw, cx, cy])
    tokens.sort(key=lambda t: (t[2], t[1]))
    groups = []
    for text, cx, cy in tokens:
        placed = False
        for g in groups:
            # x_thresh was 140 -- safe for spaced-out rectilinear labels (Gambar1-3, where
            # multi-word names either arrive pre-joined by OCR or are stacked vertically),
            # but too wide for dense layouts (Gambar4's radial pods): it merged two
            # DIFFERENT rooms' labels ~120-130px apart into one group, and the resulting
            # oversized bounding box then punched a hole straight through the real wall
            # between them (see organic_silhouette/room helpers below). 60 still groups
            # every genuine multi-word case tested and fixes the false merge.
            if abs(cy - g["cy"]) < 22 and abs(cx - g["cx"]) < 60:
                g["parts"].append((cy, cx, text))
                g["cx"] = float(np.mean([p[1] for p in g["parts"]]))
                g["cy"] = float(np.mean([p[0] for p in g["parts"]]))
                placed = True
                break
        if not placed:
            groups.append({"parts": [(cy, cx, text)], "cx": cx, "cy": cy})
    out = []
    for g in groups:
        name = " ".join(t for _, _, t in sorted(g["parts"]))
        name = name.replace("MEETINGROOM", "MEETING ROOM").replace("MEETING ROOM ROOM", "MEETING ROOM").replace("ROOM ROOM", "ROOM")
        if name == "ROOM":
            continue
        out.append((g["cx"], g["cy"], name))
    return out


def instances_with_boxes(words, x_thresh=60, y_thresh=22):
    """Same grouping as instances(), but also keeps each group's own OCR bounding box
    (union of its tokens' boxes) instead of collapsing to just a centroid -- needed so a
    label sitting close to a wall (common in dense/radial layouts) only has ITS OWN text
    punched out of the wall mask, not a blind fixed-radius circle that can eat into real
    wall material nearby. Kept separate from instances() so the already-validated
    rectilinear pipeline (which only ever needs centroids) is untouched.
    """
    tokens = []
    for box, text, score in words or []:
        if float(score) < 0.45:
            continue
        raw = text.strip().upper().replace("−", "-")
        if raw.replace(".", "").replace("-", "").replace("+", "").isdigit():
            continue
        if len(raw) < 2:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        tokens.append([raw, cx, cy, min(xs), min(ys), max(xs), max(ys)])
    tokens.sort(key=lambda t: (t[2], t[1]))
    groups = []
    for text, cx, cy, x0, y0, x1, y1 in tokens:
        placed = False
        for g in groups:
            if abs(cy - g["cy"]) < y_thresh and abs(cx - g["cx"]) < x_thresh:
                g["parts"].append((cy, cx, text))
                g["cx"] = float(np.mean([p[1] for p in g["parts"]]))
                g["cy"] = float(np.mean([p[0] for p in g["parts"]]))
                g["x0"] = min(g["x0"], x0); g["y0"] = min(g["y0"], y0)
                g["x1"] = max(g["x1"], x1); g["y1"] = max(g["y1"], y1)
                placed = True
                break
        if not placed:
            groups.append({"parts": [(cy, cx, text)], "cx": cx, "cy": cy, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    out = []
    for g in groups:
        name = " ".join(t for _, _, t in sorted(g["parts"]))
        name = name.replace("MEETINGROOM", "MEETING ROOM").replace("MEETING ROOM ROOM", "MEETING ROOM").replace("ROOM ROOM", "ROOM")
        if name == "ROOM":
            continue
        out.append((g["cx"], g["cy"], name, (g["x0"], g["y0"], g["x1"], g["y1"])))
    return out


def organic_wall_mask(gray, boxed_labels=(), thresh_gray=150, thick_dist=1.85, min_blob_area=150, margin=4):
    """Orientation-agnostic wall material: thickness-thresholded ink (same test as the
    rectilinear pipeline's ink_lines/outer_silhouette use to separate real walls from thin
    dimension/construction lines), then drop small isolated blobs (arrowheads, dimension
    ticks) by AREA rather than by morphological opening -- opening erodes based on local
    width and was found to also destroy genuinely thin-but-long real wall strokes, while an
    area filter leaves them intact since their total extent is large even when narrow. Each
    label's own OCR bounding box (not a blind fixed-radius circle) is punched out first so a
    label sitting close to a wall doesn't take a chunk of real wall material with it.
    """
    ink = (gray < thresh_gray).astype(np.uint8)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    thick = (dist >= thick_dist).astype(np.uint8)
    h, w = gray.shape
    for _cx, _cy, _name, (x0, y0, x1, y1) in boxed_labels:
        xa, ya = max(0, int(x0) - margin), max(0, int(y0) - margin)
        xb, yb = min(w, int(x1) + margin), min(h, int(y1) + margin)
        thick[ya:yb, xa:xb] = 0
    if min_blob_area > 0:
        num, comp, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
        keep = np.zeros_like(thick)
        for cid in range(1, num):
            if stats[cid, cv2.CC_STAT_AREA] >= min_blob_area:
                keep[comp == cid] = 1
        thick = keep
    return thick


def rectilinear_wall_fit_ratio(bgr):
    """Fraction of this drawing's real wall material (organic_wall_mask, any angle) that
    the H/V-projection method (_rectilinear_wall_mask) actually captures -- a direct,
    label-independent signal for whether the rectilinear room-tracing pipeline (analyze())
    is a good geometric fit for this building. Tried first via indirect symptoms of a bad
    rectilinear result (duplicate fallback boxes, count of leftover "sirkulasi" blobs) and
    found those unreliable: a genuinely good rectilinear result (Gambar2) can have MORE
    leftover circulation blobs than a genuinely broken circular one (Gambar6), once you
    account for how many OCR labels each happens to have -- symptom-counting doesn't
    generalize, but measuring the wall geometry itself directly does. Verified across all
    8 reference images: true rectilinear plans (Gambar1-3) score 0.97-0.99; every plan
    with real diagonal/curved wall material (Gambar4-8, including two never-tested-before
    fully circular radial plans) scores 0.00-0.47 -- a wide, clean margin.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    organic_px = int(organic_wall_mask(gray, boxed_labels=()).sum())
    if organic_px == 0:
        return 1.0
    rect_px = int(_rectilinear_wall_mask(bgr).sum())
    return rect_px / organic_px


def organic_silhouette(bgr, boxed_labels=(), epsilon=6.0):
    """Building envelope, orientation-agnostic wall detection, merged-mass strategy: for
    plans with curved/diagonal wall material where the mass is (or should be treated as)
    SEVERAL DISCONNECTED pieces meant as one site/plan -- e.g. several separate circular
    pods with no wall physically joining them. Self-calibrates a closing radius from the
    drawing's own real gaps (_merge_bottleneck_radius()) to bridge every piece into one,
    then traces that merged mask's outer contour -- a smooth "union of the real shapes"
    that preserves the concave notches between pieces, instead of convex_hull straight-
    lining across every notch. Falls back to the hull if closing still doesn't fully
    connect everything. Disconnected wall material isn't unique to curved plans -- see
    rectilinear_hull_silhouette() for the same strategy on a straight-walled plan.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = organic_wall_mask(gray, boxed_labels=boxed_labels)
    return _merged_silhouette(mask, epsilon=epsilon)


def organic_traced_silhouette(bgr, boxed_labels=(), epsilon=8.0):
    """Building envelope, orientation-agnostic wall detection, traced-contour strategy: for
    a SINGLE connected curvy/diagonal building mass (no disconnected pieces) -- traces the
    outer boundary of the wall material directly, same idea as outer_silhouette() but
    without assuming the walls are axis-aligned. More geometrically faithful than the hull
    when the mass really is one connected piece (keeps concave curves instead of
    straight-lining them), but will produce an artifact wherever the wall material has an
    unbridged gap (e.g. a single entrance door interrupting the exterior wall) -- no
    gap-bridging is attempted here, since every attempt tried for this tool turned out
    fragile (see analyze_organic() docstring); pick the hull strategy instead for a plan
    with any real gap in its outer wall.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = organic_wall_mask(gray, boxed_labels=boxed_labels)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(p[0][0]), int(p[0][1])] for p in approx]


def _runs(binary_row, bridge, min_len):
    xs = np.where(binary_row > 0)[0]
    if len(xs) == 0:
        return []
    out = []
    start = prev = int(xs[0])
    for x in list(xs[1:]) + [10**9]:
        x = int(x)
        if x > prev + 1 + bridge:
            if prev - start + 1 >= min_len:
                out.append((start, prev))
            start = x
        prev = x
    return out


def ink_lines(gray, holes=(), bridge_door=100):
    """Thick wall lines, with door gaps joined. Long thin exterior walls are added whole.
    Letter strokes are not walls, so a word in the middle of a room is not bridged to the walls.
    """
    ink = (gray < 100).astype(np.uint8)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    thick = (dist >= 2.2).astype(np.uint8)
    for cx, cy, _name in holes:
        cv2.circle(thick, (int(cx), int(cy)), 22, 0, -1)
    horiz, vert = wall_lines(thick, bridge=55, min_len=60, thickness=8)
    h, w = gray.shape
    for y in range(h):
        for a, b in _runs(ink[y], 2, 120):
            if float(dist[y, a:b + 1].max()) < 2.2:
                horiz.append((y, a, b))
    for x in range(w):
        for a, b in _runs(ink[:, x], 2, 120):
            if float(dist[a:b + 1, x].max()) < 2.2:
                vert.append((x, a, b))
    return horiz, vert


def folding_line(gray):
    """The real folding door: several similar dashes in a row, not a dimension string."""
    h, w = gray.shape
    best = None
    for y in range(h):
        ink = np.where(gray[y] < 90)[0]
        if len(ink) < 6:
            continue
        groups = []
        start = prev = int(ink[0])
        for x in ink[1:]:
            x = int(x)
            if x > prev + 3:
                groups.append((start, prev))
                start = x
            prev = x
        groups.append((start, prev))
        pieces = [(a, b) for a, b in groups if 8 <= b - a + 1 <= 26]
        if len(pieces) < 5:
            continue
        # Keep a consecutive run of those pieces with small gaps.
        run = [pieces[0]]
        runs = []
        for a, b in pieces[1:]:
            if a - run[-1][1] <= 22:
                run.append((a, b))
            else:
                if len(run) >= 5:
                    runs.append(run)
                run = [(a, b)]
        if len(run) >= 5:
            runs.append(run)
        for run in runs:
            span = run[-1][1] - run[0][0]
            if span < 120:
                continue
            if best is None or span > best[0]:
                best = (span, y, run[0][0], run[-1][1])
    return best


def covers(line, pos, slack):
    _c, a, b = line
    return a - slack <= pos <= b + slack


def nearest(lines, origin, pos, direction, slack):
    """direction +1 looks for coord greater than origin; -1 for smaller."""
    best = None
    for coord, a, b in lines:
        if direction > 0 and coord <= origin + 4:
            continue
        if direction < 0 and coord >= origin - 4:
            continue
        if not covers((coord, a, b), pos, slack):
            continue
        if best is None or abs(coord - origin) < abs(best - origin):
            best = coord
    return best


def overlap_box(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right - left <= 1 or bottom - top <= 1:
        return None
    return [left, top, right, bottom]


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _separate_fallback_boxes(fallback_rooms, fixed_boxes):
    """Shrinks fallback rectangle rooms (nearest-wall guess, used only when the
    flood-fill trace below could not resolve a label safely) so they never overlap
    each other or any already wall-traced room's bounding box. Fixed boxes are never
    modified -- a correctly traced polygon is trusted over a rectangle guess.
    """
    boxes = [r["_box"] for r in fallback_rooms]
    for _ in range(12):
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                hit = overlap_box(boxes[i], boxes[j])
                if hit:
                    left, top, right, bottom = hit
                    cut_vertical = (right - left) <= (bottom - top)
                    shrink = boxes[j]
                    if cut_vertical:
                        if shrink[0] >= left: shrink[0] = right
                        else: shrink[2] = left
                    else:
                        if shrink[1] >= top: shrink[1] = bottom
                        else: shrink[3] = top
                    changed = True
            for fb in fixed_boxes:
                hit = overlap_box(boxes[i], fb)
                if hit:
                    left, top, right, bottom = hit
                    cut_vertical = (right - left) <= (bottom - top)
                    shrink = boxes[i]
                    if cut_vertical:
                        if shrink[0] >= left: shrink[0] = right
                        else: shrink[2] = left
                    else:
                        if shrink[1] >= top: shrink[1] = bottom
                        else: shrink[3] = top
                    changed = True
        if not changed:
            break


def _rectilinear_wall_mask(bgr):
    """Wall-pixel mask for the rectilinear silhouette pipeline: H/V runs of thick ("core")
    ink at least 40px long. Same detection used by both outer_silhouette() (traces a single
    contour -- needs the walls to form one connected building mass) and
    rectilinear_hull_silhouette() (convex hull -- works even when the mass is split into
    several disconnected pieces, which can happen on a rectilinear plan too, not only an
    organic one, e.g. two separate rectangular buildings on the same site with no wall
    physically joining them).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 110).astype(np.uint8)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    height, width = gray.shape
    walls = np.zeros((height, width), np.uint8)
    core = dist >= 1.85

    def paint_runs(indices, write):
        if len(indices) == 0:
            return
        start = prev = int(indices[0])
        for cur in list(indices[1:]) + [10**9]:
            cur = int(cur)
            if cur > prev + 2:
                if prev - start >= 40:
                    write(start, prev)
                start = cur
            prev = cur

    for x in range(width):
        paint_runs(np.where(core[:, x])[0], lambda a, b, x=x: walls.__setitem__((slice(a, b + 1), x), 1))
    for y in range(height):
        paint_runs(np.where(core[y])[0], lambda a, b, y=y: walls.__setitem__((y, slice(a, b + 1)), 1))
    return walls


def outer_silhouette(bgr, progress=None):
    """Outer building mass via orientation-agnostic wall detection (ink-THICKNESS
    threshold, not H/V line projection) -- handles axis-aligned, diagonal, and curved
    wall material with the same logic, and naturally ignores furniture/fixture linework
    (beds, tables, chairs, sinks, stair hatching, ...) since only sufficiently thick ink
    counts as wall material in the first place, regardless of how much thin furniture ink
    is nearby. Replaced the earlier H/V-projection-only method (still available as
    _rectilinear_wall_mask()/rectilinear_hull_silhouette() for the hull-strategy case)
    after it completely missed a real plan whose walls are filled a mid-gray rather than
    near-black (Gambar5, a diagonal-walled hexagonal plan) -- confirmed via regression
    that this produces the SAME silhouette point counts on purely rectilinear plans too
    (Gambar1/2/3), so there's no separate "mode" to choose between for this step.

    Wall material broken into several pieces by door/corner gaps (a straight-but-
    sloped plan's walls are just as likely to be split up this way as an axis-aligned
    one) is first reconnected with SHORT precise lines between each gap's true nearest
    points (_bridge_wall_gaps()) rather than going straight to _merged_silhouette's
    isotropic closing, which needs an expensive, blob-shaped dilation sized for the
    single worst gap in the whole drawing.

    Verified against a real hand-corrected ground truth (Sam's saved DENAH_HEXAGON case)
    that precise bridging can go wrong in a way point-count alone never catches: if the
    disconnected piece it reconnects is actually an INTERIOR partition wall (not a
    fragment of the outer envelope itself) sitting close to the envelope, findContours'
    external contour treats the newly-bridged interior piece as a "hole" cut into the
    silhouette -- a thin correct-looking outline that's near-EMPTY inside once its true
    enclosed area (the shoelace/fillPoly area, not just its point count or bounding box)
    is checked. Comparing the bridged result's area against the _plain_ (unbridged)
    result was tried first and rejected: an isotropic closing radius large enough to
    bridge a real gap tends to round corners outward, which can make the CRUDER result
    enclose MORE raw area than a precise, correctly-fitted one without that being a sign
    of anything wrong -- confirmed on the hexagon case, where the precise result visually
    and quantitatively matched Sam's hand-verified ground truth despite having a smaller
    area than the plain fallback. The reliable signal instead is the bridged result's
    area against the wall mask's OWN convex hull area: a real hole-cut pathology collapses
    that ratio dramatically (measured 0.155 on the broken case) far below every legitimate
    result measured (0.591-1.0, hexagon included) -- so only fall back to the plain result
    when the bridged one is implausibly small relative to the mask's own extent, not
    merely smaller than the alternative.
    """
    if progress:
        progress("Mendeteksi dinding", 1, 2)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = organic_wall_mask(gray, boxed_labels=())
    if progress:
        progress("Menjiplak siluet", 2, 2)
    return _trace_silhouette_robust(mask)


def _trace_silhouette_robust(mask):
    """Shared by outer_silhouette() and analyze_organic(): precise gap-bridging first,
    falling back to isotropic closing only when bridging produces an implausible result
    (the hull-ratio check below) -- see outer_silhouette()'s docstring for the full
    reasoning and the ground-truth verification (Sam's DENAH_HEXAGON case) this was
    checked against. Pulled out as its own function after `analyze_organic()` was found
    calling `_merged_silhouette(mask)` directly instead of going through this, on a
    radial/circular plan (Gambar6) whose interior spoke walls hit the exact same
    hole-cut pathology outer_silhouette() was already fixed for -- confirmed via OCR
    labels like "KITCHEN"/"LIVING AREA" sitting well inside the real building testing as
    OUTSIDE the buggy silhouette (cv2.pointPolygonTest distance around -60 to -75px).
    """
    if not mask.any():
        return []
    bridged_pts = _merged_silhouette(_bridge_wall_gaps(mask))
    bridged_area = cv2.contourArea(np.array(bridged_pts, dtype=np.int32)) if len(bridged_pts) >= 3 else -1
    ys, xs = np.where(mask > 0)
    hull_area = cv2.contourArea(cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32)))
    if hull_area > 0 and bridged_area / hull_area >= 0.4:
        return bridged_pts
    return _merged_silhouette(mask)


def _hull_from_mask(mask, epsilon=8.0):
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return []
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    return [[int(p[0][0]), int(p[0][1])] for p in approx]


def _merge_bottleneck_radius(mask, min_component_area=800, margin=8):
    """Smallest closing radius that bridges ALL of a mask's substantial connected pieces
    into one, derived from the drawing's own real gaps instead of a guessed constant:
    pairwise nearest-point distance between every pair of "big" components (small blobs
    below min_component_area, e.g. dimension ticks/arrowheads, excluded so they can't drag
    the estimate around), then the maximum edge in the MINIMUM SPANNING TREE over those
    distances -- the one bottleneck gap that, once bridged, guarantees everything else was
    already reachable through gaps no larger. Closing needs roughly half that gap (each
    side's dilation meets the other's), plus a small margin for the kernel's discretization
    (an ellipse structuring element only approximates a true disk).
    """
    num, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    big_ids = [cid for cid in range(1, num) if stats[cid, cv2.CC_STAT_AREA] >= min_component_area]
    if len(big_ids) <= 1:
        return margin
    n = len(big_ids)
    dist_mat = np.full((n, n), np.inf)
    for i, cid in enumerate(big_ids):
        not_this = (comp != cid).astype(np.uint8)
        dt = cv2.distanceTransform(not_this, cv2.DIST_L2, 5)
        for j, cid2 in enumerate(big_ids):
            if i == j:
                continue
            ys, xs = np.where(comp == cid2)
            dist_mat[i, j] = float(dt[ys, xs].min())
    in_tree = [False] * n
    in_tree[0] = True
    max_edge = 0.0
    for _ in range(n - 1):
        best = (np.inf, -1)
        for i in range(n):
            if not in_tree[i]:
                continue
            for j in range(n):
                if in_tree[j]:
                    continue
                if dist_mat[i, j] < best[0]:
                    best = (dist_mat[i, j], j)
        if best[1] == -1:
            break
        max_edge = max(max_edge, best[0])
        in_tree[best[1]] = True
    return int(max_edge / 2) + margin


def _bridge_wall_gaps(mask, min_component_area=800, thickness=3):
    """Reconnect a wall mask's disconnected pieces with SHORT, PRECISE lines drawn
    directly between each gap's true nearest points, instead of _merged_silhouette's
    isotropic morphological closing -- which has to dilate the ENTIRE mask by half the
    largest real gap to reconnect anything, an expensive, blob-shaped operation that
    risks rounding off real corners when that gap is large. A door opening in an
    otherwise continuous wall is usually a small, targeted gap -- draw exactly that,
    exactly where it is, and nothing else needs to move.

    Bridges EVERY pairwise gap below a self-calibrated cutoff, not just a minimum
    spanning tree -- an MST only guarantees every piece is CONNECTED (n-1 edges, a tree,
    no cycles by definition), which is not the same as CLOSED: wall pieces arranged
    around a building's perimeter need every adjacent pair joined to form a ring
    enclosing the interior, otherwise the "outer" contour ends up tracing out along one
    side of the tree and back along the other, a near-zero-area sliver despite looking
    right at a glance (caught via a real ground-truth comparison: a spanning-tree version
    of this function visually looked like a correct hexagon outline but had ~12x less
    area than Sam's hand-verified silhouette for the same drawing). Bridging every close
    pair costs nothing extra worth avoiding here (still just short line segments, not an
    isotropic dilation) and is what actually closes the loop.

    Only bridges the cluster of SMALL gaps this way (a real door, architecturally, is a
    small multiple of its own wall thickness) -- a large outlier gap is a fundamentally
    different situation this function deliberately leaves alone: a site genuinely split
    into several separate masses connected only by open space (e.g. several circular pods
    sharing a floor line, not a wall) needs _merged_silhouette's smooth blob-preserving
    closing to trace the real curve between them -- a straight "shortcut" line through
    that open space would cut across it, not follow it. The cutoff is found empirically
    per-drawing: sort every pairwise gap and look for the largest proportional jump
    between consecutive sizes (a cluster of similar small door-gaps, then a jump to a
    qualitatively different kind of gap) rather than guessing an absolute pixel threshold
    that wouldn't generalize across zoom levels.
    """
    num, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    big_ids = [cid for cid in range(1, num) if stats[cid, cv2.CC_STAT_AREA] >= min_component_area]
    n = len(big_ids)
    if n <= 1:
        return mask
    comp_pts = {i: np.column_stack(np.where(comp == cid)[::-1]) for i, cid in enumerate(big_ids)}
    pairs = []  # (dist, pt_on_i, pt_on_j) for every i<j
    for i, cid in enumerate(big_ids):
        not_this = (comp != cid).astype(np.uint8)
        dt = cv2.distanceTransform(not_this, cv2.DIST_L2, 5)
        for j in range(i + 1, n):
            pts = comp_pts[j]
            dvals = dt[pts[:, 1], pts[:, 0]]
            k = int(np.argmin(dvals))
            pt_on_j = (int(pts[k, 0]), int(pts[k, 1]))
            not_j = (comp != big_ids[j]).astype(np.uint8)
            dt_j = cv2.distanceTransform(not_j, cv2.DIST_L2, 5)
            pts_i = comp_pts[i]
            dvals_i = dt_j[pts_i[:, 1], pts_i[:, 0]]
            ki = int(np.argmin(dvals_i))
            pt_on_i = (int(pts_i[ki, 0]), int(pts_i[ki, 1]))
            pairs.append((float(dvals[k]), pt_on_i, pt_on_j))

    if not pairs:
        return mask
    sorted_dists = sorted(d for d, _, _ in pairs)
    cutoff = sorted_dists[-1]
    if len(sorted_dists) > 1:
        biggest_jump, jump_cutoff = 0.0, cutoff
        for k in range(1, len(sorted_dists)):
            jump = sorted_dists[k] / max(sorted_dists[k - 1], 1e-6)
            if jump > biggest_jump:
                biggest_jump, jump_cutoff = jump, sorted_dists[k - 1]
        if biggest_jump >= 1.6:
            cutoff = jump_cutoff

    bridged = mask.copy()
    for dist, pt_i, pt_j in pairs:
        if dist <= cutoff:
            cv2.line(bridged, pt_i, pt_j, 1, thickness)
    return bridged


def _merged_silhouette(mask, epsilon=6.0, min_component_area=800):
    """Silhouette strategy for a mass split into several disconnected pieces meant as one
    site: close every real gap, then trace the merged mask's own outer contour -- a smooth
    "union of the real shapes" (concave notches between pieces preserved, rounded by the
    closing kernel) rather than convex_hull's straight lines cutting across every notch.

    The starting radius comes from _merge_bottleneck_radius() (half the largest gap in the
    minimum spanning tree over real component gaps), but that theoretical gap/2 estimate
    turned out to consistently undershoot in practice -- an OpenCV ellipse structuring
    element is only an approximation of a true Euclidean disk, especially at the large
    sizes this needs, so it doesn't reach as far as the math predicts. Rather than hardcode
    a fudge factor that would just be a different guess, this escalates (x1.5 per attempt)
    from the theoretical starting point until closing actually produces one component,
    self-correcting to whatever the kernel really needs on this image. Falls back to the
    convex hull only if that still doesn't fully connect everything within a sane number of
    attempts, so a silhouette is always returned.

    Known caveat: this operates on the WHOLE wall mask, interior partitions included, not
    just the outer envelope. For a plan whose exterior really is split into separate
    masses, each mass's own walls are normally already self-connected, so the merge only
    bridges between masses and the result stays clean. But if an INTERIOR wall happens to
    be disconnected from the exterior ring too (a detection artifact, not a real separate
    mass) it still gets pulled into the merge, and the traced contour will dip in and back
    out around it -- a jagged intrusion, not a clean envelope. Worth knowing before turning
    this on for a plan where the disconnection is a detection quirk rather than the
    building genuinely being multiple physically separate masses.
    """
    radius = _merge_bottleneck_radius(mask, min_component_area=min_component_area)
    closed = mask
    for _ in range(6):
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
        num, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        big = sum(1 for cid in range(1, num) if stats[cid, cv2.CC_STAT_AREA] >= min_component_area)
        if big <= 1:
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []
            contour = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            return [[int(p[0][0]), int(p[0][1])] for p in approx]
        radius = int(radius * 1.5) + 1
    return _hull_from_mask(mask, epsilon=epsilon)


def rectilinear_hull_silhouette(bgr):
    """Merged-mass silhouette (see _merged_silhouette()) of the same rectilinear wall-pixel
    mask outer_silhouette() uses -- for a straight-walled plan whose building mass is split
    into several disconnected pieces (no wall physically joining them) that should still be
    read as one site/denah.
    """
    walls = _rectilinear_wall_mask(bgr)
    return _merged_silhouette(walls)


def _connect_wall_corners(barrier, horiz, vert, max_gap=60, thickness=3):
    """Two wall segments on different axes that should meet at a corner (a vertical wall
    ending, a horizontal wall picking up a short jog over -- common right at a door
    threshold, where the swing-arc symbol interrupts detection right at the turn) can be
    left a short gap apart that neither line's own `bridge` tolerance closes, since that
    only bridges gaps along a single line's own axis. Connect any horizontal/vertical
    segment endpoints within `max_gap` of each other with a short straight barrier line --
    surgical (only at plausible corners), unlike a blanket morphological close which also
    erases small rooms whose own width is smaller than the closing kernel.
    """
    endpoints_h = [(coord, a) for coord, a, _b in horiz] + [(coord, b) for coord, _a, b in horiz]
    endpoints_v = [(a, coord) for coord, a, _b in vert] + [(b, coord) for coord, _a, b in vert]
    for hy, hx in endpoints_h:
        for vy, vx in endpoints_v:
            if abs(hy - vy) + abs(hx - vx) <= max_gap:
                cv2.line(barrier, (hx, hy), (vx, vy), 1, thickness)
    return barrier


def build_barrier_mask(horiz, vert, height, width, thickness=3, corner_gap=90):
    """Rasterize clustered wall segments as solid barriers across their FULL bridged
    span [a, b] -- not just where ink physically exists. wall_lines()/ink_lines() already
    bridge door-sized gaps when clustering, so this reconstructs each wall as continuous,
    closing door openings for the purpose of containing a room's flood-fill below.
    """
    barrier = np.zeros((height, width), np.uint8)
    half = thickness // 2
    for coord, a, b in horiz:
        y0, y1 = max(0, coord - half), min(height, coord + half + 1)
        x0, x1 = max(0, a), min(width, b + 1)
        if x1 > x0 and y1 > y0:
            barrier[y0:y1, x0:x1] = 1
    for coord, a, b in vert:
        x0, x1 = max(0, coord - half), min(width, coord + half + 1)
        y0, y1 = max(0, a), min(height, b + 1)
        if x1 > x0 and y1 > y0:
            barrier[y0:y1, x0:x1] = 1
    if corner_gap > 0:
        barrier = _connect_wall_corners(barrier, horiz, vert, max_gap=corner_gap, thickness=thickness)
    return barrier


def component_polygon(comp, component_id, epsilon=10.0):
    mask = (comp == component_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = [[int(p[0][0]), int(p[0][1])] for p in approx]
    return pts if len(pts) >= 3 else None


def trace_room_polygons(gray, horiz, vert, silhouette_pts, labels, max_area_frac=0.5):
    """Flood-fill each room's real footprint (L-shapes, U-shapes, etc. included) from its
    label position, bounded by the door-gap-bridged wall barrier and the outer silhouette
    (so a small gap in an exterior wall never leaks the flood outside the building).

    Returns {label_index: (points, component_id)} for labels that traced to a plausible
    single-room area. A label whose component covers an implausible fraction of the whole
    building (a sign two rooms merged into one blob, the same failure mode the silhouette
    algorithm had before) is left OUT here so the caller falls back to the nearest-wall
    rectangle for that label instead of trusting a bad trace.
    """
    height, width = gray.shape
    barrier = build_barrier_mask(horiz, vert, height, width)
    if silhouette_pts and len(silhouette_pts) >= 3:
        inside = np.zeros((height, width), np.uint8)
        cv2.fillPoly(inside, [np.array(silhouette_pts, dtype=np.int32)], 1)
        barrier[inside == 0] = 1
        silhouette_area = float(inside.sum())
    else:
        silhouette_area = float(height * width)

    free = (barrier == 0).astype(np.uint8)
    num, comp = cv2.connectedComponents(free, connectivity=4)

    used_components = set()
    assigned = {}
    for i, (cx, cy, _name) in enumerate(labels):
        row, col = int(round(cy)), int(round(cx))
        if not (0 <= row < height and 0 <= col < width):
            continue
        cid = int(comp[row, col])
        if cid == 0 or cid in used_components:
            continue
        area = float((comp == cid).sum())
        if area > max_area_frac * silhouette_area:
            continue
        pts = component_polygon(comp, cid)
        if pts is None:
            continue
        used_components.add(cid)
        assigned[i] = (pts, cid)

    return assigned, comp, num, used_components


def detect_adjacency(gray, comp, horiz, vert, room_comp_ids, min_gap=12, max_gap=90, thickness=3, min_wall_confidence=0.55, side_offset=10):
    """For each wall segment separating two different room components, look at the
    ORIGINAL ink (not the door-bridged span used to contain the flood-fill above) for a
    sub-run with no ink -- a physical door opening -- and record it as a connection
    between those two rooms. `room_comp_ids`: {component_id: room_id}, only components
    that ended up as real rooms/circulation (not raw background).

    Floor plans often have more than one nearly-parallel line detected close together at
    a doorway (a double-line wall representation, a door-leaf symbol, a jamb/reveal line)
    -- scanning any one of those in isolation for an ink gap can misattribute which two
    rooms it actually separates, since a lone sample right at the gap can land past a
    *different* nearby line into an unrelated sliver. To stay robust across different
    drawings (not just this one), a segment is only trusted as a real shared wall between
    two rooms if it consistently separates the SAME two components across most of its own
    inked length -- not just at the gap itself.
    """
    ink = (gray < 110).astype(np.uint8)
    height, width = gray.shape
    edges = {}

    def consider(cid_a, cid_b, mid_point):
        if cid_a == cid_b or cid_a not in room_comp_ids or cid_b not in room_comp_ids:
            return
        key = frozenset((room_comp_ids[cid_a], room_comp_ids[cid_b]))
        if len(key) == 2 and key not in edges:
            edges[key] = mid_point

    def gaps_in_span(ink_1d, a, b):
        run_start = None
        for x in range(a, b + 1):
            empty = not ink_1d[x]
            if empty and run_start is None:
                run_start = x
            elif not empty and run_start is not None:
                length = x - run_start
                if min_gap <= length <= max_gap:
                    yield (run_start + x - 1) // 2
                run_start = None
        if run_start is not None:
            length = b + 1 - run_start
            if min_gap <= length <= max_gap:
                yield (run_start + b) // 2

    def dominant_pair(coord, a, b, axis):
        """Sample the component on each side at every INKED position along the
        segment (not just at a candidate gap) and return the two components that
        dominate each side, only if each side is consistently a single room across
        a solid majority of samples -- otherwise this segment is too ambiguous to
        trust for door detection at all.
        """
        side_a, side_b, inked = {}, {}, 0
        for pos in range(a, b + 1):
            if axis == "h":
                if not ink[coord, pos]:
                    continue
                p0, p1 = coord - side_offset, coord + side_offset
                if p0 < 0 or p1 >= height:
                    continue
                c0, c1 = int(comp[p0, pos]), int(comp[p1, pos])
            else:
                if not ink[pos, coord]:
                    continue
                p0, p1 = coord - side_offset, coord + side_offset
                if p0 < 0 or p1 >= width:
                    continue
                c0, c1 = int(comp[pos, p0]), int(comp[pos, p1])
            inked += 1
            if c0: side_a[c0] = side_a.get(c0, 0) + 1
            if c1: side_b[c1] = side_b.get(c1, 0) + 1
        if inked < min_gap:  # segment barely has any real wall material -- too thin to trust
            return None
        dom_a = max(side_a.items(), key=lambda kv: kv[1]) if side_a else None
        dom_b = max(side_b.items(), key=lambda kv: kv[1]) if side_b else None
        if not dom_a or not dom_b or dom_a[0] == dom_b[0]:
            return None
        if dom_a[1] / inked < min_wall_confidence or dom_b[1] / inked < min_wall_confidence:
            return None
        return dom_a[0], dom_b[0]

    def door_symbol_present(coord, pos, axis, window=55, min_ink=40):
        """Not full arc/circle detection, but a real confirmation signal beyond
        "there's a doorway-width gap": a drawn door symbol (swing arc + leaf)
        puts a meaningful amount of extra ink just inside one of the two rooms,
        away from the wall's own thin band. A plain structural opening with no
        door drawn has no such ink nearby. Checking both sides and taking the
        stronger one keeps this robust to which side the door was drawn on.
        """
        if axis == "h":
            x0, x1 = max(0, pos - window // 2), min(width, pos + window // 2)
            below = int(ink[coord + 4:min(height, coord + window), x0:x1].sum())
            above = int(ink[max(0, coord - window):max(0, coord - 4), x0:x1].sum())
            return max(above, below) >= min_ink
        y0, y1 = max(0, pos - window // 2), min(height, pos + window // 2)
        right = int(ink[y0:y1, coord + 4:min(width, coord + window)].sum())
        left = int(ink[y0:y1, max(0, coord - window):max(0, coord - 4)].sum())
        return max(left, right) >= min_ink

    for coord, a, b in horiz:
        a, b = max(0, a), min(width - 1, b)
        if b <= a:
            continue
        pair = dominant_pair(coord, a, b, "h")
        if not pair:
            continue
        y0, y1 = max(0, coord - thickness - 2), min(height, coord + thickness + 3)
        row_ink = ink[y0:y1, :].max(axis=0)
        for gx in gaps_in_span(row_ink, a, b):
            if door_symbol_present(coord, gx, "h"):
                consider(pair[0], pair[1], [int(gx), int(coord)])

    for coord, a, b in vert:
        a, b = max(0, a), min(height - 1, b)
        if b <= a:
            continue
        pair = dominant_pair(coord, a, b, "v")
        if not pair:
            continue
        x0, x1 = max(0, coord - thickness - 2), min(width, coord + thickness + 3)
        col_ink = ink[:, x0:x1].max(axis=1)
        for gy in gaps_in_span(col_ink, a, b):
            if door_symbol_present(coord, gy, "v"):
                consider(pair[0], pair[1], [int(coord), int(gy)])

    return [{"a": pair[0], "b": pair[1], "door_at": mid} for pair, mid in
            ((tuple(k), v) for k, v in edges.items())]


def _drop_comb_patterns(lines, min_count=4, max_spacing=24, max_regularity=0.35, length_ratio=1.6):
    """Stair tread marks (or similar repeated hatching) show up as many short-to-medium
    wall-like lines packed close together at fairly even spacing -- a "comb" -- each one
    individually long enough to pass the normal wall-length filters, which chops the
    stair area into many tiny slivers instead of one room. A real wall is normally alone
    or paired with the opposite room's wall, not part of a tight, evenly-spaced repeating
    run of SIMILAR-length lines -- so lines are first grouped just by coordinate proximity
    (chaining through max_spacing gaps), then within each proximity group, sub-clustered
    by both overlapping span AND similar length (length_ratio) so that a real, much longer
    wall which merely happens to sit near a tread pattern's coordinate range doesn't get
    swept in and dropped along with it. A sub-cluster is only dropped as decorative when
    it has >= min_count members AND their spacing is regular (a coincidence of several
    unrelated walls landing near each other would not also be evenly spaced).
    """
    ordered = sorted(range(len(lines)), key=lambda i: lines[i][0])
    n = len(ordered)
    drop = set()
    i = 0
    while i < n:
        group = [ordered[i]]
        k = i + 1
        while k < n and lines[ordered[k]][0] - lines[group[-1]][0] <= max_spacing:
            group.append(ordered[k])
            k += 1

        used = [False] * len(group)
        for gi in range(len(group)):
            if used[gi]:
                continue
            idx0 = group[gi]
            a0, b0 = lines[idx0][1], lines[idx0][2]
            len0 = max(1, b0 - a0)
            sub = [idx0]
            used[gi] = True
            for gj in range(gi + 1, len(group)):
                if used[gj]:
                    continue
                idx1 = group[gj]
                a1, b1 = lines[idx1][1], lines[idx1][2]
                len1 = max(1, b1 - a1)
                overlap = min(b0, b1) - max(a0, a1)
                same_length = max(len0, len1) / min(len0, len1) <= length_ratio
                if same_length and overlap > 0.5 * max(len0, len1):
                    sub.append(idx1)
                    used[gj] = True
            if len(sub) >= min_count:
                coords = sorted(lines[idx][0] for idx in sub)
                gaps = [coords[t + 1] - coords[t] for t in range(len(coords) - 1)]
                mean_gap = sum(gaps) / len(gaps)
                variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                regularity = (variance ** 0.5) / mean_gap if mean_gap > 0 else 1.0
                if regularity <= max_regularity:
                    drop.update(sub)
        i = k
    return [seg for idx, seg in enumerate(lines) if idx not in drop]


def box_for(horiz, vert, cx, cy, slack):
    left = nearest(vert, cx, cy, -1, slack)
    right = nearest(vert, cx, cy, +1, slack)
    top = nearest(horiz, cy, cx, -1, slack)
    bot = nearest(horiz, cy, cx, +1, slack)
    return left, top, right, bot


def ray_box_for(mask, cx, cy, max_dist):
    """4-directional ray cast from (cx, cy) against a wall-pixel mask -- the organic-mode
    equivalent of box_for()'s nearest-wall rectangle (which needs pre-extracted H/V line
    segments; this walks the raster mask directly instead, so it works for curved/diagonal
    walls too). Gives a rough box approximating a room's real footprint from whatever real
    wall material is nearby in each direction, for when the exact shape is too hard to
    trace automatically -- a box with a plausible area beats no room at all for the
    area-dependent similarity metrics (component 2/3 both use relative floor area). A
    direction with no wall within max_dist (e.g. a genuinely open-plan side) just clamps to
    max_dist rather than the mask's edge, keeping the box a reasonable size instead of
    unbounded.
    """
    h, w = mask.shape
    cx, cy = int(round(cx)), int(round(cy))

    def cast(dx, dy):
        x, y, dist = cx, cy, 0
        while dist < max_dist:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                break
            if mask[ny, nx]:
                return nx, ny
            x, y = nx, ny
            dist += 1
        return x, y

    left_x, _ = cast(-1, 0)
    right_x, _ = cast(1, 0)
    _, top_y = cast(0, -1)
    _, bot_y = cast(0, 1)
    return left_x, top_y, right_x, bot_y


def _closing_radius_candidates(wall_mask, attempts=8):
    """Starting radius for room-separating gap closure, seeded from the wall material's
    OWN measured thickness (median distance-transform value on wall pixels) rather than
    a hardcoded pixel constant, then escalated x1.6 per attempt -- same "start
    conservative, escalate until it works, cap the attempts" pattern as
    _merged_silhouette's bottleneck-radius search. A door opening's real size relative to
    its own wall thickness varies enough across drawings (measured 4-12x across the two
    test plans this was tuned against) that betting on one ratio isn't reliable; trying a
    wide escalating range and letting the caller pick whichever attempt resolves the most
    rooms is what actually generalizes, not a better-guessed constant.
    """
    dist = cv2.distanceTransform(wall_mask, cv2.DIST_L2, 3)
    thick_vals = dist[wall_mask > 0]
    half_thickness = float(np.median(thick_vals)) if thick_vals.size else 4.0
    radius = max(8.0, half_thickness * 4.0)
    radii = []
    for _ in range(attempts):
        radii.append(int(round(radius)))
        radius *= 1.6
    return radii


def trace_rooms_organic(wall_mask, boxed_labels, height, width):
    """Flood-fill room tracing off the orientation-agnostic wall mask (real polygon,
    following whatever angle the walls actually are -- not restricted to horizontal/
    vertical) -- the precise counterpart to ray_box_for()'s box approximation, for plans
    whose walls genuinely enclose a room (a hexagonal or other straight-but-non-axis-
    aligned building is NOT "organic/curved" in any way that should block real tracing;
    only the door-sized gap in an otherwise straight wall needs bridging, same idea as a
    rectilinear plan's own door gaps). Tries several self-calibrated closing radii
    (_closing_radius_candidates) and keeps whichever resolves the most labels to their
    own distinct, plausibly-sized enclosed region.

    The flood-fill boundary uses the wall mask's own CONVEX HULL, not silhouette_pts
    directly -- a radial/circular plan (Gambar6/7) can have a genuinely large gap in its
    outer ring (an open archway, or ink too thin to pass the wall threshold there) that a
    densely-connected interior spoke network still bridges into one connected component,
    so the traced silhouette can have a real concave dip cutting inward through the
    interior at that gap (confirmed: "KITCHEN"/"LIVING AREA" label positions, genuinely
    inside the real building, tested as outside that dip by 60-75px). A concave dip in
    the silhouette used directly as a flood-fill barrier there would incorrectly wall off
    part of the room it's supposed to contain, producing exactly the jagged, wrong-
    looking traces this was built to avoid. The convex hull can't have that problem by
    construction, at the cost of being slightly more permissive at genuinely open/unwalled
    sides -- an acceptable trade since those sides have no real wall to trace against
    anyway.
    """
    inside = np.zeros((height, width), np.uint8)
    ys, xs = np.where(wall_mask > 0)
    if len(xs) >= 3:
        hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
        cv2.fillConvexPoly(inside, hull, 1)
    inside_area = float(inside.sum()) if inside.any() else float(height * width)

    best = None  # (score, comp)
    stale = 0
    for radius in _closing_radius_candidates(wall_mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k)
        barrier = closed.copy()
        if inside.any():
            barrier[inside == 0] = 1
        free = (barrier == 0).astype(np.uint8)
        _num, comp = cv2.connectedComponents(free, connectivity=4)

        seen_cids = set()
        score = 0
        for cx, cy, _name, _box in boxed_labels:
            row, col = int(round(cy)), int(round(cx))
            if not (0 <= row < height and 0 <= col < width):
                continue
            cid = int(comp[row, col])
            if cid == 0 or cid in seen_cids:
                continue
            area = float((comp == cid).sum())
            if area > 0.5 * inside_area:
                continue
            seen_cids.add(cid)
            score += 1
        if best is None or score > best[0]:
            best = (score, comp)
            stale = 0
        elif score < best[0]:
            stale += 1
        # else: tied with the current best (often 0 == 0 before any radius has bridged
        # anything yet) -- not a regression, keep escalating without counting it.
        # Once bridging genuinely-separate rooms' door gaps, further escalation only
        # over-closes (merging previously-distinct rooms back together) -- larger radii
        # get strictly more expensive (bigger morphology kernel) for no further benefit,
        # so stop once a real best has been found AND results regress for 2 attempts in
        # a row, or once every label is already resolved.
        if best[0] == len(boxed_labels) or (best[0] > 0 and stale >= 2):
            break

    if best is None or best[0] == 0:
        return {}
    _, comp = best
    used_cids = set()
    traced = {}
    for i, (cx, cy, _name, _box) in enumerate(boxed_labels):
        row, col = int(round(cy)), int(round(cx))
        if not (0 <= row < height and 0 <= col < width):
            continue
        cid = int(comp[row, col])
        if cid == 0 or cid in used_cids:
            continue
        area = float((comp == cid).sum())
        if area > 0.5 * inside_area:
            continue
        pts = component_polygon(comp, cid)
        if pts is None:
            continue
        used_cids.add(cid)
        traced[i] = pts
    return traced


def analyze_organic(bgr, ray_max_dist=None, progress=None):
    """Silhouette + rooms for plans with curved/diagonal/disconnected wall material.
    Each room first tries a real traced polygon (trace_rooms_organic(), any wall angle,
    not just H/V) from a door-gap-bridged flood-fill; only a label that doesn't resolve
    to its own distinct enclosed region (shares an open-plan space with another label, or
    the walls around it weren't fully captured) falls back to ray_box_for()'s nearest-
    wall rectangle approximation, same fallback contract trace_room_polygons() uses for
    the rectilinear pipeline. Room-boundary tracing for TRULY curved/pie-slice walls
    (e.g. rooms inside one circular pod) remains the harder, not-fully-solved case noted
    below -- several gap-bridging approaches were tried there before and failed to
    generalize (one promising-looking result turned out to be a test-methodology
    artifact: a tight crop boundary accidentally acting as a substitute wall, not a real
    fix) -- but a straight-walled plan at an arbitrary angle (e.g. a hexagonal building)
    is a meaningfully easier case and traces correctly with the same self-calibrated
    closing-radius approach used for silhouette gap-bridging.
    """
    ORGANIC_STAGES = 5
    if progress:
        progress("Membaca OCR", 1, ORGANIC_STAGES)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    engine = RapidOCR()
    words, _ = engine(bgr)
    boxed_labels = instances_with_boxes(words)
    if progress:
        progress("Mendeteksi dinding", 2, ORGANIC_STAGES)
    mask = organic_wall_mask(gray, boxed_labels=boxed_labels)
    if progress:
        progress("Mendeteksi siluet", 3, ORGANIC_STAGES)
    silhouette_pts = _trace_silhouette_robust(mask)

    vis = bgr.copy()
    if len(silhouette_pts) >= 3:
        cv2.polylines(vis, [np.array(silhouette_pts, dtype=np.int32)], True, (40, 40, 220), 3)

    # Radius/dimension callouts (R5000, R1200, ...) are common on radial/circular drawings
    # and pass instances_with_boxes()'s digit filter since they're not PURELY digits --
    # they're still not room names, so they're kept for the hole-punch above (their ink can
    # sit close to a wall too) but dropped from the room-label list handed back to Sam.
    _dim_pattern = re.compile(r"^R\d+(X\d+)?$")
    # Title-block/metadata text (plan name, drawing number, "42' DIA", total square
    # footage, "THREE BEDROOMS" spec summary, ...) sits OUTSIDE the building's own
    # silhouette on every circular reference plan tested (Gambar6/7) -- a general,
    # position-based filter catches all of these at once instead of pattern-matching
    # every possible title-block phrase, which wouldn't generalize to new drawings.
    # Still kept in boxed_labels for the wall-mask hole-punch above (any ink close to a
    # real wall, title-block or not, can interfere with wall detection there).
    # Uses the wall mask's own convex hull, not silhouette_pts directly -- same reasoning
    # as trace_rooms_organic()'s flood-fill boundary: a real gap in the outer ring can
    # give the traced silhouette a concave dip that cuts a genuinely-interior room's own
    # label position out. A convex hull can't have that problem by construction.
    ys_wm, xs_wm = np.where(mask > 0)
    hull_arr = cv2.convexHull(np.stack([xs_wm, ys_wm], axis=1).astype(np.int32)) if len(xs_wm) >= 3 else None
    def _inside_silhouette(cx, cy, margin=15.0):
        if hull_arr is None:
            return True
        return cv2.pointPolygonTest(hull_arr, (float(cx), float(cy)), True) >= -margin
    real_labels = [
        (cx, cy, name, box) for cx, cy, name, box in boxed_labels
        if not _dim_pattern.match(name) and _inside_silhouette(cx, cy)
    ]
    named = [(cx, cy, name) for cx, cy, name, _box in real_labels]

    if ray_max_dist is None:
        # Self-calibrated to this drawing's own scale instead of a fixed fraction of image
        # size: a generous first pass finds each label's raw nearest-wall distance in all 4
        # directions; the MEDIAN of the finite ones (excludes directions that are genuinely
        # open, e.g. an open-plan room's unwalled side, which would otherwise drag the
        # estimate out) approximates a typical room's real scale in this specific plan.
        # Final boxes are capped at a modest multiple of that, so a ray with no wall nearby
        # stops at a plausible size instead of ballooning across unrelated neighbors.
        probe_cap = int(0.35 * (h * h + w * w) ** 0.5)
        raw_dists = []
        for cx, cy, _name in named:
            l, t, r, b = ray_box_for(mask, cx, cy, probe_cap)
            for d in (cx - l, r - cx, cy - t, b - cy):
                if d < probe_cap:
                    raw_dists.append(d)
        typical = float(np.median(raw_dists)) if raw_dists else probe_cap
        ray_max_dist = int(max(60, min(probe_cap, 1.6 * typical)))

    if progress:
        progress("Menjiplak ruang", 4, ORGANIC_STAGES)
    # Real traced polygons (any wall angle) take priority; ray_box_for()'s rectangle
    # approximation is only the fallback for a label that doesn't resolve to its own
    # distinct enclosed region (see trace_rooms_organic()'s docstring for why that's not
    # always a detection failure -- a genuine open-plan space has nothing to trace).
    traced = trace_rooms_organic(mask, real_labels, h, w)
    labels_out = []
    for i, (cx, cy, name, _box) in enumerate(real_labels):
        pts = traced.get(i)
        if pts is not None:
            source = "poligon-organik"
        else:
            left, top, right, bot = ray_box_for(mask, cx, cy, ray_max_dist)
            if right - left < 8 or bot - top < 8:
                continue
            pts = [[left, top], [right, top], [right, bot], [left, bot]]
            source = "kotak-organik"
        labels_out.append({"name": name, "x": cx, "y": cy, "points": pts, "source": source})
        color = (0, 180, 0) if source == "poligon-organik" else (0, 140, 255)
        cv2.polylines(vis, [np.array(pts, dtype=np.int32)], True, color, 2)
        x0 = min(p[0] for p in pts)
        y0 = min(p[1] for p in pts)
        cv2.putText(vis, name, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 90, 200), 2, cv2.LINE_AA)

    if progress:
        progress("Selesai", 5, ORGANIC_STAGES)
    return vis, silhouette_pts, labels_out


# ---------- PDF input: read the drawing's own vector data directly, instead of
# rasterizing to an image and fighting compression/threshold noise. Only useful when the
# PDF is a real vector export (from CAD), not a scanned/rasterized PDF -- get_drawings()
# and get_text() simply come back near-empty for the latter, which the callers below
# already treat as "nothing found" rather than crashing.
def _require_pymupdf():
    if pymupdf is None:
        raise RuntimeError("pymupdf_missing")


def pdf_open(pdf_bytes):
    _require_pymupdf()
    return pymupdf.open(stream=pdf_bytes, filetype="pdf")


def pdf_page_count(pdf_bytes):
    doc = pdf_open(pdf_bytes)
    return len(doc)


def pdf_page_label(doc, index):
    """Best-effort short label for a page picker: the drawing's own title text if one
    reads as short/title-like, else a generic fallback -- never blocks on this.
    """
    try:
        text = doc[index].get_text()
    except Exception:
        text = ""
    candidates = [
        line.strip() for line in text.splitlines()
        if 3 <= len(line.strip()) <= 40 and any(c.isalpha() for c in line) and line.strip().upper() == line.strip()
    ]
    for line in candidates:
        if "DENAH" in line:
            return line
    if candidates:
        return candidates[0]
    return f"Halaman {index + 1}"


def pdf_page_thumbnail_png(pdf_bytes, index, zoom=0.6):
    doc = pdf_open(pdf_bytes)
    page = doc[index]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    return pix.tobytes("png")


def _straightness_residual(points):
    """How far a set of points deviates from their own best-fit line, normalized by the
    path's own span -- near 0 for a straight dimension/extension line, much larger for a
    real curve/arc. Exact vector coordinates, no raster noise at all.
    """
    pts = np.array([[p.x, p.y] for p in points], dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    mean = pts.mean(axis=0)
    span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    if span < 1e-6:
        return 0.0
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    normal = vt[1]
    resid = np.abs((pts - mean) @ normal)
    return float(resid.max() / span)


def _largest_spatial_cluster(rects, gap=40.0):
    """Groups object bounding-rects into spatially-proximate clusters (union-find over a
    "rects within `gap` of each other" graph) and returns the indices of the LARGEST
    cluster by rect count. Isolates the main floor-plan drawing from a spatially separate
    title block / legend, without assuming anything about a specific template's layout
    (position, orientation, size) -- unlike hardcoding a sidebar/strip to exclude.
    """
    n = len(rects)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def close(a, b):
        return not (a.x1 + gap < b.x0 or b.x1 + gap < a.x0 or a.y1 + gap < b.y0 or b.y1 + gap < a.y0)

    for i in range(n):
        for j in range(i + 1, n):
            if close(rects[i], rects[j]):
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return max(groups.values(), key=len)


def _pdf_vector_layers(page, zoom):
    """Wall mask (filled black polygons -- real walls, same convention CAD exports use
    regardless of straight/curved), curve-candidate mask (stroked, dark, multi-segment,
    not straight -- candidate floor-line/boundary material for open-sided rooms), and
    text tokens (room-name-like vs numeric/dimension-like), all restricted to the main
    floor-plan cluster (title block / legend excluded by spatial clustering, not by a
    hardcoded position). Coordinates are in the zoomed raster's pixel space throughout.
    """
    drawings = page.get_drawings()

    # Classify each object BEFORE clustering (not after): the raw drawing set is
    # thousands of tiny fragments (dimension ticks, letter strokes, icon segments)
    # dense enough that spatially clustering all of them transitively chains the
    # whole page -- including the title block -- into one blob. Restricting the
    # O(n^2) clustering step to only the substantial wall/curve candidates (a few
    # dozen objects) both separates the title block correctly and is far faster.
    candidates = []  # list of (kind, drawing) where kind is "wall" or "curve"
    for d in drawings:
        items = d["items"]
        fill = d.get("fill")
        color = d.get("color")
        is_dark = color is not None and sum(color) < 1.5
        if fill is not None and sum(fill) < 1.5:
            candidates.append(("wall", d))
        elif is_dark and fill is None and len(items) >= 10:
            pts = [it[1] for it in items if it[0] == "l"]
            if items and items[-1][0] == "l":
                pts.append(items[-1][2])
            if _straightness_residual(pts) < 0.02:
                continue
            candidates.append(("curve", d))

    rects = [d["rect"] for _, d in candidates]
    cluster_idx = set(_largest_spatial_cluster(rects, gap=40.0))
    plan = [candidates[i] for i in cluster_idx]
    if plan:
        xs0 = min(d["rect"].x0 for _, d in plan); xs1 = max(d["rect"].x1 for _, d in plan)
        ys0 = min(d["rect"].y0 for _, d in plan); ys1 = max(d["rect"].y1 for _, d in plan)
    else:
        r = page.rect
        xs0, ys0, xs1, ys1 = r.x0, r.y0, r.x1, r.y1

    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if pix.n == 4 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    wall_mask = np.zeros((h, w), np.uint8)
    curve_mask = np.zeros((h, w), np.uint8)
    for kind, d in plan:
        items = d["items"]
        mask = wall_mask if kind == "wall" else curve_mask
        thickness = 1 if kind == "wall" else 2
        for it in items:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                cv2.line(mask, (int(p1.x * zoom), int(p1.y * zoom)), (int(p2.x * zoom), int(p2.y * zoom)), 1, thickness)

    words = page.get_text("words")
    name_tokens, numeric_positions = [], []
    for x0, y0, x1, y1, text, *_ in words:
        if not (xs0 - 1 <= x0 <= xs1 + 1 and ys0 - 1 <= y0 <= ys1 + 1):
            continue
        raw = text.strip().upper()
        cx, cy = (x0 + x1) / 2 * zoom, (y0 + y1) / 2 * zoom
        bbox = (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)
        if any(ch.isdigit() for ch in raw):
            numeric_positions.append((cx, cy))
        elif raw:
            name_tokens.append((cx, cy, raw, bbox))
        for mask in (wall_mask, curve_mask):
            xa, ya = int(bbox[0]) - 6, int(bbox[1]) - 6
            xb, yb = int(bbox[2]) + 6, int(bbox[3]) + 6
            mask[max(0, ya):yb, max(0, xa):xb] = 0

    return bgr, wall_mask, curve_mask, name_tokens, numeric_positions


def _pdf_group_names(name_tokens, x_thresh=60, y_thresh=22):
    """Same word-grouping convention as instances_with_boxes() (stacked/adjacent OCR
    tokens into one room label), applied to the PDF's own exact text instead of OCR --
    so there are no misreads to begin with, only the grouping step is shared.
    """
    tokens = sorted(name_tokens, key=lambda t: (t[1], t[0]))
    groups = []
    for cx, cy, text, box in tokens:
        placed = False
        for g in groups:
            if abs(cy - g["cy"]) < y_thresh and abs(cx - g["cx"]) < x_thresh:
                g["parts"].append((cy, cx, text))
                g["cx"] = float(np.mean([p[1] for p in g["parts"]]))
                g["cy"] = float(np.mean([p[0] for p in g["parts"]]))
                g["x0"] = min(g["x0"], box[0]); g["y0"] = min(g["y0"], box[1])
                g["x1"] = max(g["x1"], box[2]); g["y1"] = max(g["y1"], box[3])
                placed = True
                break
        if not placed:
            groups.append({"parts": [(cy, cx, text)], "cx": cx, "cy": cy, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]})
    out = []
    for g in groups:
        name = " ".join(t for _, _, t in sorted(g["parts"]))
        out.append((g["cx"], g["cy"], name, (g["x0"], g["y0"], g["x1"], g["y1"])))
    return out


def pdf_page_silhouette(pdf_bytes, page_index, zoom=2.0, progress=None):
    """Silhouette for a single PDF page via its own vector data -- see _pdf_vector_layers()
    and _merged_silhouette(). Returns (bgr_render, silhouette_pts).
    """
    if progress:
        progress("Membaca data vektor PDF", 1, 3)
    doc = pdf_open(pdf_bytes)
    page = doc[page_index]
    bgr, wall_mask, curve_mask, _names, _nums = _pdf_vector_layers(page, zoom)
    combined = np.maximum(wall_mask, curve_mask)
    if progress:
        progress("Mendeteksi siluet", 2, 3)
    pts = _merged_silhouette(combined, epsilon=2.0 * zoom, min_component_area=300 * zoom)
    if progress:
        progress("Selesai", 3, 3)
    return bgr, pts


def pdf_page_rooms(pdf_bytes, page_index, zoom=2.0, progress=None):
    """Silhouette + room boxes for a single PDF page. Rooms use the same nearest-wall
    ray-cast box approximation as analyze_organic() (see ray_box_for()) rather than a
    traced polygon -- the wall/floor-line data is clean here (no JPEG artifacts), but
    reliably bridging every door gap for a full flood-fill trace is still the open
    problem noted in analyze_organic()'s docstring; a box beats no room for the
    relative-area metrics components 2/3 depend on. Returns (bgr_render, silhouette_pts,
    rooms), rooms = [{"name","x","y","points"}, ...].
    """
    if progress:
        progress("Membaca data vektor PDF", 1, 4)
    doc = pdf_open(pdf_bytes)
    page = doc[page_index]
    bgr, wall_mask, curve_mask, name_tokens, numeric_positions = _pdf_vector_layers(page, zoom)
    combined = np.maximum(wall_mask, curve_mask)
    if progress:
        progress("Mendeteksi siluet", 2, 4)
    silhouette_pts = _merged_silhouette(combined, epsilon=2.0 * zoom, min_component_area=300 * zoom)
    if progress:
        progress("Mendeteksi ruang", 3, 4)

    named = _pdf_group_names(name_tokens, x_thresh=60 * zoom, y_thresh=22 * zoom)
    _dim_pattern = re.compile(r"^R\d+(X\d+)?$")
    named = [t for t in named if not _dim_pattern.match(t[2]) and t[2]]

    h, w = combined.shape
    probe_cap = int(0.35 * (h * h + w * w) ** 0.5)
    raw_dists = []
    for cx, cy, _name, _box in named:
        l, t, r, b = ray_box_for(combined, cx, cy, probe_cap)
        for d in (cx - l, r - cx, cy - t, b - cy):
            if d < probe_cap:
                raw_dists.append(d)
    typical = float(np.median(raw_dists)) if raw_dists else probe_cap
    ray_max_dist = int(max(60, min(probe_cap, 1.6 * typical)))

    rooms = []
    for cx, cy, name, _box in named:
        left, top, right, bot = ray_box_for(combined, cx, cy, ray_max_dist)
        if right - left < 8 or bot - top < 8:
            continue
        rooms.append({"name": name, "x": cx, "y": cy, "points": [[left, top], [right, top], [right, bot], [left, bot]]})
    if progress:
        progress("Selesai", 4, 4)
    return bgr, silhouette_pts, rooms


def analyze(bgr, progress=None):
    ANALYZE_STAGES = 6
    if progress:
        progress("Membaca OCR", 1, ANALYZE_STAGES)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    engine = RapidOCR()
    words, _ = engine(bgr)
    labels = instances(words)
    print("labels:")
    for cx, cy, name in labels:
        print(f"  {name} @ {cx:.0f},{cy:.0f}")

    horiz, vert = ink_lines(gray, labels)
    fold = folding_line(gray)
    print("fold", fold)
    if fold:
        _span, fy, fx0, fx1 = fold
        horiz.append((fy, fx0 - 30, fx1 + 40))
    print("H", len(horiz), "V", len(vert))

    def away(lines, axis):
        kept = []
        for coord, a, b in lines:
            close_to_text = False
            for lx, ly, _n in labels:
                if axis == "h" and abs(ly - coord) < 26 and a - 8 <= lx <= b + 8 and (b - a) < 110:
                    close_to_text = True
                if axis == "v" and abs(lx - coord) < 26 and a - 8 <= ly <= b + 8 and (b - a) < 110:
                    close_to_text = True
            if not close_to_text:
                kept.append((coord, a, b))
        return kept

    if progress:
        progress("Mendeteksi dinding", 2, ANALYZE_STAGES)
    horiz = away(horiz, "h")
    vert = away(vert, "v")
    horiz = _drop_comb_patterns(horiz)
    vert = _drop_comb_patterns(vert)
    slack = 36

    if progress:
        progress("Mendeteksi siluet", 3, ANALYZE_STAGES)
    silhouette_pts = outer_silhouette(bgr)
    sil_bbox = _bbox(silhouette_pts) if silhouette_pts and len(silhouette_pts) >= 3 else [0, 0, w, h]
    if progress:
        progress("Menjiplak ruang", 4, ANALYZE_STAGES)
    traced, comp, num_comp, used_components = trace_room_polygons(gray, horiz, vert, silhouette_pts, labels)

    rooms = []
    comp_to_room_id = {}
    next_id = [1]

    def alloc_id():
        rid = f"r{next_id[0]}"
        next_id[0] += 1
        return rid

    for i, (cx, cy, name) in enumerate(labels):
        if i in traced:
            pts, cid = traced[i]
            rid = alloc_id()
            rooms.append({"id": rid, "name": name, "source": "tulisan", "points": pts})
            comp_to_room_id[cid] = rid
            print(f"ROOM {name} {rid} poligon({len(pts)} titik)")
            continue
        # Flood-fill couldn't resolve this label safely (merged blob, or landed on a
        # barrier pixel) -- fall back to the nearest-wall rectangle instead of dropping it.
        left, top, right, bot = box_for(horiz, vert, cx, cy, slack)
        if left is None: left = sil_bbox[0]
        if top is None: top = sil_bbox[1]
        if right is None: right = sil_bbox[2]
        if bot is None: bot = sil_bbox[3]
        # The rectangle above can bleed into a neighbouring room/circulation area the
        # label's own flood-fill never touched (this is the source of real overlaps
        # seen in testing) -- clip it down to pixels that are actually part of the
        # label's own connected component, even though that component was too large
        # to trust as a clean traced polygon on its own.
        row, col = int(round(cy)), int(round(cx))
        label_cid = int(comp[row, col]) if 0 <= row < h and 0 <= col < w else 0
        if label_cid > 0:
            region = comp[max(0, top):bot, max(0, left):right] == label_cid
            ys, xs = np.where(region)
            if len(xs) >= 4:
                left, right = max(0, left) + int(xs.min()), max(0, left) + int(xs.max()) + 1
                top, bot = max(0, top) + int(ys.min()), max(0, top) + int(ys.max()) + 1
        if right - left < 8 or bot - top < 8:
            print("OPEN", name, left, top, right, bot)
            continue
        rid = alloc_id()
        rooms.append({"id": rid, "name": name, "source": "tulisan",
                       "points": [[int(left), int(top)], [int(right), int(top)], [int(right), int(bot)], [int(left), int(bot)]],
                       "_box": [int(left), int(top), int(right), int(bot)]})
        print(f"ROOM {name} {rid} (kotak fallback) [{left},{top},{right},{bot}]")

    fallback_rooms = [r for r in rooms if "_box" in r]
    if fallback_rooms:
        other_boxes = [_bbox(r["points"]) for r in rooms if "_box" not in r]
        _separate_fallback_boxes(fallback_rooms, other_boxes)
        for r in fallback_rooms:
            x0, y0, x1, y1 = r["_box"]
            r["points"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            del r["_box"]

    # Circulation: any leftover flood-fill component no label claimed, above the noise floor.
    for cid in range(1, num_comp):
        if cid in used_components:
            continue
        area = float((comp == cid).sum())
        if area < 600:
            continue
        pts = component_polygon(comp, cid)
        if pts is None:
            continue
        rid = alloc_id()
        rooms.append({"id": rid, "name": "sirkulasi", "source": "terkaan", "points": pts})
        comp_to_room_id[cid] = rid
        print("SIRKULASI", rid, "luas", int(area))

    if progress:
        progress("Mendeteksi hubungan pintu", 5, ANALYZE_STAGES)
    adjacency = detect_adjacency(gray, comp, horiz, vert, comp_to_room_id)

    vis = bgr.copy()
    for r in rooms:
        pts = np.array(r["points"], dtype=np.int32)
        color = (80, 170, 70)
        if r["name"] == "sirkulasi":
            color = (0, 210, 255)
        elif "MEETING" in r["name"]:
            color = (200, 120, 40)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], color)
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        cv2.polylines(vis, [pts], True, color, 2)
        x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        cv2.putText(vis, r["name"], (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)

    centroid_by_id = {r["id"]: np.array(r["points"], dtype=np.float64).mean(axis=0) for r in rooms}
    for edge in adjacency:
        pa, pb = centroid_by_id.get(edge["a"]), centroid_by_id.get(edge["b"])
        if pa is None or pb is None:
            continue
        cv2.line(vis, tuple(pa.astype(int)), tuple(pb.astype(int)), (40, 40, 220), 2, cv2.LINE_AA)
        dx, dy = edge["door_at"]
        cv2.circle(vis, (int(dx), int(dy)), 5, (40, 40, 220), -1)
    for cid, pt in centroid_by_id.items():
        cv2.circle(vis, tuple(pt.astype(int)), 4, (20, 20, 20), -1)

    if progress:
        progress("Selesai", 6, ANALYZE_STAGES)
    return vis, rooms, adjacency


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    __import__("pathlib").Path(OUT).mkdir(exist_ok=True)
    out_png = sys.argv[2] if len(sys.argv) > 2 else str(__import__("pathlib").Path(OUT) / "gambar3_solved.png")
    bgr = cv2.imread(src)
    if bgr is None:
        raise SystemExit("tidak bisa membaca " + src)
    vis, rooms, adjacency = analyze(bgr)
    cv2.imwrite(out_png, vis)
    meetings = [_bbox(r["points"]) for r in rooms if r["name"] == "MEETING ROOM" and _bbox(r["points"])[0] < 400]
    if len(meetings) >= 2:
        x0 = max(0, min(b[0] for b in meetings) - 30)
        y0 = max(0, min(b[1] for b in meetings) - 20)
        x1 = min(vis.shape[1], max(b[2] for b in meetings) + 80)
        y1 = min(vis.shape[0], max(b[3] for b in meetings) + 30)
        __import__("pathlib").Path(OUT).mkdir(exist_ok=True)
        cv2.imwrite(str(__import__("pathlib").Path(OUT) / "gambar3_solved_meeting.png"), vis[y0:y1, x0:x1])
    json_path = out_png.rsplit(".", 1)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rooms": rooms, "adjacency": adjacency}, f, indent=2)
    print("wrote", out_png, len(rooms), "rooms", len(adjacency), "sambungan pintu")


if __name__ == "__main__":
    main()
