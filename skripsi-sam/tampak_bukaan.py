"""Komponen 4: rasio bukaan fasad -- prototipe deteksi dari gambar tampak (elevasi).

Status: bekerja baik untuk elevasi CAD garis bersih/datar (lihat contoh-tampak/tampak0,
tampak1, tampak2 -- bukaan persegi, tanpa balkon/reveal berperspektif). Mulai gagal
sebagian pada gambar dengan balkon/reveal miring berperspektif (contoh-tampak/tampak3
dan seterusnya) -- lihat "KETERBATASAN" di bawah.

---- METODE ----

1. Envelope (batas gambar bangunan): dicari lewat kolom mana saja yang punya tinta
   (garis) sampai MENDEKATI baris teratas seluruh gambar. Ini secara khusus dirancang
   supaya kebal terhadap pohon di depan fasad (yang sama gelapnya dengan garis gambar,
   dan gampang tersambung ke bangunan lewat dilasi) -- pohon yang ditanam di permukaan
   tanah tidak pernah mencapai setinggi atap bangunan, jadi kolom yang "tinggi" ~pasti
   bagian bangunan, terlepas dari serapat apapun tekstur pohonnya.

2. Sel tertutup: threshold tinta (ambang lunak, ikut menangkap garis mullion/kusen yang
   lebih tipis) lalu dibalik -- setiap wilayah putih yang terkurung garis jadi satu
   connected component ("sel"). ini menangkap SEMUA wilayah yang dikelilingi garis:
   panel kaca jendela, panel pintu, TAPI JUGA potongan dinding polos yang kebetulan
   dikelilingi garis pemisah lantai/kusen di sekitarnya.

3. Klasifikasi sel jadi bukaan vs dinding, berdasarkan BENTUK (bukan warna -- pada
   gambar CAD garis, kaca jendela sama putihnya dengan dinding polos; satu-satunya
   pembeda adalah framing/garis di sekitarnya):
   - dibuang kalau sisi terpendeknya < MIN_DIM piksel (celah tipis dari garis pemisah
     lantai bergaris-dobel, bukan bukaan sungguhan)
   - dibuang kalau "extent" (luas sel / luas bounding box)-nya < MIN_EXTENT (bentuk
     tidak beraturan -- biasanya pohon/tekstur yang menerobos sebuah sel)
   - dibuang kalau porsi keliling yang SEJAJAR SUMBU (horizontal/vertikal) kurang dari
     MIN_AXIS_FRACTION -- bukaan asli pada elevasi datar selalu berupa persegi panjang
     sejajar sumbu; reveal/balkon berperspektif punya sisi miring yang jadi porsi
     signifikan kelilingnya.
   - dibuang kalau sel itu punya LUBANG SUBSTANSIAL di dalamnya -- ink yang terkurung
     di dalam sel ini tapi tidak menyambung ke tepi luarnya, dan lubang itu sendiri
     cukup besar (min(w,h) >= MIN_DIM) untuk bisa jadi bukaan asli. Ini menangkap
     "dinding polos yang kebetulan membungkus satu atau lebih bukaan asli di
     dalamnya" (bentuk topologinya = bingkai-dengan-potongan, bukan satu bukaan
     utuh) -- lihat riwayat bug di §5.2/test_tampak_bukaan.py. Lubang yang KECIL
     (mis. batang-batang tipis gerbang berkisi, atau noise anti-aliasing) sengaja
     TIDAK menggugurkan selnya -- gerbang/pintu berkisi tetap harus terhitung SATU
     bukaan, bukan ditolak jadi dinding hanya karena berlubang-lubang kecil.
   - sisanya dianggap BUKAAN (jendela, pintu, pintu garasi -- semua dihitung sebagai
     bukaan, sesuai konvensi "window-to-wall ratio" standar, dikonfirmasi ke Sam).

---- VALIDASI ----

test_tampak_bukaan.py menguji ratio ini terhadap gambar sintetis dengan geometri
PERSIS diketahui (per standar metodologi_skripsi.md #5.2: "cukup uji kebenaran
hitung, bandingkan ke perhitungan manual pada contoh sederhana") -- LULUS, selisih
1.45% dari nominal, dijelaskan penuh oleh piksel garis kusen yang sengaja tak
dihitung ke kategori manapun (lihat docstring di file itu). Uji itu jugalah yang
awalnya MENEMUKAN bug "dinding-membungkus-bukaan" di atas, sebelum diperbaiki --
bukan divalidasi setelah diasumsikan benar.

---- KETERBATASAN YANG DIKETAHUI ----

- Balkon/reveal berperspektif miring (tampak3+): kadang MENYATU jadi SATU sel dengan
  jendela di sebelahnya (garis pemisahnya tidak selalu tertutup penuh di sudut
  pertemuan), sehingga sel gabungan itu jadi "sebagian besar sejajar sumbu" (karena
  porsi jendelanya) dan lolos filter MIN_AXIS_FRACTION walau seharusnya ditolak.
  Sudah dicoba: menolak berdasarkan PANJANG ABSOLUT sisi miring (bukan cuma porsinya)
  -- GAGAL, karena jendela asli yang sudut-nya kena garis arsir miring (garis bayangan
  di bawah atap teras) menghasilkan sisi miring yang panjangnya serupa, jadi ikut
  tertolak juga (regresi ke tampak0). Root cause sebenarnya butuh sinyal yang beda:
  mengenali bukaan asli dari FRAME GARIS-DOBEL di sekelilingnya (bukan cuma bentuk
  selnya), belum diimplementasikan.
- Pita dinding polos TANPA subdivisi garis apa pun sama sekali (bukan "membungkus
  bukaan" -- benar-benar kosong, tidak ada garis internal sama sekali) masih bisa
  salah kena hitung sebagai bukaan kalau bentuknya kebetulan persegi panjang bersih
  (mis. pita fondasi di tampak1). Ini KASUS BERBEDA dari bug bingkai-dengan-potongan
  di atas (yang sudah diperbaiki) -- di sini tidak ada lubang untuk dideteksi sama
  sekali, jadi perbaikan hole_area_fraction tidak menyentuh kasus ini. Masih belum
  ada perbaikan.
- Belum diuji pada gambar ilustrasi/render fotorealistik (tampak9, tampak10) --
  gambar-gambar itu tidak lagi berupa garis CAD bersih sama sekali, kemungkinan besar
  butuh pendekatan yang sama sekali berbeda (bukan threshold garis tinta).

---- CARA PAKAI ----

    python tampak_bukaan.py <path_gambar> [prefix_output]

Mencetak bbox envelope, jumlah piksel bukaan/dinding, dan void-to-solid ratio (metrik
yang dikutip metodologi_skripsi.md #4.4: area bukaan / area dinding solid -- BUKAN
dibagi total, jadi bisa > 1.0 kalau bukaan lebih dominan dari dinding) plus dua rasio
tambahan sebagai konteks diagnostik saja. Menyimpan <prefix>_overlay.png (oranye=bukaan,
biru muda=dinding, kotak hijau=envelope) untuk diperiksa manual -- SELALU periksa
overlay-nya, jangan percaya angkanya begitu saja (pola yang sudah berulang kali
terbukti perlu di proyek ini).

CAKUPAN (per metodologi_skripsi.md #4.4): rasio LUAS sudah divalidasi lewat uji
kebenaran hitung (lihat VALIDASI di atas), dan sekarang JUGA dilaporkan berdampingan
dengan JUMLAH bukaan dan SEBARAN UKURANNYA (result["opening_count"], result["openings"],
result["opening_sizes_px"]/min/max/median) -- memenuhi syarat pelaporan #4.4 secara
penuh, bukan cuma satu angka rasio. Sel-sel bukaan (tiap panel kaca/pintu yang
terpisah mullion) dikelompokkan jadi satu bukaan logis lewat cluster_openings() --
lihat CLUSTER_GAP_PX di bawah untuk detail dan keterbatasannya.
"""

import sys
import math

import cv2
import numpy as np

MIN_DIM = 15
MIN_EXTENT = 0.5
MIN_AREA = 30
AXIS_TOL_DEG = 14
MIN_AXIS_FRACTION = 0.75


def find_envelope(gray):
    """Building bbox, robust to foreground trees -- see METODE step 1 above."""
    h, w = gray.shape
    _, outline = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    has_ink_col = outline.any(axis=0)
    if not has_ink_col.any():
        # Blank/very faint image: no line dark enough to be a drawing at all.
        raise ValueError("no_building")
    topmost = np.where(has_ink_col, outline.argmax(axis=0), h)
    roofline = int(topmost.min())
    is_building_col = topmost < (roofline + 0.35 * h)
    col_run = cv2.morphologyEx(
        is_building_col.astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8)
    ).flatten()
    _, _, run_stats, _ = cv2.connectedComponentsWithStats(col_run.reshape(1, -1), connectivity=8)
    best_run = 1 + int(np.argmax(run_stats[1:, cv2.CC_STAT_AREA]))
    ex = int(run_stats[best_run, cv2.CC_STAT_LEFT])
    ew = int(run_stats[best_run, cv2.CC_STAT_WIDTH])
    col_slice = outline[:, ex:ex + ew]
    ink_rows = np.where(col_slice.any(axis=1))[0]
    ey, eh = int(ink_rows.min()), int(ink_rows.max() - ink_rows.min() + 1)
    return ex, ey, ew, eh


MIN_HOLE_FRACTION = 0.05  # a hole must eat at least 5% of the cell's own area to
                           # count -- separates a genuine "wall wrapping around one or
                           # more real openings" cell from a legitimate window pane
                           # cell that merely has a microscopic anti-aliasing fleck
                           # (a hole of ~2px total, confirmed on a real tampak3 window
                           # pane) inside it.


def hole_area_fraction(cell_mask_u8):
    """Fraction of this cell's own outer area taken up by SUBSTANTIAL holes -- ink
    enclosed inside it that isn't connected to its outer edge, restricted to holes
    that are themselves chunky enough to plausibly be a real separate opening
    (min(w,h) >= MIN_DIM). A "frame with cutouts" shape (a plain wall body wrapping
    around one or more separate window/door openings inside it) has a LARGE such
    fraction; a legitimate window pane with a stray anti-aliasing fleck has a
    negligible one either way. The size floor matters separately: a slatted gate's
    frame cell is riddled with holes too (confirmed on tampak0's ground-floor gate --
    96 of them), but every one is a single ~3px-wide bar, not a real opening; without
    the size floor those would sum to a large fraction and wrongly disqualify the
    gate as a whole (it should count as ONE opening, a door, same as any other).
    RETR_EXTERNAL alone can't see holes at all (it only traces the outer silhouette),
    which is exactly how a synthetic ground-truth test (plain rectangular wall, two
    windows, one door, nothing else -- metodologi_skripsi.md #5.2's own validation
    standard for this metric) caught the whole wall body being misread as one giant
    "opening": high extent, axis-aligned outer edge, holes invisible to shape checks
    that only looked at the outside."""
    contours, hierarchy = cv2.findContours(cell_mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return 0.0
    # RETR_CCOMP hierarchy: [next, prev, first_child, parent]. A contour with no
    # parent is the outer boundary; contours with a parent are holes cut into it.
    outer_area = max((cv2.contourArea(c) for c, h in zip(contours, hierarchy[0]) if h[3] == -1), default=0.0)
    hole_area = sum(
        cv2.contourArea(c) for c, h in zip(contours, hierarchy[0])
        if h[3] != -1 and min(cv2.boundingRect(c)[2], cv2.boundingRect(c)[3]) >= MIN_DIM
    )
    return (hole_area / outer_area) if outer_area > 0 else 0.0


def axis_aligned_fraction(cell_mask_u8):
    """Fraction of a cell's simplified boundary (by length) that runs horizontal/
    vertical -- real elevation openings are axis-aligned rectangles; perspective
    surfaces (reveals, roof-overhang hatch triangles) are not."""
    contours, _ = cv2.findContours(cell_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    if peri < 1e-6:
        return 0.0
    approx = cv2.approxPolyDP(c, 0.02 * peri, True).reshape(-1, 2)
    if len(approx) < 4:
        return 0.0
    aligned_len, total_len = 0.0, 0.0
    n_pts = len(approx)
    for i in range(n_pts):
        p0, p1 = approx[i], approx[(i + 1) % n_pts]
        dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 90
        total_len += seg_len
        if min(ang, 90 - ang) <= AXIS_TOL_DEG:
            aligned_len += seg_len
    return (aligned_len / total_len) if total_len > 0 else 0.0


CLUSTER_GAP_PX = 7  # dilation kernel size for grouping opening cells into logical
                     # openings -- bridges gaps up to 6px. Went through two rounds of
                     # tightening: 11 (bridges <=10px) merged ALL FIVE floors of
                     # tampak0's windows into one giant "opening", because its floor-
                     # to-floor separator gap (~9px) turned out almost as narrow as
                     # its own internal mullion gap (~3px); 9 (bridges <=8px) fixed
                     # most of that but still merged one adjacent floor pair. 7 was
                     # verified clean on both confirmed cases: bridges tampak1's
                     # ~6px mullion gaps (its nearest real inter-opening gap is
                     # ~35px, comfortable margin), and no longer bridges any of
                     # tampak0's ~9px floor separators.
                     #
                     # Known remaining imprecision (tampak0, accepted): a narrow
                     # single-pane accent window sandwiched between two wider windows
                     # sometimes sits close enough to ONE of them that it merges with
                     # that neighbor instead of staying its own opening -- the other
                     # neighbor (slightly farther) stays correctly separate. A purely
                     # gap-distance heuristic can't fully resolve this; the real fix
                     # is the same one noted for the perspective-reveal problem
                     # elsewhere in this file: recognize a genuine opening by its own
                     # double-line FRAME, not by gap distance to its neighbors.


def cluster_openings(opening_mask):
    """Group individual opening CELLS (window panes split apart by mullions, door
    leaves split by a center stile) into logical OPENINGS (one whole window, one
    whole door) -- metodologi_skripsi.md #4.4 asks for opening COUNT and SIZE
    DISTRIBUTION alongside the ratio, and neither is meaningful counted per-pane (a
    3-pane window must count as 1 opening, not 3). Dilate the accepted-opening mask
    by CLUSTER_GAP_PX to bridge mullion-sized gaps, cluster on THAT, then measure each
    cluster's area back on the UNDILATED mask so dilation never inflates the areas
    being reported."""
    if not opening_mask.any():
        return []
    dilated = cv2.dilate(opening_mask.astype(np.uint8), np.ones((CLUSTER_GAP_PX, CLUSTER_GAP_PX), np.uint8))
    n_clusters, cluster_labels = cv2.connectedComponents(dilated, connectivity=8)
    openings = []
    for cid in range(1, n_clusters):
        cluster_true_mask = opening_mask & (cluster_labels == cid)
        area = int(cluster_true_mask.sum())
        if area == 0:
            continue
        ys, xs = np.where(cluster_true_mask)
        openings.append({
            "area_px": area,
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
        })
    openings.sort(key=lambda o: -o["area_px"])
    return openings


def analyze(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return analyze_image(bgr)


def analyze_image(bgr):
    """Same as analyze(), on an already-decoded BGR image (used by demo_server.py,
    which receives the upload as bytes and never writes it to disk)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ex, ey, ew, eh = find_envelope(gray)

    _, ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    space = cv2.bitwise_not(ink)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(space, connectivity=4)
    border = (set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
              | set(labels[:, 0].tolist()) | set(labels[:, -1].tolist()))
    ink_ys, ink_xs = np.where(ink > 0)
    ink_label = labels[ink_ys[0], ink_xs[0]] if len(ink_ys) else -1

    overlay = bgr.copy()
    opening_px = wall_px = 0
    opening_mask = np.zeros(gray.shape, dtype=bool)
    for i in range(n):
        if i == ink_label or i in border:
            continue
        x, y, ww, hh, area = stats[i]
        if area < MIN_AREA:
            continue
        if not (x >= ex - 2 and y >= ey - 2 and x + ww <= ex + ew + 2 and y + hh <= ey + eh + 2):
            continue
        extent = area / (ww * hh)
        mask = labels == i
        is_opening = False
        if min(ww, hh) >= MIN_DIM and extent >= MIN_EXTENT:
            cell_u8 = mask[y:y + hh, x:x + ww].astype(np.uint8) * 255
            if hole_area_fraction(cell_u8) < MIN_HOLE_FRACTION:
                is_opening = axis_aligned_fraction(cell_u8) >= MIN_AXIS_FRACTION
        if is_opening:
            opening_px += int(area)
            opening_mask |= mask
            overlay[mask] = (0, 140, 255)
        else:
            wall_px += int(area)
            overlay[mask] = (255, 220, 200)

    # Opening COUNT and SIZE DISTRIBUTION (metodologi_skripsi.md #4.4's explicit
    # mitigation requirement -- "laporkan rasio luas berdampingan dengan jumlah dan
    # sebaran ukuran bukaan", precisely because one big opening, mis. a garage door,
    # can dominate the area ratio and mask the real composition). Reported alongside
    # the ratio, per #3.1's cross-component principle -- never folded into it.
    openings = cluster_openings(opening_mask)
    for idx, o in enumerate(openings, start=1):
        ox, oy, ow, oh = o["bbox"]
        cv2.putText(overlay, str(idx), (ox + 2, oy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, str(idx), (ox + 2, oy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(overlay, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    sizes = [o["area_px"] for o in openings]
    return {
        "envelope": (ex, ey, ew, eh),
        "opening_px": opening_px,
        "wall_px": wall_px,
        # The actual metric metodologi_skripsi.md §4.4 cites (Carmona et al.: "area of
        # openings divided by the area of the solid wall") -- NOT bounded to [0,1], can
        # exceed 1.0 when openings outweigh solid wall (e.g. tampak0's full-glazing
        # style). This is the number that should be reported as "void-to-solid ratio".
        "void_to_solid_ratio": opening_px / wall_px if wall_px else float("inf"),
        # Auxiliary, not the cited metric -- kept only as extra diagnostic context.
        "ratio_opening_of_classified": opening_px / (opening_px + wall_px) if (opening_px + wall_px) else 0.0,
        "ratio_opening_of_envelope": opening_px / (ew * eh) if ew * eh else 0.0,
        # Count + size distribution, required alongside the ratio by #4.4.
        "opening_count": len(openings),
        "openings": openings,  # each: {"area_px", "bbox": (x,y,w,h)}, largest first
        "opening_sizes_px": sizes,
        "opening_size_min": min(sizes) if sizes else None,
        "opening_size_max": max(sizes) if sizes else None,
        "opening_size_median": float(np.median(sizes)) if sizes else None,
        "overlay": overlay,
    }


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else r"contoh-tampak\tampak0.jpg"
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "tampak_bukaan_out"
    result = analyze(img_path)
    print("envelope bbox:", result["envelope"])
    print("opening_px:", result["opening_px"], "wall_px:", result["wall_px"])
    print(f"void-to-solid ratio (opening/wall, per metodologi_skripsi.md #4.4) = {result['void_to_solid_ratio']:.4f}")
    print(f"  [aux, not the cited metric] opening/(opening+wall) = {result['ratio_opening_of_classified']:.4f}")
    print(f"  [aux, not the cited metric] opening/envelope_bbox_area = {result['ratio_opening_of_envelope']:.4f}")
    print(f"jumlah bukaan (per #4.4): {result['opening_count']}")
    if result["openings"]:
        print(f"  sebaran ukuran (px^2): min={result['opening_size_min']} "
              f"median={result['opening_size_median']:.0f} max={result['opening_size_max']}")
        for idx, o in enumerate(result["openings"], start=1):
            print(f"  #{idx}: area={o['area_px']} bbox={o['bbox']}")
    cv2.imwrite(f"{out_prefix}_overlay.png", result["overlay"])
    print(f"saved {out_prefix}_overlay.png -- selalu periksa manual, jangan percaya angka saja")
