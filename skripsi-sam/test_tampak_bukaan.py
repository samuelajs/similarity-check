"""Uji kebenaran hitung untuk void-to-solid ratio (metodologi_skripsi.md #4.4/#5.2).

metodologi_skripsi.md #5.2 menetapkan standar validasi untuk rasio void-to-solid
secara eksplisit: "monoton by construction -- tidak perlu uji empiris, cukup uji
kebenaran hitung (bandingkan ke perhitungan manual pada contoh sederhana)".

Skrip ini membangun satu gambar tampak SINTETIS dengan geometri yang PERSIS diketahui
(satu dinding persegi panjang, dua jendela, satu pintu, semua di posisi/ukuran nominal
yang ditentukan sendiri) -- lalu membandingkan output tampak_bukaan.analyze() terhadap
rasio yang dihitung tangan dari dimensi nominal itu.

Catatan penting yang DITEMUKAN lewat uji ini (bukan diasumsikan benar dari awal):
selisih kecil (~1.5%) antara rasio nominal dan rasio hasil tool bukan noise/bug --
sepenuhnya dijelaskan oleh garis outline (cv2.rectangle, tebal 2px) yang piksel-nya
sengaja TIDAK dihitung ke kategori bukaan maupun dinding (sama seperti bagaimana tool
ini memperlakukan garis kusen pada gambar asli manapun). Ambang toleransi di bawah
disetel longgar (5%) untuk mengakomodasi ini secara eksplisit, bukan untuk
menyembunyikan galat sungguhan.

Uji ini JUGA yang menemukan bug nyata (sebelum ambang toleransi ditambahkan): tanpa
pengecekan "hole" pada tampak_bukaan.hole_area_fraction, seluruh badan dinding (yang
topologinya = "bingkai dengan potongan" -- membungkus dua jendela dan satu pintu)
salah terhitung sebagai SATU bukaan raksasa, karena pengecekan bentuk lama hanya
melihat siluet LUAR tiap sel, tidak pernah memeriksa apakah sel itu membungkus
tinta yang terisolasi di dalamnya. Uji regresi kedua di bawah (gerbang berkisi-kisi,
diambil dari tampak0 asli) memverifikasi perbaikan itu tidak jadi terlalu agresif
sebaliknya (jangan sampai gerbang/pintu berkisi malah ikut ditolak).

Uji ketiga (jumlah bukaan) memakai gambar sintetis yang sama untuk memvalidasi
cluster_openings() -- ketiga bukaannya sengaja berjarak jauh satu sama lain, jadi
jawaban yang benar diketahui PERSIS (3), bukan sekadar "kelihatannya masuk akal".

Jalankan: python test_tampak_bukaan.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tampak_bukaan import analyze  # noqa: E402

TOLERANCE = 0.05  # relatif -- lihat catatan di atas soal kenapa bukan 0


def make_synthetic(path):
    """Dinding 340x580 (nominal), dua jendela, satu pintu -- posisi/ukuran EKSAK
    ditentukan di sini, dipakai juga untuk menghitung rasio yang diharapkan."""
    W, H = 500, 700
    img = np.full((H, W, 3), 255, np.uint8)
    building = (80, 60, 420, 640)      # x0,y0,x1,y1
    window1 = (130, 130, 230, 330)     # 100x200
    window2 = (280, 400, 380, 500)     # 100x100
    door1 = (150, 450, 250, 630)       # 100x180

    for rect in (building, window1, window2, door1):
        cv2.rectangle(img, rect[:2], rect[2:], (30, 30, 30), 2)
    cv2.imwrite(path, img)

    bx0, by0, bx1, by1 = building
    building_area = (bx1 - bx0) * (by1 - by0)
    opening_area = sum((r[2] - r[0]) * (r[3] - r[1]) for r in (window1, window2, door1))
    wall_area = building_area - opening_area
    return opening_area / wall_area


def test_synthetic_ground_truth(tmp_path):
    expected_ratio = make_synthetic(tmp_path)
    result = analyze(tmp_path)
    actual_ratio = result["void_to_solid_ratio"]
    rel_err = abs(actual_ratio - expected_ratio) / expected_ratio
    status = "LULUS" if rel_err <= TOLERANCE else "GAGAL"
    print(f"[{status}] uji kebenaran hitung (dinding+2 jendela+1 pintu, geometri diketahui)")
    print(f"         rasio nominal (tangan)  = {expected_ratio:.4f}")
    print(f"         rasio tool (opening/wall = {result['opening_px']}/{result['wall_px']}) = {actual_ratio:.4f}")
    print(f"         selisih relatif = {rel_err*100:.2f}% (ambang {TOLERANCE*100:.0f}%, "
          f"selisih diharapkan dari piksel garis kusen yang tak dihitung ke kategori manapun)")
    return rel_err <= TOLERANCE


def test_synthetic_opening_count(tmp_path):
    """Uji kebenaran hitung untuk JUMLAH bukaan (metodologi_skripsi.md #4.4's syarat
    pelaporan count+sebaran ukuran, bukan cuma rasio luas) -- ketiga bukaan sintetis
    (2 jendela + 1 pintu) sengaja diberi jarak lebar (35-150px, jauh di atas
    CLUSTER_GAP_PX) satu sama lain, jadi jawaban yang benar diketahui PERSIS: 3,
    tidak lebih (tergabung) atau kurang (terpecah)."""
    make_synthetic(tmp_path)
    result = analyze(tmp_path)
    ok = result["opening_count"] == 3
    status = "LULUS" if ok else "GAGAL"
    print(f"[{status}] uji kebenaran hitung jumlah bukaan (2 jendela+1 pintu, "
          f"terpisah jauh -> harus persis 3, didapat {result['opening_count']})")
    return ok


def test_gate_not_falsely_rejected():
    """Regresi: gerbang berkisi-kisi (tampak0, banyak batang tipis) tidak boleh
    ditolak sebagai "dinding berlubang" hanya karena tiap celah antar-batang secara
    teknis adalah lubang -- gerbang tetap harus terhitung SATU bukaan (pintu),
    persis seperti keputusan metodologi_skripsi.md #4.4 soal pintu."""
    example = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contoh-tampak", "tampak0.jpg")
    if not os.path.exists(example):
        print("[LEWAT] contoh-tampak/tampak0.jpg tidak ada, uji regresi gerbang dilewati")
        return True
    result = analyze(example)
    # Sebelum perbaikan MIN_HOLE_FRACTION+size-floor, opening_px turun jauh di bawah
    # ini karena kedua gerbang (dan jendela lain) salah tertolak.
    ok = result["opening_px"] > 100000
    status = "LULUS" if ok else "GAGAL"
    print(f"[{status}] regresi: gerbang berkisi-kisi tetap terhitung bukaan "
          f"(opening_px={result['opening_px']}, harus > 100000)")
    return ok


if __name__ == "__main__":
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_synth_tampak.png")
    try:
        ok1 = test_synthetic_ground_truth(tmp)
        ok2 = test_synthetic_opening_count(tmp)
        ok3 = test_gate_not_falsely_rejected()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print()
    all_ok = ok1 and ok2 and ok3
    print("SEMUA LULUS" if all_ok else "ADA YANG GAGAL")
    sys.exit(0 if all_ok else 1)
