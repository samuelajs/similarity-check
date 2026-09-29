# Serah Terima ke Claude Code — Tool Prototipe Kemiripan Bentuk Arsitektur

## 1. Konteks & Tujuan Besar

Skripsi Sam (S1 Arsitektur) awalnya soal "orisinalitas & plagiarisme arsitektur," tapi setelah masukan sidang proposal, dipangkas jadi **murni tool kuantitatif untuk mengukur kemiripan bentuk objek arsitektur** — tanpa klaim hukum/plagiarisme. Prototipe ini adalah bukti-konsep tool tersebut untuk mendukung penulisan Bab 3 (metodologi) skripsi.

**Lima komponen kemiripan yang direncanakan** (baru komponen 1 & sebagian 2 yang dikerjakan):
1. **Bentuk massa/siluet** — EFD (Elliptical Fourier Descriptors) + tumpang-tindih area → **sudah jalan & teruji baik**
2. **Topologi ruang** — Graph Edit Distance pada graf adjacency ruang → **baru tahap deteksi ruang, GED-nya sendiri belum dikerjakan**
3. Hierarki ruang (sentralitas graf) → belum dikerjakan
4. Rasio bukaan fasad → belum dikerjakan
5. Pola second-skin (Fourier 2D) → baru didemokan di Python (bukan di prototipe web)

## 2. Lokasi File

- **Prototipe utama (published sebagai Claude Artifact):** `/mnt/user-data/outputs/similarity_prototype.html` — satu file HTML mandiri (drawing kanvas manual + EFD + tumpang-tindih + deteksi siluet otomatis + deteksi ruang OCR + editor ruang)
- **Salinan standalone** (untuk dibuka lepas dari claude.ai): `/mnt/user-data/outputs/pembanding_kemiripan_bentuk_standalone.html` — isinya sama persis, disinkronkan manual tiap kali prototipe utama diubah
- **Framework Python yang jadi acuan/sumber logika** (dari Sam, sudah diverifikasi bekerja baik oleh Sam sendiri): `/home/claude/uploaded_framework/untuk-sam/`
  - `segment_denah.py` — kode inti: OCR (RapidOCR) + deteksi garis dinding + kotak ruangan + siluet luar
  - `demo_server.py` + `demo.html` — server Python + UI web terpisah (BUKAN client-side, butuh Python jalan di background)
  - `contoh/Gambar3.jpeg` — gambar kerja uji (denah kantor, TOILET/PANTRY/WAITING ROOM/NURSERY ROOM/3x MEETING ROOM/OFFICE/MUSHOLLA)
  - `hasil/` — output tervalidasi Sam sendiri dari Python (gambar3_solved.png, siluet_preview.png, JSON)
- **Bukti kegagalan versi JS/browser** (dari Sam, sangat penting sebagai bahan debug): `/home/claude/kegagalan/`
  - `Kegagalan Siluet.jpeg` — screenshot prototipe web: titik siluet mengikuti jalur DALAM bangunan (sekitar Meeting Room/Office), bukan lompat lurus ke sudut-sudut luar
  - `Siluet yang seharusnya.png` — bentuk L bersih yang benar (6 sudut)
  - `Segmentasi Ruang Claude.png` — hasil deteksi ruang versi JS: OCR salah baca ("POEMS", "[M]"), kotak-kotak tidak sejajar dinding, beberapa ruang bercampur jadi satu label
  - `Segmentasi Ruang Python.jpeg` — hasil Python yang benar (9 ruangan + 1 sirkulasi, semua sejajar dinding rapi) — ini identik dengan `hasil/gambar3_solved.png`

## 3. Yang SUDAH Diverifikasi Benar (jangan dirombak tanpa alasan kuat)

- **EFD (Elliptical Fourier Descriptors)** ditulis ulang di JS murni, diuji di Node: bentuk identik (termasuk diputar/diskalakan) → jarak 0,000; bentuk berbeda → jarak signifikan. Terbukti matematis benar dan **rotation-invariant secara inheren** (tidak butuh pencarian rotasi seperti metode tumpang-tindih).
- **Tumpang-tindih area + pencarian rotasi optimal**: bekerja, diuji dengan kasus persegi/L-shape.
- **Logika `boxFor` + `separateBoxes`** (pencarian kotak ruangan dari garis dinding terdekat, & resolusi tumpang-tindih antar kotak): diuji dengan data **PERSIS** dari Python (garis dinding & posisi label yang sama persis diekspor dari `segment_denah.py`) → **9 dari 9 kotak ruangan cocok sampai ke angka piksel terakhir**. Logika geometri intinya terbukti benar diterjemahkan.
- **Bug `paintRuns` (versi lama) ditemukan & diperbaiki**: versi lama salah menerjemahkan algoritma Python (bekerja pada larik padat, bukan daftar indeks jarang seperti Python asli) — terbukti lewat kasus uji sintetis (celah 1 piksel harusnya tersambung, versi lama malah memecah jadi dua).
- **Bug resolusi ditemukan**: diuji langsung di Python dengan menurunkan resolusi gambar `Gambar3.jpeg` secara bertahap — algoritma siluet **rusak di bawah ~800px lebar kerja** (dinding dalam ikut "bocor" jadi bagian siluet luar), bersih kembali di 900px+. Working resolution di JS sudah dinaikkan ke 1000px untuk kompensasi.

## 4. Yang MASIH GAGAL (bukti di folder `/home/claude/kegagalan/`)

**Meski dua bug di atas sudah diperbaiki, hasil di browser Sam MASIH salah** — titik siluet masih mengikuti jalur dalam bangunan, dan segmentasi ruang masih berantakan (OCR salah baca, kotak tidak sejajar). Ini artinya **ada bug lain yang belum ditemukan**, kemungkinan di salah satu atau kombinasi dari:

1. **`cv.distanceTransform` / `cv.threshold` versi OpenCV.js (`opencv.js-webassembly@4.2.0` dari jsDelivr) mungkin berperilaku beda dari `cv2` Python** — belum pernah diverifikasi langsung, cuma diasumsikan API-nya identik. Package ini kemungkinan berbasis versi OpenCV lama/berbeda dari `opencv-python` 4.13 yang dipakai untuk verifikasi Python.
2. **Kualitas Tesseract.js jauh di bawah RapidOCR** untuk kasus nyata ("POEMS", "[M]" adalah bacaan yang jelas ngaco, tidak ada padanannya di teks asli).
3. Kemungkinan ada bug lain di `inkLines`/`wallLines` (fungsi deteksi garis dinding untuk ruangan, terpisah dari fungsi siluet) yang belum diuji presisi seperti `outerSilhouetteCV`.
4. **Sudah dipasang tapi belum sempat dipakai**: alat debug visual (`debugOut` di `outerSilhouetteCV`, menyimpan mask `core`/`walls`/`solid` mentah) — infrastrukturnya ada di kode, tapi UI untuk menampilkannya ke Sam **belum selesai dipasang** saat sesi ini terhenti (tombol "Lihat Debug" sudah ditambahkan di HTML, canvas penampil & wiring JS-nya belum).

## 5. PERMINTAAN EKSPLISIT SAM — PRIORITASKAN PYTHON

> "Berikan saya rinciannya... minta agar prioritaskan penggunaan python agar bisa lebih baik hasilnya seperti pada contoh yang saya berikan zipnya."

Sam eksplisit minta **hasil akhir setara kualitas Python** (`Segmentasi Ruang Python.jpeg`, `hasil/gambar3_solved.png`), dan **JS/browser sejauh ini terbukti tidak mencapai itu** meski logika geometrinya sudah diverifikasi identik secara terpisah. Claude Code perlu **mempertimbangkan ulang arsitekturnya**, bukan cuma menambal bug JS lagi:

### Opsi A — Tetap 1 file HTML, JS-only (arsitektur saat ini)
Lanjutkan debug OpenCV.js/Tesseract.js. Risiko: sudah 2 siklus debug tidak menyelesaikan masalah; kemungkinan root cause di library pihak ketiga yang tidak sepenuhnya terkontrol.

### Opsi B — Python jadi backend, HTML/JS jadi frontend saja (RISIKO PALING RENDAH, DIREKOMENDASIKAN)
Pakai `segment_denah.py` (sudah terbukti benar oleh Sam sendiri) **apa adanya** sebagai backend — lewat server lokal kecil (mirip `demo_server.py` yang sudah ada di paket Sam), dipanggil dari halaman web yang sudah dibangun (mengirim gambar via `fetch()` ke `localhost`, menerima JSON kotak ruangan/siluet, ditampilkan pakai kanvas & editor yang SUDAH ada). Konsekuensi: Sam perlu menjalankan satu perintah Python di terminal sebelum pakai tool-nya (`python demo_server.py` atau sejenis) — bukan lagi murni "buka file HTML saja." Ini trade-off yang perlu dikonfirmasi ke Sam, tapi kemungkinan besar memberi hasil yang SAMA PERSIS dengan `hasil/gambar3_solved.png`, karena literally kode yang sama yang menghasilkan itu.

### Opsi C — Hibrida: EFD/tumpang-tindih/editor tetap JS (sudah bagus), deteksi siluet + ruang pindah ke Python backend
Kombinasi paling masuk akal: komponen yang **sudah terbukti sempurna di JS** (EFD, tumpang-tindih, drawing manual, editor ruang) **tetap di JS**, sementara komponen yang **terus gagal di JS** (siluet otomatis, deteksi ruang OCR) **dilempar ke Python backend** apa adanya dari `segment_denah.py`.

**Rekomendasi saya (Claude, sesi sebelumnya): mulai dari Opsi C.** Ini paling realistis mengejar kualitas Python tanpa membuang kerja JS yang sudah teruji bagus.

## 6. Daftar Tugas Konkret (urutan prioritas)

1. **Putuskan arsitektur** (A/B/C di atas) — sebaiknya didiskusikan dulu ke Sam sebelum coding besar, karena mengubah cara tool ini dijalankan (butuh Python jalan atau tidak).
2. Kalau pilih B/C: bangun server Python minimal (boleh reuse `demo_server.py`, sudah ada) yang menerima gambar, menjalankan `segment_denah.py` (baik fungsi `outer_silhouette` maupun pipeline OCR+ruangan penuh), balas JSON. Sambungkan ke UI HTML yang sudah ada (ganti pemanggilan `outerSilhouetteCV`/`detectRooms` versi JS dengan `fetch()` ke server ini).
3. Kalau tetap lanjut opsi A (JS-only): selesaikan dulu UI "Lihat Debug" yang sudah setengah jalan (tombol ada, canvas penampil & wiring-nya belum) — ini satu-satunya cara memverifikasi lebih lanjut tanpa Claude bisa melihat browser Sam langsung. Render `debugOut.core`, `debugOut.walls`, `debugOut.solid` (semua `Uint8Array` 0/1, ukuran `ww x wh`) sebagai gambar hitam-putih supaya Sam bisa screenshot dan menunjukkan persis di tahap mana algoritmanya salah.
4. **Constraint anti-tumpang-tindih untuk editor ruang manual** — sudah dipasang sebagian (`separateBoxes(state.rooms)` dipanggil tiap kali ruangan baru ditambah via drag), tapi **belum ada constraint saat resize/drag ulang kotak yang SUDAH ada** (fitur itu sendiri — drag untuk resize kotak existing — belum dibangun sama sekali, cuma delete+rename+add-new yang ada sekarang). Kalau mau penuh: tambahkan handle resize di tiap kotak ruangan di kanvas editor, jalankan `separateBoxes` tiap kali handle digeser.
5. Setelah arsitektur & deteksi ruang/siluet solid, baru lanjut ke Graph Edit Distance (komponen 2) — belum dikerjakan sama sekali, cuma dirancang konsepnya (lihat riwayat chat metodologis, bukan chat ini).

## 7. Batasan yang Perlu Diingat Claude Code

- Ini **prototipe pendukung skripsi**, bukan produk — prioritaskan **kebenaran hasil yang bisa dipertanggungjawabkan di sidang** di atas kerapian arsitektur kode.
- Semua keputusan besar (terutama soal arsitektur A/B/C di atas) **sebaiknya dikonfirmasi ke Sam dulu**, karena berdampak ke cara dia menjalankan/mendemokan tool ini nanti.
- Chat ini (Claude di claude.ai) akan fokus ke sisi metodologis skripsi (Bab 2/3, argumentasi hukum, dsb.) — tidak lagi menangani coding tool ini setelah serah terima ini.
