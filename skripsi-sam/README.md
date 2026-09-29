# Untuk Sam

Folder ini saja yang diserahkan. Isinya cukup untuk mencoba deteksi ruang sampai progres sekarang.

## Yang ada di folder ini

| File | Untuk apa |
|---|---|
| `jalankan.bat` | Menjalankan demo di komputer |
| `demo.html` | Halaman unggah gambar |
| `demo_server.py` | Menerima gambar dan mengembalikan hasil. Tidak online. |
| `segment_denah.py` | Kode yang membaca denah |
| `requirements.txt` | Pustaka Python yang dibutuhkan |
| `contoh/Gambar3.jpeg` | Contoh denah berlabel bahasa Inggris |

## Cara mencoba

1. Pasang Python 3.
2. Klik dua kali `jalankan.bat`.
3. Browser membuka `http://127.0.0.1:8765/`.
4. Tekan **Pakai contoh Gambar3**, lalu **Deteksi ruang**. Atau pilih gambar denah lain.

Halaman punya tiga mode. **Ruang** dan **Siluet** menyimpan titik ke `hasil/<nama-gambar>-ruang.json` dan `hasil/<nama-gambar>-siluet.json`. **Banding siluet** membaca dua berkas siluet itu dan menghitung kemiripan batas luar setelah ukuran disamakan. Nama hanya diisi jika tulisan itu terbaca pada gambar yang sedang dibuka. Poligon baru ditambah dengan tombol Tambah poligon, lalu ditutup dengan tombol Tutup poligon atau klik lingkaran di titik pertama.

## Yang tidak ikut

Folder percobaan (`tools`, model CubiCasa, skrip lama) tidak diperlukan untuk demo ini. Hosting dan akun Hugging Face juga tidak dipakai.
