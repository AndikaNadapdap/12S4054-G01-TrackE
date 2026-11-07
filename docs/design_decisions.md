DOKUMEN KEPUTUSAN DESAIN EKSPERIMEN

Checkpoint 3 — Kelompok G01

1. Informasi Proyek

Mata Kuliah: Recommender Systems (12S4054)

Kelompok: G01

Judul Penelitian:

Lexical or Semantic: A Controlled Comparison of TF-IDF and Sentence-Embedding Item Representations for Top-N Recommendation on Amazon Musical Instruments.

Track: Track E — Lexical vs Semantic.

Tujuan Dokumen:

Dokumen ini mencatat keputusan desain eksperimen yang telah disepakati oleh seluruh anggota kelompok untuk pelaksanaan Checkpoint 3 (CP3). Dokumen ini menjadi acuan bersama dalam proses implementasi, eksperimen, evaluasi, dan penyusunan manuskrip.

⸻

2. Pemilihan Model Embedding

Model utama: all-MiniLM-L6-v2

Alasan pemilihan:

* Model sudah dikutip dalam bagian Related Work pada CP2.
* Memiliki ukuran yang relatif ringan sehingga sesuai untuk eksekusi menggunakan CPU.

Model alternatif (opsional):

* BLaIR atau MPNet.

Penggunaan model alternatif hanya dipertimbangkan apabila waktu dan sumber daya komputasi mencukupi.

Penanggung jawab: Pariama.

Status: Disetujui.

⸻

3. Panjang Maksimum Input Embedding

Konfigurasi berikut digunakan untuk menguji pengaruh panjang input terhadap representasi Sentence Embedding.

Konfigurasi	max_seq_length
E4-A	128
E4-B	256
E4-C	512

Seluruh konfigurasi akan dievaluasi menggunakan validation split.

Penanggung jawab: Pariama.

Status: Disetujui.

⸻

4. Pembagian Strata Aktivitas Pengguna (RQ2)

Aktivitas pengguna diukur berdasarkan jumlah interaksi pada training data.

Pembagian strata pengguna:

Kelompok Pengguna	Jumlah Interaksi Training
Low Activity	1–4
Medium Activity	5–9
High Activity	≥10

Batas setiap strata ditetapkan sebelum hasil eksperimen diperiksa.

Jumlah pengguna pada setiap strata juga akan diperiksa untuk memastikan keterwakilan data.

Penanggung jawab: Joy.

Status: Disetujui.

⸻

5. Parameter Controlled Comparison

Untuk memastikan perbandingan yang adil antara TF-IDF CBF dan Sentence-Embedding CBF, komponen berikut menggunakan konfigurasi yang sama.

Komponen	Konfigurasi
Input teks	content_text_basic
Pembentukan User Profile	Rata-rata representasi item positif
Fungsi Similarity	Cosine Similarity
Candidate Set	Warm-start candidate universe dan aturan filtering yang sama

Perbedaan utama kedua pendekatan terletak pada representasi item:

* TF-IDF: Lexical Representation.
* Sentence Embedding: Semantic Representation.

Pengaturan ini bertujuan mengisolasi pengaruh representasi item terhadap kualitas rekomendasi.

Penanggung jawab: Joy dan Pariama.

Status: Disetujui.

⸻

6. Kriteria Pemilihan Model (Model Selection)

Pemilihan konfigurasi model dilakukan berdasarkan performa pada validation split.

Kriteria pemilihan:

1. Konfigurasi dengan nilai NDCG@10 tertinggi pada validation split dipilih sebagai konfigurasi terbaik.
2. Recall@10 digunakan sebagai tie-breaker apabila terdapat nilai NDCG@10 yang sama.
3. Budget tuning TF-IDF dan Sentence Embedding dibuat setara.

Test split tidak digunakan dalam proses pemilihan model.

Penanggung jawab: Andika.

Status: Disetujui.

⸻

7. Pengujian Signifikansi Statistik (Statistical Significance Testing)

Metode pengujian statistik yang digunakan adalah Paired Bootstrap.

Parameter	Konfigurasi
Metode	Paired Bootstrap
Jumlah Resampling	1.000
Random Seed	42
Confidence Interval	95%

Prosedur bootstrap digunakan untuk mengestimasi confidence interval dari perbedaan performa antara TF-IDF CBF dan Sentence-Embedding CBF.

Penanggung jawab: Joy.

Status: Disetujui.

⸻

8. Persetujuan Kelompok

Seluruh keputusan desain eksperimen telah disepakati oleh keempat anggota kelompok dan menjadi acuan pelaksanaan CP3.

No.	Nama Anggota	Status Persetujuan
1	Joy Valeda Silalahi	Disetujui
2	Andika Immanuel Nadapdap	Disetujui
3	Pariama Valentino Sihombing	Disetujui
4	Gracia Pardede	Disetujui

Status Dokumen: Final — Disetujui Seluruh Anggota Kelompok.

⸻

9. Riwayat Revisi

Versi	Keterangan
0.1	Penyusunan awal rancangan desain eksperimen CP3
1.0	Finalisasi keputusan desain eksperimen berdasarkan persetujuan seluruh anggota kelompok