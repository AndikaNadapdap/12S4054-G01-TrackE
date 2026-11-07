SEARCH SPACE EKSPERIMEN

Checkpoint 3 — Kelompok G01

1. Tujuan

Search space disusun untuk menentukan parameter dan konfigurasi yang digunakan dalam eksperimen rekomendasi. Seluruh pemilihan konfigurasi model dilakukan berdasarkan performa pada validation split.

Metrik utama adalah NDCG@10, sedangkan Recall@10 digunakan sebagai tie-breaker apabila terdapat nilai NDCG@10 yang sama. Test split tidak digunakan dalam proses tuning maupun pemilihan model.

2. Search Space Model

2.1 Popularity

Popularity digunakan sebagai baseline non-personalized recommendation.

Parameter	Konfigurasi
Dasar perhitungan	Jumlah interaksi positif pada training
Ambang rating positif	≥ 4
Pengurutan	Jumlah interaksi positif tertinggi
Tie-breaker	Item ID ascending
Tuning	Tidak dilakukan

2.2 Alternating Least Squares (ALS)

ALS digunakan sebagai baseline collaborative filtering dengan pendekatan implicit feedback.

Parameter	Kandidat Nilai
Factors	32, 64, 128
Regularization	0.01, 0.1, 1.0
Alpha	10, 40
Iterations	15
Random Seed	42

Total terdapat 18 kombinasi konfigurasi ALS yang direncanakan.

Seluruh konfigurasi ALS menggunakan interaksi positif pada training data sebagai implicit feedback. Interaksi dengan rating ≥ 4 diperlakukan sebagai positive preference.

Pada tahap implementasi saat ini, konfigurasi ALS yang telah dijalankan sebagai baseline menggunakan factors 64, regularization 0.1, alpha 10, dan 15 iterasi. Dengan demikian, 18 kombinasi merupakan ruang pencarian yang direncanakan dan tidak menunjukkan bahwa seluruh kombinasi telah dijalankan.

2.3 TF-IDF Content-Based Filtering

TF-IDF digunakan untuk membentuk representasi leksikal item berdasarkan atribut teks content_text_basic.

Tiga konfigurasi tuning yang dievaluasi:

Konfigurasi	N-gram Range	Max Features
TFIDF-A (E3-A)	(1,1)	5.000
TFIDF-B (E3-B)	(1,1)	20.000
TFIDF-C (E3-C)	(1,2)	20.000

Parameter tetap:

* Min Document Frequency: 2
* Max Document Frequency: 0.5
* Sublinear TF: True
* Stop Words: English
* Lowercase: True
* Normalisasi: L2

Ketiga konfigurasi dievaluasi menggunakan validation split. Konfigurasi terbaik dipilih berdasarkan NDCG@10 tertinggi, dengan Recall@10 sebagai tie-breaker.

Konfigurasi E3 digunakan sebagai baseline TF-IDF awal berdasarkan konfigurasi CP2, sedangkan E3-A hingga E3-C digunakan untuk mengevaluasi variasi ukuran vocabulary dan penggunaan bigram.

2.4 Sentence-Embedding Content-Based Filtering

Sentence Embedding digunakan untuk membentuk representasi semantik item berdasarkan atribut teks content_text_basic.

Parameter	Konfigurasi
Model	sentence-transformers/all-MiniLM-L6-v2
Dimensi Embedding	384
Max Sequence Length	128, 256, 512
Batch Size	64
Device	CPU
Normalisasi	L2
User Profile	Rata-rata representasi item positif pada training
Similarity	Cosine Similarity

Tiga konfigurasi max sequence length digunakan untuk mengamati pengaruh panjang input terhadap kualitas rekomendasi.

Konfigurasi E4 digunakan sebagai baseline sentence embedding dengan max sequence length 256. E4-A, E4-B, dan E4-C digunakan untuk mengevaluasi variasi panjang input.

3. Kesetaraan Budget Tuning

TF-IDF dan Sentence Embedding masing-masing menggunakan tiga konfigurasi untuk proses perbandingan.

Seluruh konfigurasi dievaluasi menggunakan validation split dan jumlah pengguna evaluasi yang sama.

Pemilihan konfigurasi terbaik dilakukan berdasarkan NDCG@10 tertinggi. Recall@10 digunakan sebagai tie-breaker apabila nilai NDCG@10 sama.

Kesetaraan jumlah konfigurasi tersebut diterapkan untuk menjaga agar perbandingan kedua pendekatan content-based filtering menggunakan budget tuning yang sebanding.

4. Experiment Matrix

ID	Model	Konfigurasi	Tujuan
E1	Popularity	Fixed, positive interaction count	Baseline popularitas
E2	ALS	Factors 64, regularization 0.1, alpha 10, 15 iterasi	Baseline collaborative filtering
E3	TF-IDF CBF	Konfigurasi awal CP2	Baseline representasi leksikal
E3-A	TF-IDF CBF	Unigram, max_features 5.000	Evaluasi ukuran vocabulary
E3-B	TF-IDF CBF	Unigram, max_features 20.000	Evaluasi ukuran vocabulary
E3-C	TF-IDF CBF	Unigram + bigram, max_features 20.000	Evaluasi penggunaan bigram
E4	Sentence Embedding	MiniLM-L6-v2, max sequence length 256	Baseline representasi semantik
E4-A	Sentence Embedding	Max sequence length 128	Evaluasi panjang input
E4-B	Sentence Embedding	Max sequence length 256	Evaluasi panjang input
E4-C	Sentence Embedding	Max sequence length 512	Evaluasi panjang input
E5	TF-IDF vs Sentence Embedding	Konfigurasi terbaik berdasarkan validation	Perbandingan berdasarkan strata aktivitas pengguna

Eksperimen E3-A hingga E3-C dan E4-A hingga E4-C menggunakan tiga konfigurasi untuk masing-masing pendekatan content-based filtering.

Eksperimen E5 merupakan tahap perbandingan antara konfigurasi TF-IDF dan Sentence Embedding yang dipilih berdasarkan hasil validation. Perbandingan dilakukan secara keseluruhan dan berdasarkan kelompok aktivitas pengguna:

* Low activity: 1–4 interaksi training.
* Medium activity: 5–9 interaksi training.
* High activity: ≥ 10 interaksi training.

Pemilihan konfigurasi pada E5 dilakukan tanpa menggunakan test split.

5. Reproducibility

Ketentuan reproducibility eksperimen:

* Random seed ditetapkan sebesar 42.
* Seluruh eksperimen menggunakan official training dan validation split.
* Official temporal split digunakan tanpa melakukan pembagian ulang data.
* Test split tidak digunakan dalam proses tuning maupun pemilihan model.
* Konfigurasi setiap eksperimen disimpan dalam file YAML.
* Hasil eksperimen dicatat dalam results/runs.jsonl.
* Embedding item disimpan dalam cache .npy.
* Hash teks digunakan untuk membantu memastikan kesesuaian cache dengan input teks.
* Informasi konfigurasi, Git commit, jumlah evaluable users, metrik, timestamp, seed, dan runtime dicatat untuk setiap run.
* Setiap hasil eksperimen harus dapat ditelusuri ke konfigurasi YAML dan versi kode yang digunakan.
* Perubahan working tree yang belum di-commit perlu diperhatikan ketika mencatat versi kode eksperimen.

6. Kriteria Pemilihan Konfigurasi

Konfigurasi terbaik dipilih berdasarkan ketentuan berikut:

1. NDCG@10 tertinggi pada validation split.
2. Recall@10 menjadi tie-breaker apabila nilai NDCG@10 sama.
3. Budget tuning TF-IDF dan Sentence Embedding dibuat setara, yaitu masing-masing tiga konfigurasi.
4. Pemilihan konfigurasi terbaik dilakukan hanya menggunakan validation split.
5. Test split hanya digunakan untuk evaluasi akhir sesuai ketentuan evaluasi final.

⸻

Status: Rancangan Search Space CP3.