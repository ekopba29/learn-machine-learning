# Laporan Proyek Machine Learning - Eko

## Project Overview
Dalam beberapa tahun terakhir, industri anime telah tumbuh secara signifikan di seluruh dunia. Menyaksikan anime telah menjadi hobi yang populer bagi banyak orang, dan dengan ribuan judul anime yang tersedia, pemirsa seringkali menghadapi kesulitan dalam memilih anime yang sesuai dengan preferensi mereka. Inilah mengapa rekomendasi anime yang efektif menjadi penting. Machine learning menjadi solusi yang sangat efisien dalam menghadapi tantangan ini.

Proyek ini penting karena dapat meningkatkan pengalaman menonton anime penggemar dan membantu platform streaming meningkatkan retensi pengguna. Sistem rekomendasi yang kuat, berdasarkan teknologi machine learning, dapat menganalisis sejarah tontonan pengguna, preferensi, dan pola-pola yang sulit dikenali secara manual. Dengan pemahaman yang lebih baik tentang preferensi individu, sistem ini dapat merekomendasikan anime yang lebih sesuai, memungkinkan penggemar untuk menemukan konten yang mungkin mereka lewatkan atau tidak pernah ditemui sebelumnya. Hal ini menciptakan pengalaman menonton yang lebih memuaskan dan memikat.

Menurut tesis "Peningkatan Popularitas Anime Jepang di Pasar Amerika Serikat di Era Pandemi Covid-19" oleh Aldo Priyo Banyu Aji et al. (2022), "Saat layanan streaming mulai bermunculan, distribusi anime yang awalnya menggunakan DVD dan kontrak dengan stasiun TV beralih menjadi pemasaran digital melalui penjualan hak siar kepada penyedia layanan streaming. Digitalisasi ini bertepatan dengan terjadinya pandemi Covid-19 yang memaksa orang-orang melakukan hampir seluruh kegiatan sehari-hari di rumah termasuk dalam hal mencari hiburan. Banyaknya waktu luang yang dimiliki masyarakat Amerika Serikat menyebabkan mereka kembali mengakses anime dan mendorong peningkatan popularitas anime di pasar Amerika Serikat".

Dalam penelitian yang dilakukan oleh Az Zayyad dan Kurniawardhani (2020) yang berjudul "Penerapan Metode Deep Learning pada Sistem Rekomendasi Film," disimpulkan bahwa pemanfaatan deep learning dalam sistem rekomendasi film memberikan hasil yang sangat positif. Hasil penelitian menunjukkan bahwa deep learning, termasuk metode seperti Restricted Boltzmann Machine dan Autoencoder, memiliki akurasi dan performa yang baik dalam memberikan rekomendasi pada sebuah sistem. Penelitian ini juga mengemukakan bahwa penggunaan deep learning dalam sistem rekomendasi mampu meningkatkan kepuasan pengguna aplikasi tersebut. Selain itu, metode ini banyak digunakan oleh berbagai perusahaan di industri teknologi, mengindikasikan popularitas dan efektivitasnya dalam meningkatkan kualitas layanan rekomendasi.


## Business Understanding
Anggap saja suatu situs anime yang sedikit pengunjung padahal anime yang disediakan cukup lengkap. Bisa saja situs sedikit pengunjung karena mereka tidak dapat menemukan apa yang mereka inginkan. Nah padahal sudah menyediakan cukup lengkap tapi mengapa pengunjung masih tidak dapat menemukan anime. 
Tentu website harus ditingkatkan kembali terutama fitur pencarian atau fitur rekomendasi.Setiap pengguna anime memiliki preferensi yang berbeda. Sistem rekomendasi ML memungkinkan platform untuk menyediakan pengalaman yang sangat disesuaikan, memastikan bahwa pengguna mendapatkan konten yang benar-benar mereka nikmati.
Ketika pengguna melihat konten yang relevan, mereka lebih cenderung terlibat dan terus menggunakan platform. Ini dapat meningkatkan tingkat retensi pengguna dan meningkatkan frekuensi kunjungan. Dengan sistem rekomendasi yang efektif, pengguna cenderung kembali ke platform secara berkala. Hal ini mengurangi tingkat churn (pengguna yang berhenti menggunakan platform) dan membantu dalam mempertahankan pangsa pasar.

Masalah yang ingin diselesaikan dengan pendekatan Content-Based Filtering adalah memberikan rekomendasi anime kepada pengguna berdasarkan karakteristik dan konten dari anime yang sudah mereka tonton sebelumnya. Teknik ini akan mempertimbangkan faktor-faktor seperti genre, studio pembuat, tahun rilis, dan elemen-elemen lain yang terkait dengan setiap anime. Hal ini bertujuan untuk:

Pendekatan Collaborative Filtering, masalah yang ingin diselesaikan dengan pendekatan Collaborative Filtering adalah memberikan rekomendasi anime kepada pengguna berdasarkan perilaku tontonan pengguna itu sendiri dan kesamaan dengan pengguna lain. Teknik ini akan mempertimbangkan faktor-faktor seperti peringkat yang diberikan pengguna pada anime tertentu, preferensi yang mirip dengan pengguna lain, dan interaksi pengguna dengan platform


Bagian laporan ini mencakup:

### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi harga diamonds untuk menjawab permasalahan berikut.
- Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?
- Dengan data rating yang dimiliki, bagaimana perusahaan dapat merekomendasikan restoran lain yang mungkin disukai dan belum pernah dikunjungi oleh pengguna? 

### Goals
Untuk  menjawab pertanyaan tersebut, buatlah sistem rekomendasi dengan tujuan atau goals sebagai berikut:
- Menghasilkan sejumlah rekomendasi anime yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
- Menghasilkan sejumlah rekomendasi anime yang sesuai dengan preferensi pengguna dan belum pernah dikunjungi sebelumnya dengan teknik collaborative filtering.


- Untuk meraih tujuan sistem rekomendasi anime yang dipersonalisasi dengan teknik berbasis kolaborasi / Collaborative Filtering, maka diusulkan dua pendekatan berikut: 

    ### Solution statements
    - Collaborative Filtering Approach, rekomendasi akan berfokus pada penggunaan perilaku tontonan pengguna dan kesamaan dengan pengguna lain untuk memberikan rekomendasi anime yang sesuai. Pendekatan ini akan menciptakan rekomendasi berdasarkan pengalaman kolektif pengguna.
    - Content-Based Approach: Dalam pendekatan Content-Based, rekomendasi akan diberikan berdasarkan analisis konten atau fitur-fitur dari anime itu sendiri, seperti genre, tipe (movie, TV, OVA, dll.),  Pendekatan ini akan menciptakan rekomendasi yang lebih dipersonalisasi berdasarkan preferensi konten pengguna.
    
## Data Understanding
Kumpulan dataset  berisi informasi mengenai data preferensi pengguna dari 73.516 pengguna terhadap 12.294 anime. Terdapat 2 file yaitu anime.csv dan rating.csv [Keggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data).

Variabel-variabel pada anime.csv dataset adalah sebagai berikut:
- anime_id - ID unik myanimelist.net yang mengidentifikasi sebuah 
- name - nama lengkap anime.
- genre - daftar genre anime yang dipisahkan oleh 
- type - film, TV, OVA, dll.
- episodes - jumlah episode dalam serial ini. (1 jika film).
- rating - rating rata-rata dari 10 untuk anime ini.
- members - jumlah anggota komunitas yang ada dalam "kelompok" anime ini.

Variabel-variabel pada anime.csv dataset adalah sebagai 
- user_id - ID pengguna yang dihasilkan secara acak dan tidak dapat diidentifikasi.
- anime_id - anime yang dinilai oleh pengguna ini.
- rating - penilaian dari 10 yang diberikan oleh pengguna ini (-1 jika pengguna menontonnya tetapi tidak memberikan penilaian).

Untuk memahami data yang ada dilakukan beberapa EDA yang menghasilkan :
- Total data anime di dataset ini sebanyak 12.294 judul.
- Terdapat beberapa data yang tidak lengkap (NA) dalam dataset anime, yaitu:
    - Genre: Terdapat 62 entri anime yang tidak memiliki informasi genre.
    - Type: Terdapat 25 entri anime yang tidak memiliki informasi tipe (TV, Movie, OVA, dll.).
    - Rating: Terdapat 230 entri anime yang tidak memiliki informasi peringkat rating.
- Total anime yang sudah di rating sebanyak 11.200 judul.
- Untuk data NA dilakukan drop karena pertimbangan data yang cukup banyak, supaya tidak mempengaruhi rekomendasi.
- Tidak ada data NA dalam dataset rating, yang berarti dataset ini lengkap.
- :::::::

## Data Preparation
Supaya data-data yang ada bisa dijadikan data latih maka dilakukan proses sebagai berikut:
- Menghapus data NA pada data anime di anime.csv, karena data yang tidak lengkap tidak memberikan informasi yang cukup untuk analisis dan rekomendasi yang baik.
- Merename kolom rating pada dataset anime menjadi rating_all agar tidak tertimpa dengan kolom "Rating" dari dataset rating saat kedua dataset digabungkan. Hal ini penting agar data rating pada dataset anime tetap tersedia setelah penggabungan.
- menggabungkan dataset anime dan rating menjadi satu dataframe supaya kita bisa mengetahui detail anime didalam rating.
- Menghapus duplikasi data pada dataframe hasil gabungan dua dataset berdasarkan user_id, anime_id, Duplikasi data pada dataframe hasil gabungan dua dataset harus dihapus agar data yang digunakan dalam pengembangan model bersih dan tidak mengandung duplikasi. Hal ini untuk menghindari bias dalam hasil rekomendasi.
- Mengubah anime_id menjadi list tanpa nilai yang sama
- Melakukan proses encoding anime_id, ini digunakan dalam proses prediksi
- Melakukan proses encoding angka ke anime_id, ini digunakan dalam proses prediksi
- Mapping anime_id ke dataframe anime_id_encoded, ini digunakan dalam proses prediksi
- Mapping anime_id ke dataframe anime_id_encoded, ini digunakan dalam proses prediksi
- Mengubah rating menjadi nilai float, untuk memastikan semua rating bertipe float
- Membagi dataset menjadi 80% data digunakan untuk melatih model dan 20% untuk mengukur akurasi dan kinerja model.
- MinMaxScaler pada label train dan transform ke label test supaya menghasilkan RMSE yang lebih rendah atau bisa dibilang untuk menghasilkan prediksi yang lebih baik.
- Menggabungkan tipe dan genre kedalam kolom type_and_genre guna keperluan content base filtering, karena content base learning menggunakan 1 kolom saja yang bisa digunakan untuk dilatih
- Untuk kolom type_genre yang NA direplace dengan string kosong, untuk menghindari eror ketika notebook dijalankan

## Modeling
#### Collaborative Filtering
Menggunakan metode embedding untuk merepresentasikan pengguna (users) dan anime dalam bentuk vektor yang memuat informasi tentang hubungan antara pengguna dan anime. Representasi vektor ini memungkinkan model untuk memahami pola interaksi antara pengguna dan anime dalam ruang berdimensi rendah.

Metode embedding digunakan dalam collaborative filtering untuk mengonversi informasi tentang pengguna dan item ke dalam representasi vektor yang padat dan kontinu. Representasi ini memungkinkan model untuk memahami hubungan antara pengguna dan item dalam ruang vektor yang terdefinisi dengan baik.

Dalam model yang dibuat, memiliki dua layer embedding terpisah: satu untuk pengguna (user_embedding) dan satu untuk anime (anime_embedding). Masing-masing layer ini memiliki bobot yang dipelajari selama pelatihan model untuk menghasilkan representasi embedding yang optimal. Embedding ini kemudian digunakan untuk menghitung prediksi peringkat anime berdasarkan pola interaksi antara pengguna dan anime.

Metode embedding adalah salah satu teknik yang sangat berguna dalam sistem rekomendasi karena memungkinkan model untuk memahami hubungan kompleks antara entitas (seperti pengguna dan anime) dalam ruang berdimensi rendah yang dapat dipelajari secara efisien selama pelatihan.

Menggunakan Dropout, Dropout digunakan untuk menghindari overfitting dengan secara acak menonaktifkan sebagian kecil neuron dalam jaringan selama pelatihan.

Contoh : 

| ID Pengguna | Nama Pengguna | ID Film | Judul Film | Peringkat |
|-------------|---------------|--------|------------|-----------|
| 0           | Alice         | 0      | Inception  | 4.5       |
| 1           | Bob           | 1      | Shawshank Redemption | 5.0 |
| 2           | Carol         | 2      | Interstellar | 4.0     |
| 0           | Alice         | 3      | The Matrix | 4.2       |
| 1           | Bob           | 4      | Gladiator | 4.8         |
| 2           | Carol         | 0      | Inception | 4.7        |

Ketika menerapkan metode embedding, menggantikan nama pengguna dan judul film dengan ID unik. Hasilnya mungkin terlihat seperti ini:

| ID Pengguna | ID Film | Peringkat |
|-------------|---------|-----------|
| 0           | 0       | 4.5       |
| 1           | 1       | 5.0       |
| 2           | 2       | 4.0       |
| 0           | 3       | 4.2       |
| 1           | 4       | 4.8       |
| 2           | 0       | 4.7       |

Kemudian, mengembangkan model yang menggunakan embedding untuk menghasilkan vektor yang merepresentasikan pengguna dan film. Misalnya, dapat memiliki vektor embedding untuk Alice (pengguna) dan vektor embedding untuk Inception (film).

Dengan model ini, dapat menghitung seberapa mirip vektor embedding pengguna dan vektor embedding film. Semakin mirip kedua vektor tersebut, semakin besar kemungkinan Alice akan memberikan peringkat tinggi pada film Inception.

Jadi, metode embedding membantu mengubah data entitas menjadi representasi numerik yang bisa digunakan untuk memodelkan preferensi pengguna terhadap item, yang pada gilirannya dapat digunakan untuk membuat rekomendasi yang lebih baik.

Kelebihan dari pendekatan berbasis kolaboratif adalah kemampuannya untuk memberikan rekomendasi yang sangat personal berdasarkan pola peringkat pengguna yang serupa. Ini sangat berguna dalam mengidentifikasi preferensi yang mungkin tidak jelas dari data anime yang pernah dilihat oleh pengguna. Namun, pendekatan ini memiliki beberapa kekurangan, seperti sensitivitas terhadap perubahan dalam preferensi pengguna dan kebutuhan akan data peringkat yang signifikan.

Top N: 
Algoritma ini mencari pengguna yang memiliki preferensi serupa dengan pengguna target dan kemudian merekomendasikan item yang disukai oleh pengguna serupa ini. Misalnya, jika pengguna A dan B memiliki preferensi yang serupa, jika pengguna A suka Anime X, maka pengguna B mungkin juga akan dianjurkan untuk menonton Anime X.

Dari rekomendasi sistem yang dibuat berdasarkan yang telah dibuat Content Base Filtering, didapatkan hasil yang memuaskan. Sebagai contoh user menyukai anime berjudul "Lupin III: Lupin Ikka Seizoroi" yang memiliki genre "Special Adventure, Comedy, Shounen" sistem rekomendasi merekomendasikan beberapa anime berikut :
|      name                                   |        genre                             |
|--------------------------------------------|----------------------------------------|
| City Hunter: Goodbye My Sweetheart        | Adventure, Comedy, Shounen              |
| Lupin III: The Last Job                    | Adventure, Comedy, Shounen              |
| Lupin III: Otakara Henkyaku Daisakusen!!   | Adventure, Comedy, Shounen              |
| Lupin VIII                                 | Adventure, Comedy, Shounen              |
| Lupin III: Sweet Lost Night - Mahou no Lamp wa... | Adventure, Comedy, Shounen     |

#### Content Base Filtering
Menggunakan TfidfVectorizer, digunakan untuk mengonversi teks pada kolom "type_and_genre" dari dataset anime menjadi representasi vektor berdasarkan nilai TF-IDF (Term Frequency-Inverse Document Frequency). 

Nilai Tf-idf adalah kombinasi dari frekuensi kata dalam dokumen (Tf) dan kebalikannya dalam seluruh kumpulan dokumen (Idf). Tf (Term Frequency): Ini adalah jumlah berapa kali kata tertentu muncul dalam dokumen tersebut. Ini membantu mengukur pentingnya kata dalam konteks dokumen tersebut. Idf (Inverse Document Frequency): Ini mengukur seberapa unik dan pentingnya kata dalam seluruh kumpulan dokumen. Kata-kata yang jarang muncul di seluruh dokumen mendapatkan skor Idf yang tinggi.

Content Based Filtering sistem rekomendasi akan menggunakan metode berbasis konten untuk memberikan rekomendasi kepada pengguna. Algoritma ini akan menganalisis karakteristik anime, seperti genre, tipe, episode, dan rating, untuk memberikan rekomendasi anime yang memiliki karakteristik serupa dengan anime yang pernah disukai oleh penggun

Top N :
- TfidfVectorizer: Algoritma berbasis konten sering menggunakan metode seperti TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengukur sejauh mana kata-kata atau atribut tertentu penting dalam menggambarkan suatu anime. Dengan metode ini, sistem dapat mengukur kesamaan antara anime berdasarkan fitur-fitur dalam kolom type_and_genre.
- Cosine Similarity: Untuk menghasilkan "top N rekomendasi" dalam algoritma berbasis konten, sistem akan menghitung kesamaan kosinus antara preferensi anime yang sudah ditonton dan kolom type_and_genre. Anime dengan kesamaan kosinus tertinggi akan direkomendasikan. Sistem akan mengambil N anime dengan kesamaan tertinggi dan merekomendasikannya kepada pengguna.

Dalam test yang sudah dilakukan sistem dapat merekomendasikan anime yang belum pernah dirating oleh user dan memberikan rekomendasi yang hampir memiliki kemiripan di genre dengan anime yang sudah pernah dia rating sebelumnya. Sistem juga merekomendasikan 

##### Anime yang direkomendasikan berdasarkan kemiripan kesukaan dengan user yang di test.
| Anime ID | Title                                    | Genre                                                |
|----------|------------------------------------------|------------------------------------------------------|
| 5630     | Higashi no Eden                          | Action, Comedy, Drama, Mystery, Romance, Sci-Fi, Thriller |
| 5680     | K-On!                                    | Comedy, Music, School, Slice of Life                |
| 6637     | Higashi no Eden Movie II: Paradise Lost | Action, Comedy, Drama, Mystery, Romance, Thriller    |
| 596      | Chobits: Chibits                         | Comedy, Romance                                      |
| 357      | Bokusatsu Tenshi Dokuro-chan             | Comedy, Ecchi, Magic                                 |

##### Top 10 Anime Recommendations
| Anime ID | Title                                  | Genre                                         |
|----------|----------------------------------------|-----------------------------------------------|
| 4181     | Clannad: After Story                   | Drama, Fantasy, Romance, Slice of Life, Supernatural |
| 7311     | Suzumiya Haruhi no Shoushitsu          | Comedy, Mystery, Romance, School, Sci-Fi, Supernatural |
| 28171    | Shokugeki no Souma                     | Ecchi, School, Shounen                        |
| 16498    | Shingeki no Kyojin                     | Action, Drama, Fantasy, Shounen, Super Power  |
| 4224     | Toradora!                              | Comedy, Romance, School, Slice of Life        |
| 10620    | Mirai Nikki (TV)                       | Action, Mystery, Psychological, Shounen, Supernatural, Thriller |
| 14227    | Tonari no Kaibutsu-kun                 | Comedy, Romance, School, Shoujo, Slice of Life |
| 8247     | Bleach Movie 4: Jigoku-hen             | Action, Comedy, Shounen, Super Power, Supernatural |
| 14467    | K                                      | Action, Super Power, Supernatural             |
| 8074     | Highschool of the Dead                 | Action, Ecchi, Horror, Supernatural           |

## Evaluation
Root Mean Square Error (RMSE) adalah metrik yang digunakan untuk mengukur kualitas rekomendasi dalam konteks sistem rekomendasi. RMSE mengukur sejauh mana perbedaan antara nilai yang diprediksi oleh sistem rekomendasi dengan nilai sebenarnya (observasi atau peringkat yang diberikan oleh pengguna) dalam bentuk akar kuadrat rata-rata dari selisih kuadrat antara nilai prediksi dan nilai sebenarnya.
Semakin rendah nilai RMSE, semakin baik kualitas rekomendasi sistem rekomendasi, karena ini mengindikasikan bahwa sistem memiliki kemampuan yang lebih baik untuk memprediksi preferensi pengguna. Sebaliknya, RMSE yang tinggi menunjukkan tingkat ketidakakuratan dalam prediksi sistem. Dengan menggunakan RMSE, pengembang sistem rekomendasi dapat mengukur dan memantau sejauh mana rekomendasi akurat.

Pada ML (Machine Learning) rekomendasi anime, penggunaan loss Binary Cross-Entropy (BCE) berkaitan dengan penginginan untuk membuat prediksi biner, yaitu apakah pengguna akan suka atau tidak suka terhadap suatu item atau anime tertentu. Binary Cross-Entropy: BCE cocok untuk tugas ini karena mencoba mengukur "ketidakpastian" model dalam membuat prediksi biner. Ketidakpastian ini berkaitan dengan sejauh mana model yakin dalam memprediksi apakah pengguna akan suka atau tidak suka. Jika model sangat yakin dalam prediksi, BCE mendekati nol. Jika model tidak yakin atau salah dalam prediksi, nilai BCE akan meningkat

Optimizer Adam digunakan untuk mengoptimalkan model rekomendasi anime dengan mengatur tingkat pembelajaran (learning rate) secara adaptif. Tujuannya adalah agar model dapat belajar dari data pelatihan dengan efisien dan menghasilkan prediksi yang lebih akurat tentang preferensi pengguna terhadap anime. Hal ini membantu meningkatkan kualitas rekomendasi yang diberikan kepada pengguna.

- Pada ML rekomendasi anime ini menggunakan loss Binary Cross-Entropy, digunakan karena menginginkan prediksi biner, yaitu apakah pengguna akan suka atau tidak.
- Menggunakan Optimizer Adam,  ini bertugas untuk menyesuaikan bobot model berdasarkan data pelatihan. Dalam konteks ini, model akan belajar dari riwayat peringkat pengguna terhadap anime. Optimizer Adam digunakan untuk mengoptimalkan pembelajaran dengan mengatur tingkat pembelajaran (learning rate).
- Menggunakan metrik Root Mean Squared Error, Metrik ini digunakan untuk mengukur kualitas rekomendasi yang diberikan oleh model. Root Mean Squared Error (RMSE) mengukur seberapa akurat prediksi model terhadap peringkat anime. Semakin kecil nilai RMSE, semakin akurat model dalam merekomendasikan anime yang sesuai dengan preferensi pengguna.


## Conclusion
Dari dua jenis rekomendasi, baik yang bersumber dari metode kolaboratif maupun berbasis konten, keduanya memiliki keunggulan tersendiri dalam memberikan rekomendasi kepada pengguna. Metode berbasis konten mampu memberikan rekomendasi anime berdasarkan kemiripan dengan anime yang disukai oleh pengguna. Sebagai contoh, jika pengguna menyukai suatu judul anime, sistem akan merekomendasikan anime lain yang memiliki karakteristik serupa.

Sementara itu, metode kolaboratif dapat memberikan rekomendasi anime yang memiliki potensi besar untuk membuat pengguna tertarik. Rekomendasi ini didasarkan pada data pengguna lain yang memiliki preferensi serupa. Sehingga, pengguna dapat menemukan anime yang belum pernah mereka tonton sebelumnya, tetapi memiliki kesamaan dengan selera mereka.

Kombinasi kedua jenis rekomendasi ini dapat memberikan pengalaman yang lebih kaya dan memuaskan kepada pengguna. Dengan begitu, sistem rekomendasi dapat membantu pengguna menemukan anime-anime yang sesuai dengan preferensi mereka dan bahkan mengeksplorasi anime baru yang mungkin tidak mereka ketahui sebelumnya.


#### Collaborative Filtering
Sebagai catatan penting, perlu diakui bahwa metode kolaboratif yang telah diimplementasikan masih menunjukkan tingkat RMSE sebesar 0.3085 dalam pemberian rekomendasi. Oleh karena itu, terus pengembangan dan penyempurnaan metode kolaboratif menjadi tantangan yang perlu dihadapi guna meningkatkan kualitas rekomendasi anime kepada pengguna.
**---Ini adalah bagian akhir laporan---**
Az Zayyad, M.R., & Kurniawardhani, A. (Tahun tidak disebutkan). Penerapan Metode Deep Learning pada Sistem Rekomendasi Film. Jurnal [Nama Jurnal], [Volume Jurnal](Nomor Jurnal), [Halaman-Halaman Artikel]. Diakses dari https://journal.uii.ac.id/AUTOMATA/article/view/17426/10934