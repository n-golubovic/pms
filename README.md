#  Prepoznavanje žanrova filmova na osnovu opisa

## Jupyter notebooks

1. [Data explore and filter](0-data-explore-and-filter.ipynb)
   - pregled i filtriranje podataka
2. [Train val test split](1-train-test-split.ipynb)
   - podjela podataka na trening, validacioni i testni skup
3. [Learning rate estimation](2-estimate-learning-rate.ipynb)
   - Notebook za estimaciju learning rate-a
4. [Train and model estimate](3-train-and-estimate.ipynb)
   - Treniranje modela i evaluacija validacionim skupom
5. [Test](4-test.ipynb)
   - evaluacija najboljeg modela testnim skupom
5. [Processing movies](5-process-all-movies.ipynb)
   - Označavnje žanrova za sve filmove
6. [Gzip ML-KNN classification](6-zip-classification.ipynb)
   - Zanimljiiv pristup za klasifikaciju žanrova

## Python skripte

- [Pregled podataka](app_0_data_review.py)
- [Postavka Milvus baze](app_1_db_setup.py)
  - Popunjavanje Milvus baze sa vektorima
- [Movie Search App](app_2_ui.py)
  - Aplikacija za pretragu i preporuku filmova
  - Implementirana kao GUI aplikacija sa Tkinter bibliotekom
- [ML-KNN classification](app_3_zip_local_example.py)
  - Skripta za lokalnu klasifikaciju žanrova.
  - Isti kod kao i u notebook pristupu [Gzip ML-KNN classification](6-zip-classification.ipynb)

## Podaci

- movie_metadata.csv
  - Metapodaci o filmovima iz originalnog skupa podataka
- movie_metadata_filtered.csv
  - Filtrirani metapodaci o filmovima dobijeni nakon izvršenja prvog notebooka
- test.csv, train.csv, val.csv
  - Podaci za trening, validaciju i testiranje modela dobijeni nakon izvršenja drugog notebooka
- movie_data.csv
  - Podaci o filmovima dobijeni nakon izvršenja petog notebooka

## Docker compose

Postavka za pokretanje Milvus baze.
