





# OpenAI Embeddings (v2)

This supercedes earlier approach to fetch embeddings. In this second attempt we are grabbing user-level as well as tweet-level embeddings, to compare the difference in these approaches.

 1. The "Exporting Embeddings" notebook takes embeddings stored in BigQuery (see app/openai_embeddings_v2/README.md), and exports them to CSV / parquet files on Google Drive for easier and cheaper access


  2. The "De duping and Averaging" notebook de-duplicates status embeddings, and also calculates average tweet-level embeddings per user, and saves these CSV files to drive.


  3. The "Analysis Template" notebook provides an example of how to load the files from drive for further analysis.

  4. The "User vs Tweet Level Embeddings" notebook performs dimensionality reduction on user embeddings vs tweet embeddings averaged for each user. The results are saved to drive, and then copied to the "results/openai_embeddings_v2" folder in this repo.
