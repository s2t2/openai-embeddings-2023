





# OpenAI Embeddings (v2)

This supercedes earlier approach to fetch embeddings. In this second attempt we are grabbing user-level as well as tweet-level embeddings, to compare the difference in these approaches.

The "Exporting Embeddings" notebook takes embeddings stored in BigQuery (see app/openai_embeddings_v2/README.md), and exports them to CSV / parquet files on Google Drive for easier and cheaper access

The "Analysis Template" notebook provides an example of how to load the files from drive for further analysis.
