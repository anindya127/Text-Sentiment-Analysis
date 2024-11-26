import os
import requests
import zipfile

def download_glove_embeddings(embedding_dim=100):
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = "glove.6B.zip"
    extract_path = "."

    if not os.path.exists(f"glove.6B.{embedding_dim}d.txt"):
        print(f"Downloading GloVe embeddings...")
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)

        print("Extracting GloVe embeddings...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract(f"glove.6B.{embedding_dim}d.txt", extract_path)

        os.remove(zip_path)
        print(f"GloVe embeddings (dimension: {embedding_dim}) downloaded and extracted successfully.")
    else:
        print(f"GloVe embeddings file already exists.")

if __name__ == "__main__":
    download_glove_embeddings()

