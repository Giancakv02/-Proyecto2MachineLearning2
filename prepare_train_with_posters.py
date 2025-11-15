import os
import re
import pandas as pd
import requests
from tqdm import tqdm

# ========= CONFIG ==========
TMDB_API_KEY = "ba9c5b92700845ff51e7256c39a19f41"  # TMDb API key
INPUT_CSV = "data/movies_train.csv"
OUTPUT_CSV = "data/movies_train_with_posters.csv"
POSTER_DIR = "data/posters"

os.makedirs(POSTER_DIR, exist_ok=True)


def extract_title_and_year(title_raw):
    """Extrae el título limpio y el año de un string como 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)\s*$", title_raw)
    if match:
        year = int(match.group(1))
        title_clean = title_raw[:match.start()].strip()
    else:
        year = None
        title_clean = title_raw.strip()
    return title_clean, year


def tmdb_search_movie(title, year=None):
    """Busca una película por título (y opcionalmente año) en TMDb."""
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }
    if year:
        params["year"] = year

    r = requests.get(base_url, params=params)
    if r.status_code != 200:
        return None
    data = r.json()
    results = data.get("results", [])
    if not results:
        return None
    return results[0].get("poster_path", None)


def tmdb_download_poster(poster_path, save_path):
    """Descarga el póster desde TMDb y lo guarda como JPG."""
    if not poster_path:
        return False
    img_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
    r = requests.get(img_url, stream=True)
    if r.status_code != 200:
        return False
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(4096):
            f.write(chunk)
    return True


def main():
    df = pd.read_csv(INPUT_CSV)
    df["poster_path"] = ""

    print(f"[1] Buscando y descargando pósters de {len(df)} películas...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        movie_id = row["movieId"]
        title_raw = row["title"]

        title_clean, year = extract_title_and_year(title_raw)
        local_path = os.path.join(POSTER_DIR, f"{movie_id}.jpg")

        if os.path.exists(local_path):
            df.loc[i, "poster_path"] = local_path
            continue

        poster_path_tmdb = tmdb_search_movie(title_clean, year)
        if not poster_path_tmdb:
            continue

        success = tmdb_download_poster(poster_path_tmdb, local_path)
        if success:
            df.loc[i, "poster_path"] = local_path

    df_with_poster = df[df["poster_path"] != ""]
    df_with_poster.to_csv(OUTPUT_CSV, index=False)
    print(f"[✔] Listo. Se guardaron {len(df_with_poster)} películas con póster válido en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()