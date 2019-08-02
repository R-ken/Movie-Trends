import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# resource for cleaning: https://www.mikulskibartosz.name/how-to-split-a-list-inside-a-dataframe-cell-into-rows-in-pandas/


def main():
    wikidata = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True, encoding="utf8")
    genres = pd.read_json('genres.json.gz', orient='record', lines=True, encoding="utf8")
    genre_data = wikidata.genre.apply(pd.Series)
    wikidata = wikidata.merge(genre_data, left_index = True, right_index = True)
    wikidata = wikidata.drop(["genre"], axis=1)
    wikidata = wikidata.melt(id_vars = ['enwiki_title', 'wikidata_id'], value_name = 'genre_id').drop('variable', axis=1).dropna()
    wikidata['genre_id'] = wikidata['genre_id'].astype(str)
    genres['wikidata_id'] = genres['wikidata_id'].astype(str)
    wikidata_with_genres = wikidata.merge(genres, left_on='genre_id', right_on='wikidata_id')
    wikidata_with_genres = wikidata_with_genres.drop(['wikidata_id_x', 'wikidata_id_y', 'genre_id'], axis=1)
    wikidata_with_genres = wikidata_with_genres.sort_values(by=['enwiki_title']).reset_index(drop=True)

    wikidata_with_genres.to_csv('titles-with-genres')

if __name__ == "__main__":
    main()