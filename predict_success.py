import numpy as np
import pandas as pd
from sklearn import preprocessing

wikidata = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True, encoding="utf8")
rotten_tomato = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)


def main():
    wikidata = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True, encoding="utf8")
    rotten_tomato = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)
    # Here we will consider a movie good if it has a critic rating > 80% on rotten tomatoes.
    rotten_tomato = rotten_tomato[['rotten_tomatoes_id', 'critic_percent']]
    rotten_tomato['good'] = (rotten_tomato['critic_percent'] >= 80)
    rotten_tomato = rotten_tomato.drop(columns=['critic_percent'])
    rotten_tomato = rotten_tomato.set_index('rotten_tomatoes_id')
    wikidata_with_cast = wikidata[wikidata.cast_member.notna()]
    cast_members_by_movie = wikidata_with_cast[['cast_member', 'rotten_tomatoes_id']]

    # Sample a small number for testing
    # cast_members_by_movie = cast_members_by_movie.head(200)


    cast_members_by_movie = cast_members_by_movie.cast_member.apply(pd.Series)     .merge(cast_members_by_movie, left_index = True, right_index = True)     .drop(["cast_member"], axis = 1)     .melt(id_vars = ['rotten_tomatoes_id'], value_name = "cast_member")     .drop('variable', axis = 1)     .dropna()
    cast_members_by_movie = cast_members_by_movie.set_index('rotten_tomatoes_id')
    cast_members_by_movie_with_rating = rotten_tomato.join(cast_members_by_movie)
    cast_members_by_movie_with_rating = cast_members_by_movie_with_rating.dropna()
    # Using 26Gb of memory here - could convert to spark job
    categorical_rep_of_cast_in_movies = pd.get_dummies(cast_members_by_movie_with_rating['cast_member'])
    categorical_rep_of_cast_in_movies = categorical_rep_of_cast_in_movies.groupby('rotten_tomatoes_id').any().astype(int)
    categorical_rep_of_cast_in_movies.to_csv('categorized_actors.csv') 


if __name__ == "__main__":
    main()