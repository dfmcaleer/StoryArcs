import urllib
from bs4 import BeautifulSoup
import imdb
import pandas as pd
import csv

def get_movie_gross(movie_name):
    ia = imdb.IMDb()

    s_result = ia.search_movie(movie_name)

    if len(s_result) != 0:

        first_res = s_result[0]
        ia.update(first_res)
        movie_id = first_res.movieID

        page = urllib.urlopen('http://www.imdb.com/title/tt{}/?ref_=fn_al_nm_1a'.format(movie_id))
        soup = BeautifulSoup(page.read(), "lxml")
        for h4 in soup.find_all('h4'):
            if "Gross:" in h4:
                print "getting gross for {}".format(movie_name)
                return h4.next_sibling.strip()
            else:
                pass
    else:
        pass

titles = pd.read_csv('titles.csv')
df = pd.DataFrame()
df['title'] = titles

print "at line 25"

df['gross'] = df['title'].apply(get_movie_gross)

print "finished gross"

df.to_csv('movie_gross.csv',index=False)
