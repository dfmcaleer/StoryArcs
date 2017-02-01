import imdb
import pandas as pd

def get_year(name):

    ia = imdb.IMDb()

    s_result = ia.search_movie(name)

    if len(s_result) != 0:

        first_res = s_result[0]
        ia.update(first_res)

        if 'year' in first_res.keys():
            Year = first_res['year']
            #movie_id = first_res.movieID
            print "getting year data for {}".format(name)

            return Year
        else:
            pass

    else:
        pass

def get_rating(name):

    ia = imdb.IMDb()

    s_result = ia.search_movie(name)

    if len(s_result) != 0:

        first_res = s_result[0]
        ia.update(first_res)

        if 'rating' in first_res.keys():
            Rating = first_res['rating']
            #movie_id = first_res.movieID
            print "getting rating data for {}".format(name)

            return Rating
        else:
            pass

    else:
        pass

def get_mpaa(name):

    ia = imdb.IMDb()

    s_result = ia.search_movie(name)

    if len(s_result) != 0:

        first_res = s_result[0]
        ia.update(first_res)

        if 'mpaa' in first_res.keys():
            Mpaa = first_res['mpaa']
            #movie_id = first_res.movieID
            print "getting mpaa data for {}".format(name)

            return Mpaa
        else:
            pass

    else:
        pass

if __name__ == '__main__':

    titles = pd.read_csv('titles.csv')

    df = pd.DataFrame()
    df['title'] = titles

    #df['year'] = df['title'].apply(get_year)
    df['rating'] = df['title'].apply(get_rating)
    df['mpaa'] = df['title'].apply(get_mpaa)

    print "finished general data"
    df.to_csv('movie_other.csv',index=False)
