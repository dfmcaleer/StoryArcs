import imdb
import pandas as pd


def scrape_data(name):
    ia = imdb.IMDb()

    s_result = ia.search_movie(name)

    if len(s_result) != 0:

        first_res = s_result[0]
        ia.update(first_res)

        if 'runtimes' in first_res.keys():
            Runtime = first_res[('runtimes')][0]
            print "getting runtime data for {}".format(name)

        if 'country codes' in first_res.keys():
            CountryCode = first_res['country codes'][0]
            print "getting country code data for {}".format(name)

        # if 'plot' in first_res.keys():
        #     Plot = first_res['plot']
        #     print "getting plot data for {}".format(name)

        if Runtime and CountryCode:
            return Runtime, CountryCode#, Plot

if __name__ == '__main__':

    titles = pd.read_csv('titles.csv')
    titles = titles[0:30]

    df = pd.DataFrame()
    df['title'] = titles

    new_vars = df['title'].apply(scrape_data).apply(pd.Series)
    new_vars.columns = ['runtime', 'country']
    df = pd.concat([df, new_vars], axis=1)
    print df

    print "finished general data"
    #df.to_csv('movie_other2.csv',index=False)
