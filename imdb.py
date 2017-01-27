import imdb

ia = imdb.IMDb()

s_result = ia.search_movie('It Follows')
first_res = s_result[0]
ia.update(first_res)
print first_res['genre']
