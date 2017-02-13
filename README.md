# StoryArcs
## Galvanize Data Science Capstone Project
## Demetria McAleer

### Story shapes

+ The emotional ups and downs of every story form its shape.

+ We can plot these shapes using sentiment analysis, specifically polarity (a measure of how positive or negative the words are in a given window of the story).

+ The purpose of this project is to see what we can learn about how stories work by analyzing their shapes.

![alt text](https://github.com/dfmcaleer/StoryArcs/blob/master/plots/princess-bride.png =250x250)

### Creating story shapes

+ I web-scraped about 1100 movie scripts from the Internet Movie Script Database.

+ Then, I split each script into equal sized windows, and for each window, took the mean polarity score using TextBlob.

+ Now, each movie is represented at a vector of the sentiments at each point in the movie.

+ When you plot all the movie scripts together, you get... story shape spaghetti, like so:

![alt text](https://github.com/dfmcaleer/StoryArcs/blob/master/plots/all-the-movies.png "Madness!")

### Clusters

+ My next step was to use K-means clustering to group common story shapes together.  But a problem immediately emerged...

### Change in sentiment vs. absolute sentiment

+ Because some movies just are happier or sadder than others, movies wind up clustered on their overall average sentiment rather than the actual shape of the story.

+ For this reason, at this point I switch to representing each story as a vector of the *differences* between the sentiments at each point, rather than the absolute sentiment.

### Clusters, but actually working this time

+ Using K-means, four distinct clusters of movies occur.

+ Why four clusters, you might ask?  Whenever I chose a number higher than four, the clusters would overlay each other - essentially, wanting to form four clusters even when I tried higher numbers.

![alt tag](https://github.com/dfmcaleer/StoryArcs/blob/master/plots/four-clusters.png)

+ It's fun to see which movies fall into each cluster!  For example, the green cluster that goes sad-happy-sad is *Star Wars: The Empire Strikes Back*, and the yellow cluster that has some ups and downs but ends happy is *Star Wars: Return of the Jedi*.

### But what do the clusters mean?
