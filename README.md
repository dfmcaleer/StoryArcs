# StoryArcs
## Galvanize Data Science Capstone Project
## Demetria McAleer

### Story shapes

+ The emotional ups and downs of every story form its shape.

+ We can plot these shapes using sentiment analysis, specifically polarity (a measure of how positive or negative the words are in a given window of the story).

+ The purpose of this project is to see what we can learn about how stories work by analyzing their shapes.

![alt text](https://github.com/dfmcaleer/StoryArcs/blob/master/plots/princess-bride.png)

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

+ Because I used unsupervised learning to form the clusters, I was worried that they weren't "real" - that is, I was only seeing this shapes because I had told the algorithm to cluster, rather than because all the stories in each cluster actually had something in common.

+ So, I started trying to tie the clusters to observable characteristics of the movies.

+ Results were... mixed.

+ Two of the clusters had a statistically significant relationship with box office revenues, two didn't.

+ There were patterns in which clusters tended to have more of each genre (for example, many romances follow the yellow shape).  But overall, a chi-squared test on genre and clusters was only marginally statistically significant.

+ Overall, results were not as strong as I had hoped for.  Time to keep thinking!

### What are story shapes telling us?

+ Stories tend to follow up-and-down patterns.  We do not see any movies that are flat, or even always trending up or always trending down.  Every movie has a mix.

+ But the shapes of these patterns are largely unrelated to popularity (as measured by box office) or content (as measured by genre).

+ However, sentiment in one part of the story does seem to be correlated with sentiment in other parts of the story.  What else might we be able to do with that?

### Ending prediction!

+ For this part of the analysis, I remove the ending of the story (approximately the final 20%), and use the pattern of sentiment change (i.e. story shape) in the first 80% to predict the ending.

+ The dependent variable is binary: happy (1) or sad (0).

+ Again, it looks at change in sentiment at the end rather than absolute sentiment.  (That is, a movie can be fairly bleak, but if it takes an upturn at the end, it counts as a happy ending.)

+ Classes are relatively balanced (about 55% of movies turn upward at the end).

+ Tried both Random Forest and Gradient Boosting model, but performance was similar.  Final analysis refers to Random Forest.

### Results

![alt tag](https://github.com/dfmcaleer/StoryArcs/blob/master/plots/ending-prediction.png)

+ Baseline models that either always predict a happy ending or predict a happy ending 55% of the time have relatively poor performance in both accuracy and F1 score.

+ Adding other features of the movie such as box office, genre, and release year improve model performance to accuracy in the high 50s and F1 in the low 60s.

+ Finally, adding sentiment from the first 80% of the movie bring both accuracy and F1 up to 70%.

+ I think this is a cool result!  It means that we don't have to know anything about the substantive content of a story - just the story's mood - to make a pretty good prediction about how the story is going to end.
