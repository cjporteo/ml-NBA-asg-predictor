# NBA All-Star Predictor

## 🧐 Link to the code (Jupyter Notebook)

[https://github.com/cjporteo/ml-NBA-asg-predictor/blob/master/model_ASG.ipynb](https://github.com/cjporteo/ml-NBA-asg-predictor/blob/master/model_ASG.ipynb)

## What is this project about?

As a lifelong basketball fan, I always wondered what exactly goes into being selected as an NBA All-Star - why do certain players get picked and others get glossed over? Which aspects of their performance are most valued by fans and media? What makes a star a star?

Over the years, I’ve formed my own personal opinions on these debates, but they were just that - opinions. I wasn’t quite satisfied with this, so that’s when I decided to go digging for the relevant data and quantitatively explore this problem. My goal was to create a machine learning model that could accurately classify players as All-Stars or non-All-Stars, then put the model under the microscope to extract useful insights and quell some of my curiosity.

A thorough 5000+ word write-up for this project (**links at bottom of page**) can be found on [Medium](https://medium.com/@cjporteo) as a two-part series, published in Towards Data Science.

![](https://i.imgur.com/E0YvkTu.png)

![](https://i.imgur.com/RaEvrQM.png)

This project uses a custom web-scraping utility to construct a usable dataset and and uses gradient boosted tree modelling (XGBoost) to capture relationships between player statistics and their All-Star selection decision. I also applied model exlainability and interpretation techniques to shine a quantitative light on the real story going on behind this problem.

I’m proud of this project because of the effort I had to put in to collect and process the data. The starting point for many data science projects is a nice pre-assembled dataset, and the engineering aspects of the data science workflow are often neglected. With this project, I started out with nothing but curiosity and had to employ a wide range of problem solving and data science techniques to take this problem from end-to-end.

## Findings

This model correctly predicted 22/24 All-Stars for the 2020 NBA All-Star Game. Here are the predictions:

### East

![](https://i.imgur.com/7L5Zpwu.png)

### West

![](https://i.imgur.com/oUVN2Vm.png)

## Article Links - Towards Data Science

Even if you aren't a Medium subscriber, you can use these links to read the complete article.

[Using machine learning to predict NBA All-Stars, Part 1: Data collection](https://medium.com/@cjporteo/using-machine-learning-to-predict-nba-all-stars-part-1-data-collection-9fb94d386530?source=friends_link&sk=a96c9598bd868f16f508e75c6dff3158)


[Using machine learning to predict NBA All-Stars, Part 2: Modelling](https://medium.com/@cjporteo/using-machine-learning-to-predict-nba-all-stars-part-2-modelling-a66e6b534998?source=friends_link&sk=98afe5974104d088d4d3c99e0d305a38)
