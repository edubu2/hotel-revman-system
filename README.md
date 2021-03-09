# Building a Hotel Revenue Management (Rate Recommender) System from Scratch

(IN PROGRESS)

By Elliot Wilens, Data Scientist and Hotel Revenue Manager

Metis Data Science Bootcamp | Project 5: Passion Project

Project Duration: 3 weeks

Blog post here: 

___
## Introduction

I was a Hotel Revenue Manager with Marriott for five years, and was always curious about the inner-workings of the revenue management systems employed there. The COVID-19 pandemic provided me the opportunity to delve into programming and Data Science. Fast forward one year, and I'm now approaching the end of the 12-week Metis Data Science Bootcamp, and our fifth (and final) project is the 'Passion Project'. I wanted to combine my newly acquired knowledge with the old into the creation of a revenue management system.

Since the results of this project had to be be made public, I feared I wouldn't be able to find a dataset. I had almost selected a different problem to work on when I found a fully anonymous dataset containing reservation-level hotel data for two European hotels.
___
## The Dataset

The dataset contains a sample of reservations for both H1 (Hotel 1, resort hotel) and H2 (Hotel 2, city hotel). The sample contained all reservations that touched the hotel nights of July 1, 2015 - August 31, 2017, including cancelled reservations. The hotel and guest information were anonymized, and the only contextual hotel information we have is the hotel type (resort/city).

There were 40,060 reservations for H1 and 79,330 reservations for H2. For more information about the dataset, see the source website ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S2352340918315191)).

___
## The Recommender System

For this to work, I had to choose a cutoff date that separated training data from testing data.

**For the purpose of this project, we assume that today is August 1, 2017, and the user is a Revenue Manager responsible for managing available room inventory (how many rooms are selling online for each future arrival date), and the selling price of the rooms on each night for the remainder of August.**

This allows our model one year of training data (July 2016 - Jul 2017). What happened to the year before that? Well, **pace** data is important for hotels. **Pace** is how many rooms are on the books (OTB) for a future date, compared to how many rooms we had OTB same-time-last-year (STLY) for the same date (adjusted for day of week). I assumed this was important to have, so I could train the model to 'understand,' if you will, whether certain spikes in demand recur annually, or was a one-off occurrence (i.e. an annual holiday parade, or a FIFA World Cup soccer game). If there's no unusual spike in demand, pace can be a good indicator of a necessary price adjustment.

___
## Tech Stack


___
## Navigating this Repository


___
## Modeling


___
## Model Results
