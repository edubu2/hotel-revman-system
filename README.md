# Building a Hotel Revenue Management (Rate Recommender) System from Scratch

(IN PROGRESS)

By Elliot Wilens, Data Scientist and Hotel Revenue Manager

Metis Data Science Bootcamp | Project 5: Passion Project

Project Duration: 3 weeks

Blog post here: 

___
## Introduction

I was a Hotel Revenue Manager with Marriott International for five years, and was always curious about the inner-workings of the revenue management systems employed there. The COVID-19 pandemic provided me the opportunity to delve into programming and Data Science. Fast forward one year, and I'm now approaching the end of the 12-week Metis Data Science Bootcamp, and our fifth (and final) project is the 'Passion Project'. I wanted to combine my newly acquired knowledge with the old into the creation of a revenue management system.

Since the results of this project had to be be made public, I feared I wouldn't be able to find a dataset. I had almost selected a different problem to work on when I found a fully anonymous dataset containing reservation-level hotel data for two European hotels.
___
## The Dataset

The dataset contains a sample of reservations for both H1 (Hotel 1, resort hotel) and H2 (Hotel 2, city hotel). The sample contained all reservations that touched the hotel nights of July 1, 2015 - August 31, 2017, including cancelled reservations. The hotel and guest information were anonymized, and the only contextual hotel information we have is the hotel type (resort/city).

There were 40,060 reservations for H1 and 79,330 reservations for H2. For more information about the dataset, see the source website ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S2352340918315191)).

___
## The Recommender System

For this to work, I had to choose a cutoff date that separated training data from testing data.

**For the purpose of this project, we assume that today is August 1, 2017, and the user is a Revenue Manager responsible for managing available room inventory (how many rooms are selling online for each future arrival date), and the selling price of the rooms on each night for the remainder of August.**

This allows our model one year of training data (July 2016 - Jul 2017). What happened to the year before that? Well, **pace** data is important for hotels. **Pace** is how many rooms are on the books (OTB) for a future date, compared to how many rooms we had OTB same-time-last-year (STLY) for the same date (adjusted for day of week). 

Due to the nature of the dataset I'm working with, and the fact that all reservations are all historical (2015-2017), there are several key features that I won't be able to include in my models:
* Pricing information
  * We have average daily rate (ADR) for rooms sold, but not historical selling prices.
* 'Walked' guest coding
  * We don't know if the hotels walked any guests in our time frame. If they did, we don't know if these were coded as a cancellation or no-show.
* Competitor pricing (from shops)
  * In practice, with a market-driven pricing model, I would measure our hotel's change in demand in relation to competitor pricing changes, and make rate recommendations based largely off that. Since we don't have this data, we don't know the price threshold (if any) that will price us out of the market. 
* Reservation changes
  * Each reservation in the dataset represents the reservation's snapshot at the latest stage (for example, at the time of cancellation, or checkout).
  * If a reservation is shortened during check-in, it should register as a cancellation for the nights reduced (that's not the case here)
* Special events (e.g. 'annual holiday parade')
___
## Tech Stack


___
## Navigating this Repository

* Highlight `demand_model_selection.ipynb`

___
## Modeling


___
## Model Results
