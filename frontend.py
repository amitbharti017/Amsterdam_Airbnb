import gradio as gd
import joblib
from logger import logger
from pathlib import Path
import numpy as np
import pandas as pd


# Creating a function by which Gradio will present the predicted price

# <!-- def estimate(Brand, Product, ProcessorType, ProcessorCore,
#                       ProcessorGen, RAM, Opsys, HDD_Storage,
#                       SSD_Storage, Display, Warranty, Rating):
    
#         model=joblib.load("LaptopPriceModel2.sav")

#         outcome = model.predict([[Brand, Product, ProcessorType, ProcessorCore,
#                               ProcessorGen, RAM, Opsys, HDD_Storage,
#                               SSD_Storage, Display, Warranty, Rating]])
#         return outcome
#  -->

def predict_function(host_since, host_is_superhost,host_listings_count,accommodates, private,bathrooms,beds, number_of_reviews,review_scores_rating,availability_365,minimum_nights,room_type,neighbourhood_cleansed):
    model = joblib.load("artifacts/model_selection/best_model.pkl")
    preprocessor = joblib.load(Path("artifacts/data_transformation/data_transformer.joblib"))
    #Since the distribution of minimum_nights has long tale so all the values greater then 30 are set to 31 same as preprocessor
    minimum_nights = 31 if (minimum_nights > 30) else minimum_nights
    availability_365 = 365 if (availability_365 > 365) else availability_365

    ## Applying log to the numeric columns which have skewed distribution:
    numeric_skewed_columns = [host_since, number_of_reviews,availability_365]
    for col in numeric_skewed_columns:
        col = np.log1p(col)
    df_test = pd.DataFrame.from_dict(
            {
        "host_since" : [host_since],
        "host_is_superhost" :[1 if host_is_superhost else 0],
        "host_listings_count" : [int(host_listings_count)],
        "accommodates" : [accommodates],
        "private" : [1 if private else 0],
        "bathrooms" : [bathrooms],
        "beds" : [beds],
        "number_of_reviews" : [number_of_reviews],
        "review_scores_rating" : [review_scores_rating],
        "availability_365" : [availability_365],
        "minimum_nights" : [minimum_nights],
        "room_type": [room_type],
        "neighbourhood_cleansed": [neighbourhood_cleansed]
            }
    )
    logger.info("Input data is %s",df_test)

    df_test = preprocessor(df_test)
    prediction = model.predict(df_test)
    return prediction





# Creating inputs for Gradio iterface

host_since = gd.Number(label = "Select the number of days from which you want host to be listed in AirBnb")
host_is_superhost = gd.Checkbox(label = "Super Host",info="Room is listed by super host or not")
host_listings_count = gd.Slider(1,25,value=1,label = "Host listing count",info="Number of listing of the host in Airbnb Amsterdam")
neighbourhood_cleansed = gd.Dropdown(['ijburg_-_zeeburgereiland', 'centrum-oost',
       'de_baarsjes_-_oud-west', 'oud-oost', 'centrum-west',
       'bos_en_lommer', 'gaasperdam_-_driemond',
       'de_pijp_-_rivierenbuurt', 'noord-west', 'zuid', 'oud-noord',
       'bijlmer-centrum', 'westerpark', 'slotervaart',
       'de_aker_-_nieuw_sloten', 'watergraafsmeer',
       'geuzenveld_-_slotermeer',
       'oostelijk_havengebied_-_indische_buurt', 'buitenveldert_-_zuidas',
       'noord-oost', 'bijlmer-oost', 'osdorp'],label = "Neighbourhood",info ="Area where booking is required")
accommodates = gd.Number(label= "Number of person")
bathrooms = gd.Number(label = "Number of bathrooms")
private_bathroom = gd.Checkbox(label = "Private Bathroom",info="if customer wants private bathroom, else shared bathroom will be considered")
beds = gd.Number(label = "Beds required", info ="Number of beds required")
number_of_reviews = gd.Number(label = "Number of reviews")
availability_365 = gd.Number(label = "availability", info ="Number of days the housing can be listed in Airbnb in a year")
minimum_nights = gd.Number(label = "Minimum nights", info ="Lowest number of days housing could be booked for")
room_type = gd.Dropdown(['entire_home/apt', 'private_room', 'hotel_room', 'shared_room'],label = "Select the room type")
review_scores_rating = gd.Slider(1,5,value=4.5,label = "Rating")


# Defining the output
Price = gd.Number()
# setting up the web app entries

webapp = gd.Interface(fn = predict_function, inputs = [host_since, host_is_superhost,
                                            host_listings_count, 
                                            neighbourhood_cleansed,
                                            room_type, 
                                            accommodates, bathrooms, 
                                            private_bathroom, beds, 
                                            number_of_reviews, availability_365, 
                                            minimum_nights,review_scores_rating], outputs = Price)

# Deploying the model
webapp.launch(share = 'True')