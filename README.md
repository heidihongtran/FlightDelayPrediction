# Flight Delay Prediction
This is a group work in my post-graduate program.

Project requirements: 

The file FlightDelays.csv contains information on all commercial flights that departed the Washington, D.C., area and arrived in New York in January 2004. For each flight, there is information on the departure and arrival airports, the distance of the route, the scheduled time and date of the flight, and so on. The variable that we are trying to predict is whether a flight is delayed. A delay is defined as an arrival that is at least 15 minutes later than scheduled.

METHODS:

Flight delays have serious impacts on both the airline industry and consumers alike. Thus, classifying whether or not a flight will be delayed is critical to the industry. To do this, multiple models were created with the intention of selecting the most accurate one. The contents of all models were built from a dataset originally containing over 12 predictor variables. However, most of the predictors were categorical, they were converted into numeric form as needed to comply with the requirements of the various models tested, including Logistic Regression, Naïve Bayes, and Decision Tree. Furthermore, to ensure that the model is lightweight and efficient in terms of processing power, some predictor variables were removed based on a score calculated by running a Feature Importance technique. To elaborate further, all predictor variables assigned a score of zero from feature importance were removed from the model. For the airline industry, this model can be applied as soon as a flight departs. It is a requirement for the model to be run after the flight departs as departure time is a variable used in the model. Nevertheless, once a flight departs the model can
be implemented to predict if the flight will be delayed or not. In the case where the flight will be delayed, measures can be taken at the destination to ensure a seamless transition in terms of logistics. In addition to this, in-flight adjustments can be made to reduce the travel time preventing the actual flight from being
delayed. Thus, this model would be critical in the development of contingency plans preventing or reducing the impact delayed flights have on operational efficiencies.
