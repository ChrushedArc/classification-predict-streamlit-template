"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/count_vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification using supervised learning methods")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Prediction",'Visuals']
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("This application is designed to predict the sentiment of tweets from twitter data.  There are four different sentiment classes: Belief in Climate Change, Disbelief in Climate Change, Neither believes nor disbelieves, and News Articles about Climate Change")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.markdown('Below is the raw data')
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
			
	if selection == "Visuals":
		st.info("Visuals")
		st.markdown('This page shows a variety of visuals regarding the data')
		st.subheader("Visual Selection ")
		if st.checkbox('Show Distribution of Target Variable'):
			st.markdown('Below is the distribution plot of the sentiment score between the classes') 
			st.image("resources/newplot.PNG", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
		if st.checkbox('Show Positve Hashtags'):
			st.markdown('Below is the distribution plot of hasthags of that tweets show that believe in climate change') 
			st.image("resources/Hashtag Believe.PNG", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
		if st.checkbox('Show Negative Hashtags'):
			st.markdown('Below is the distribution plot of hasthags of that tweets show that believe in climate change') 
			st.image("resources/Hashtag Disb.PNG", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
		if st.checkbox('Show Neutral Hashtags'):
			st.markdown('Below is the distribution plot of hasthags of that tweets show that believe in climate change') 
			st.image("resources/Neutralht.PNG", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
		if st.checkbox('Show News Related Hashtags'):
			st.markdown('Below is the distribution plot of hasthags of that tweets show that believe in climate change') 
			st.image("resources/Newst.PNG", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
		if st.checkbox('Show Positve Twitter Handles'):
			st.markdown('Below is the distribution plot of Twitter Handles of that tweets show that believe in climate change') 
			st.image("resources/Handle Believe.PNG", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
		if st.checkbox('Show Negative Twitter Handles'):
			st.markdown('Below is the distribution plot of Twitter Handles of that tweets show that do not believe in climate change') 
			st.image("resources/Handle Disb.PNG", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
		if st.checkbox('Show Neutral Twitter Handles'):
			st.markdown('Below is the distribution plot of Twitter Handles of that tweets show that neither believe nor disbelieve in climate change') 
			st.image("resources/Neutralhd.PNG", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
		if st.checkbox('Show News Article Twitter Handles'):
			st.markdown('Below is the distribution plot of Twitter Handles of that tweets that are news articles regarding climate change') 
			st.image("resources/Newshd.PNG", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')


	
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		option = st.selectbox('Which model would you like to use?',('SVC', 'Logistic Regression','Extra Trees', 'Voting Classifier'))
		st.write('You selected:', option)
		if option == 'SVC':
			st.markdown('A support Vector Classifier attempts to seperate the responses using a hyperplane.  Any response above the plane is classified as A, and if it is below, it is classified as not A.')
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/svmmodel.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 1:
					st.success('This tweet signifies belief in climate change')
				elif prediction == -1:
					st.success('This tweet signifies disbelief in climate change')
				elif prediction == 0:
					st.success('This tweet neither signifies belief nor disbelief in climate change')
				elif prediction == 2:
					st.success('This tweet is a news article about climate change')
		if option == 'Logistic Regression':
			# Creating a text box for user input
			st.markdown('Logistic regression models the probability that this response belongs to a particular category/class. Logistic regression uses the sigmoid curve. Using a threshold , any observation falling above this threshold gets classified to class A')
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/logistic regression"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 1:
						st.success('This tweet signifies belief in climate change')
				elif prediction == -1:
						st.success('This tweet signifies disbelief in climate change')
				elif prediction == 0:
						st.success('This tweet neither signifies belief nor disbelief in climate change')
				elif prediction == 2:
					st.success('This tweet is a news article about climate change')
		if option == 'Extra Trees':
			# Creating a text box for user input
			st.markdown('A multitude of decision trees are built and the final output is the mode of the classes.')
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/extra trees"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 1:
						st.success('This tweet signifies belief in climate change')
				elif prediction == -1:
						st.success('This tweet signifies disbelief in climate change')
				elif prediction == 0:
						st.success('This tweet neither signifies belief nor disbelief in climate change')
				elif prediction == 2:
					st.success('This tweet is a news article about climate change')
		if option == 'Voting Classifier':
			# Creating a text box for user input
			st.markdown('A Voting Classifier is a machine learning model that that uses an ensemble of different classification methods, and then chooses the output based on the highest probability.  The model aggregates the findings of each model that gets passed into the voting classifier and it predicts the class based on a majority vote')
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Voting_Classifier"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 1:
						st.success('This tweet signifies belief in climate change')
				elif prediction == -1:
						st.success('This tweet signifies disbelief in climate change')
				elif prediction == 0:
						st.success('This tweet neither signifies belief nor disbelief in climate change')
				elif prediction == 2:
					st.success('This tweet is a news article about climate change')


			

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
