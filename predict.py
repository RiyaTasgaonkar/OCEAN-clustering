import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

model = pickle.load(open("Data/model.pkl", 'rb')) 
dataclusters = pd.read_csv('Data/clusters.csv')

questions = [
    "I am the life of the party.",
    "I don't talk a lot.",
    "I feel comfortable around people.",
    "I keep in the background.",
    "I start conversations.",
    "I have little to say.",
    "I talk to a lot of different people at parties.",
    "I don't like to draw attention to myself.",
    "I don't mind being the center of attention.",
    "I am quiet around strangers.",
    "I get stressed out easily.",
    "I am relaxed most of the time.",
    "I worry about things.",
    "I seldom feel blue.",
    "I am easily disturbed.",
    "I get upset easily.",
    "I change my mood a lot.",
    "I have frequent mood swings.",
    "I get irritated easily.",
    "I often feel blue.",
    "I feel little concern for others.",
    "I am interested in people.",
    "I insult people.",
    "I sympathize with others' feelings.",
    "I am not interested in other people's problems.",
    "I have a soft heart.",
    "I am not really interested in others.",
    "I take time out for others.",
    "I feel others' emotions.",
    "I make people feel at ease.",
    "I am always prepared.",
    "I leave my belongings around.",
    "I pay attention to details.",
    "I make a mess of things.",
    "I get chores done right away.",
    "I often forget to put things back in their proper place.",
    "I like order.",
    "I shirk my duties.",
    "I follow a schedule.",
    "I am exacting in my work.",
    "I have a rich vocabulary.",
    "I have difficulty understanding abstract ideas.",
    "I have a vivid imagination.",
    "I am not interested in abstract ideas.",
    "I have excellent ideas.",
    "I do not have a good imagination.",
    "I am quick to understand things.",
    "I use difficult words.",
    "I spend time reflecting on things.",
    "I am full of ideas.",
]

answers = np.zeros(50)

def generate_questions(questions, position, answers):
    c1 = ['1', '2', '3', '4', '5']
    left, right = st.beta_columns((6,2))
    left.markdown(questions[position])
    answers[position] = right.slider('', min_value = 1, max_value = 5, value = 1, step = 1, key = position)

def fill_progress(bar, completion):
    for percent_complete in range(completion):
        time.sleep(0.001)
        bar.progress(percent_complete + 1)

st.title('OCEAN Model')
st.markdown('')
st.markdown('')
st.markdown('Personlaity trait score prediction using clustering.')
st.markdown('Please answer all these questions to predict your scores for each trait')
st.markdown('1 - Disagree, 3 - Neutral, 5 - Agree')
st.markdown('')
st.markdown(' ')

for i in range(len(questions)):
    generate_questions(questions, i, answers)
submit = st.button('Submit')

if submit:
    answers = np.array(answers).reshape(1,-1)
    cluster = model.predict(answers)
    scores = np.array(dataclusters.iloc[cluster,1:6], dtype = 'int32')
    o = 'Openness - ' + str(scores[0][4]) + ' %'
    c = 'Conscientiousness - ' + str(scores[0][3]) + ' %'
    e = 'Extraversion - ' + str(scores[0][0]) + ' %'
    a = 'Agreableness - ' + str(scores[0][2]) + ' %'
    n = 'Neuroticism - ' + str(scores[0][1]) + ' %'
    
    st.markdown(o)
    o_bar = st.progress(0)
    fill_progress(o_bar, scores[0][4])
    st.markdown(c)
    c_bar = st.progress(0)
    fill_progress(c_bar, scores[0][3])
    st.markdown(e)
    e_bar = st.progress(0)
    fill_progress(e_bar, scores[0][0])
    st.markdown(a)
    a_bar = st.progress(0)
    fill_progress(a_bar, scores[0][2])
    st.markdown(n)
    n_bar = st.progress(0)
    fill_progress(n_bar, scores[0][1])
    
    
