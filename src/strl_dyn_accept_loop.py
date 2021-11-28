import streamlit as st
import matplotlib.pyplot as plt
from os import listdir, path

import dyn_accept_loop as dal

import lightgbm as lgbm
from sklearn.tree import DecisionTreeClassifier

datapath = 'C:/Projects/Information-Systems-Seminar/prepared_data/'

st.title('Simulate Dynamic Acceptance Process')
st.write('Seminar Information Systems - Group C1 - *Bias-Removing Autoencoder for Reject Inference*')


col1, col2 = st.columns(2)

datasetname = col1.selectbox('Dataset:', [f  for f in listdir(datapath) if path.isfile(path.join(datapath, f)) and f.endswith('.csv')])

modeltype = col2.selectbox('Model', ('LGBMClassifier', 'DecisionTreeClassifier'))
if modeltype == 'LGBMClassifier':
    model = lgbm.LGBMClassifier()
elif modeltype == 'DecisionTreeClassifier':
    model = DecisionTreeClassifier()

initial_trainsplit = st.slider('Split for initial model fitting', 0.01, 1.0, value=0.1)

validationsplit = st.slider('Split for model testing', 0.00001, 1.0, value=0.1)

n_years = st.number_input('Number of iterations to simulate', 2, 70, value=20)

start_btn = st.button('Start Simulation...')

st.markdown('---')

if start_btn:

    with st.spinner(text='Loading Data...'):
        sim = dal.Simulate_acceptance_loop(datasetname, model, 0.001, validationsplit, n_years)
        st.success('Data loaded and split!')

    plot_element = st.empty()
    progress_element = st.empty()
    text_element = st.empty()

    sim_iteration = sim.run()

    metrics = None

    for iteration in sim_iteration:
        metrics = iteration[3]

        x = range(0, iteration[0] + 1)
        fig, ax = plt.subplots()
        ax.plot(x, metrics["model"]["holdout"]['roc_auc'], label = 'roc_auc-model')
        ax.plot(x, metrics["oracle"]["holdout"]['roc_auc'], label = 'roc_auc-oracle')
        ax.legend(loc = 'lower right')
        plt.ylim([0.5, 1.0])
        plot_element.pyplot(fig)
        plt.close()

        progress_element.progress((iteration[0]/(n_years - 1)))
        text_element.write(f'Itteration: {iteration[0]}) Accepted: {iteration[1].count(True)} | Denied: {iteration[1].count(False)} - New train set size: {iteration[2]}')
    
    st.balloons()