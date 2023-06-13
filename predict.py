import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

#Definimos función para cargar el modelo
def load_model():
    with open ('modelo_lda.pkl', 'rb') as file:
        data = pkl.load(file)
    return data

data = load_model()

model_loaded = data["model"]
le_agente = data["le_agente"]
le_rolagente = data["le_rolagente"]
le_mapa = data["le_mapa"]
le_premade = data["le_premade"]

#En esta función definimos los parámetros que necesitamos rellenar en la app, junto con el diseño de la página
def show_predict():

    st.set_page_config(
        page_title="App Web Valorant",
        layout="centered",
        page_icon="random")
    
    #st.image("header.png")

    st.header("Completa la información:")

    st.info("1) Información general:")

    #st.markdown(f'<h1 style="color:#a6a6a6;font-size:18px;">{"Complete la siguiente información para realizar la predicción:"}</h1>', unsafe_allow_html=True)

    agente = (
        "fade",
        "omen",
        "sage",
        "chamber",
        "killjoy",
        "breach",
        "cypher",
        "viper",
    )

    rolagente = (
        "iniciador",
        "controlador",
        "centinela",
    )

    mapa = (
        "haven",
        "lotus",
        "pearl",
        "split",
        "ascent",
        "fracture",
        "icebox",
    )

    premade = (
        "duoq",
        "soloq",
        "team",
        "trioq",
    )

    agentes = st.selectbox("Agente:", agente)

    rolagentes = st.selectbox("Rol Agente:", rolagente)

    mapas = st.selectbox("Mapa:", mapa)

    st.info("2) Composición de equipo:")

    duelistas = st.slider("Duelistas:", 0, 5, 1)

    iniciadores = st.slider("Iniciadores:", 0, 5, 1)

    centinelas = st.slider("Centinelas:", 0, 5, 1)

    controladores = st.slider("Controladores:", 0, 5, 1)

    premades = st.selectbox("Premade:", premade)

    ok = st.button("Predecir")
    if ok:
        X = np.array([[agentes, rolagentes, mapas, duelistas, iniciadores, centinelas, controladores, premades]])
        X[:, 0] = le_agente.transform(X[:, 0])
        X[:, 1] = le_rolagente.transform(X[:, 1])
        X[:, 2] = le_mapa.transform(X[:, 2])
        X[:, 7] = le_premade.transform(X[:, 7])
        X = X.astype(float)

        prediccion = model_loaded.predict(X)
        if prediccion==1:
            st.header("VICTORIA!")
        else:
            st.header("DERROTA")

    #Otros parámetros de diseño, aquí definimos la fuente a utilizar
    st.write("""
    <style>
    
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300&display=swap');html, body, [class*="css"]  {
        font-family: 'Rajdhani', sans-serif;
    }
    
    </style>""", unsafe_allow_html=True)

    st.markdown(f'<h1 style="color:#dacb8d;font-size:20px;text-align:center">{"https://github.com/mckrena - http://linkedin.com/in/mckrena"}</h1>', unsafe_allow_html=True)

    #Agregamos background
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.ibb.co/wg0bfh2/Background-1.png");
             background-attachment: fixed;
             background-size: cover
         }}

         </style>
         """,
         unsafe_allow_html=True
     )