import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar el modelo RandomForestRegressor
loaded_model = joblib.load('rforest.joblib')
#model2 = joblib.load('rforest_2.joblib')
# Obtener la versión de Joblib
joblib_version = joblib.__version__


# Nombres de las características
feature_names = ['Caracteristicas', 'Habitaciones', 'Aseos', 'Terraza',
                 'Piscina', 'Garaje', 'Metros', 'Barri', ' Poblacio_ocupada',
                 ' renda_mitjana_per_persona', 'PreuM2', 'preu_teoric']

# Interfaz de Streamlit
st.title("Predicció d'Immobles a Barcelona")
st.write("Versión de Joblib:", joblib_version)
st.write('Introdueix els valors de les característiques y prem el botó '
         'corresponent.')
caracteristicas_valores = ('Apartament', 'Àtic', 'Finca Rústica', 'Duplex',
                           'Pis a plantes intermitges', 'Planta baixa',
                           'Casa o Xalet', 'Loft', 'Casa Adosada', 'Estudi')
col1, col2 = st.columns(2, gap='large')
# Obtener los valores de entrada de las características
with col1:

    caracteristicas_valor = st.selectbox('Características',
                                         caracteristicas_valores)
    def convertir_caracteristiques(value):
        if value == 'Apartament':
            return 0
        elif value == 'Àtic':
            return 1
        elif value == 'Finca Rústica':
            return 2
        elif value == 'Duplex':
            return 3
        elif value == 'Pis a plantes intermitges':
            return 4
        elif value == 'Planta baixa':
            return 5
        elif value == 'Casa o Xalet':
            return 6
        elif value == 'Loft':
            return 7
        elif value == 'Casa Adosada':
            return 8
        elif value == 'Casa adosada':
            return 9
        elif value == 'Estudi':
            return 10


    caracteristicas = convertir_caracteristiques(caracteristicas_valor)

    habitaciones = st.selectbox('Habitacions', (0, 1, 2, 3, 4, 5, 6))
    aseos = st.selectbox('Lavabos', (1, 2, 3, 4))
    valor_terraza = st.selectbox('Terrassa', ('Sí', 'No'))
    terraza = 1 if valor_terraza == 'Sí' else 0
    valor_piscina = st.selectbox('Piscina', ('Sí', 'No'))
    piscina = 1 if valor_piscina == 'Sí' else 0
    valor_garaje = st.selectbox('Garatge', ('Sí', 'No'))
    garaje = 1 if valor_garaje == 'Sí' else 0
    metros = st.number_input('Metres', value=50)
    #codigoPostal = st.number_input('Codi postal', value=np.nan)
    value_barris = ('Sants-Montjuïc', 'Sant Martí', 'Ciutat Vella', 'Nou Barris',
                    'Sarrià-Sant Gervasi', 'Gràcia', 'Les Corts', 'Eixample',
                    'Sant Andreu', 'Horta-Guinardó')
    barri_valor = st.selectbox('Barri', value_barris)
    

    def convertir_barri_numeric(value):
        barri = ''
        preuM2 = ''
        pob_ocupada = ''
        renta_mitjana = ''

        if value == 'Ciutat Vella':
            barri = 0
            preuM2 = 4669.14
            pob_ocupada = 70.125
            renta_mitjana = 365.5
        elif value == 'Eixample':
            barri = 1
            preuM2 = 5536.92
            pob_ocupada = 94.943
            renta_mitjana = 117.028
        elif value == 'Gràcia':
            barri = 2
            preuM2 = 4533.45
            pob_ocupada = 104.16
            renta_mitjana = 116.94
        elif value == 'Horta-Guinardó':
            barri = 3
            preuM2 = 3218.85
            pob_ocupada = 105.18
            renta_mitjana = 89.105
        elif value == 'Les Corts':
            barri = 4
            preuM2 = 5745.78
            pob_ocupada = 96.86
            renta_mitjana = 141.76
        elif value == 'Nou Barris':
            barri = 5
            preuM2 = 2502.43
            pob_ocupada = 90.18
            renta_mitjana = 90.06
        elif value == 'Sant Andreu':
            barri = 6
            preuM2 = 3150.92
            pob_ocupada = 96.01
            renta_mitjana = 90.36
        elif value == 'Sant Martí':
            barri = 7
            preuM2 = 4411.24
            pob_ocupada = 99.53
            renta_mitjana = 107.71
        elif value == 'Sants-Montjuïc':
            barri = 8
            preuM2 = 3447.63
            pob_ocupada = 98.23
            renta_mitjana = 103.85
        elif value == 'Sarrià-Sant Gervasi':
            barri = 9
            preuM2 = 5602.69
            pob_ocupada = 100.9
            renta_mitjana = 151.516

        return barri, preuM2, pob_ocupada, renta_mitjana

    barri, preuM2, pob_ocupada, renta_mitjana = \
        convertir_barri_numeric(barri_valor)
    preuTeoric = preuM2 * metros
    
    def format_number(num):
        return f"{num:03d}"

    opcions = list(range(151))

    districte_censal = st.selectbox("Selecciona el districte censal", opcions)

    districte_censal_str = format_number(districte_censal)


with col2:
    # Botón para mostrar las características
    if st.button('Mostrar Característiques'):
        # Crear un gráfico de barras para visualizar las importancias
        # de las características
        importances = loaded_model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(range(len(importances)), importances)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Importància')
        ax.set_title('Importància de les característiques')

        # Mostrar el gráfico de barras en Streamlit
        st.pyplot(fig)

    # Botón para realizar la predicción
    if st.button('Predir'):
        # Crear un DataFrame con los valores de entrada
        input_data = pd.DataFrame([[caracteristicas, habitaciones, aseos,
                                    terraza, piscina, garaje, metros, barri,
                                    pob_ocupada, renta_mitjana, preuM2,
                                    preuTeoric]], columns=feature_names)

        # Realizar la predicción
        if districte_censal_str == '000':
            prediction = np.round(loaded_model.predict(input_data)/1000)*1000
        else:
            st.write('falta')
            

        # Mostrar el resultado de la predicción
        st.write(
            f"<h2 style='color: blue;'>El valor predit és "
            f"{int(prediction[0])}€</h2>", unsafe_allow_html=True)
