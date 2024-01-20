#Import required libraries
import os 
#from Apikey import apikey 

import streamlit as st
import pandas as pd
import csv
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
#########New Libraries
import os

os.environ['OPENAI_KEY']
#load_dotenv(find_dotenv())
os.environ.get('OPENAI_KEY')
from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
#from langchain.agents.agent_toolkits import create_python_agent
#from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
#from langchain.agents.agent_types import AgentType

# Parte criada Por Mim
import time  # to simulate a real time data, time loop
import pandas as pd
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # data web app development
import pandas as pd
st.set_page_config(
    page_title="Real-Time - Data Science Dashboard",
    page_icon="",
    layout="wide",
)
import pandas as pd
import pandas as pd
import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.figure_factory as px
import plotly.figure_factory as line
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import csv
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv, find_dotenv

#OpenAIKey
#os.environ['OPENAI_API_KEY'] = apikey


#Titulo 1

st.title("Estatisticas da Saúde Mental no Período de 2004 a 2020 ")

st.write(
    """Estudo Efetuado pelo INE - Instiituto Nacional de Estatística sobre o estado da Saúde Mental dos Portugueses 
    divídidos por Género, Faixa Étaria e Ocupação Profissional
    """
)

#Title
st.title('AI aprendizagem para Ciência de Dados  🤖')

#Welcoming message
st.write("Olá, 👋 Eu Sou um assistente de Data Science que pode ajudar a analizar os seus projetos.")

#Explanation sidebar
with st.sidebar:
    st.write('* Uma aventura de data science começa com um csv.*')
    st.caption('''** como já deves saber precisamos de um dataset
    precisamos que nos envie o seu DataSeT, Assim que tivermos os seus dados sabemos responder e relacionar para prever o que podemos fazer com ele**
    ''')


#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

#Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Vamos começar!", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your CSV file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        #llm model
        llm = OpenAI(temperature = 0)

        #Function sidebar
        @st.cache_data
        def steps_eda():
            steps_eda = llm('What a virtual assistant do withis dataSet to improve medical and resources for Mental Health')
            return steps_eda

        #Pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)

        #Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head(20))
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
        
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            #correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            #st.write(correlation_analysis)
            #outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            #st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y =[user_question_variable])
            st.area_chart(df, y =[user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            #outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            #st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            #missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            #st.write(missing_values)
            return
        
        

        #Main

        st.header('Exploração de Dados')
        st.subheader('Informação Geral sobre o Dataset')

        with st.sidebar:
            with st.expander('What a Virtual assistant Can do for Health : '):
                st.write(steps_eda())

        function_agent()
        ##### Alteração da Linha 328 para a linha 158
        def filter_data(df: pd.DataFrame) ->pd.DataFrame:
            options = st.multiselect("escolha Um Fator de Risco ", options=df.columns)
            st.write('Voçê selecionou as seguintes opções', options)
            df = pd.read_csv(user_csv, low_memory=False)
            #st.dataframe(filter_dataframe(df))
            
        def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            
            df = pd.read_csv(user_csv, low_memory=False)
            st.dataframe(filter_dataframe(df))
            """
            Adds a UI on top of a dataframe to let viewers filter columns
            Args:
                df (pd.DataFrame): Original dataframe
            Returns:
                pd.DataFrame: Filtered dataframe
            """

            modify = st.text_input(
                "Escolha os Fatores 👇", df.columns,
                #label_visibility=st.session_state.visibility,
                #disabled=st.session_state.disabled,
                #placeholder=st.session_state.placeholder,

            )
            if not modify:
                return df

            df = df.copy()

            # Try to convert datetimes into a standard format (datetime, no timezone)
            for col in df.columns:
                if is_object_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception:
                        pass

                if is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.tz_localize(None)

            modification_container = st.container()

            with modification_container:
                to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
                for column in to_filter_columns:
                    left, right = st.columns((1, 20))
                    left.write("↳")
                    # Treat columns with < 10 unique values as categorical
                    if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                        user_cat_input = right.multiselect(
                            f"Values for {column}",
                            df[column].unique(),
                            default=list(df[column].unique()),
                        )
                        df = df[df[column].isin(user_cat_input)]
                    elif is_numeric_dtype(df[column]):
                        _min = float(df[column].min())
                        _max = float(df[column].max())
                        step = (_max - _min) / 100
                        user_num_input = right.slider(
                            f"Values for {column}",
                            _min,
                            _max,
                            (_min, _max),
                            step=step,
                        )
                        df = df[df[column].between(*user_num_input)]
                    elif is_datetime64_any_dtype(df[column]):
                        user_date_input = right.date_input(
                            f"Values for {column}",
                            value=(
                                df[column].min(),
                                df[column].max(),
                            ),
                        )
                        if len(user_date_input) == 2:
                            user_date_input = tuple(map(pd.to_datetime, user_date_input))
                            start_date, end_date = user_date_input
                            df = df.loc[df[column].between(start_date, end_date)]
                    else:
                        user_text_input = right.text_input(
                            f"Substring or regex in {column}",
                        )
                        if user_text_input:
                            df = df[df[column].str.contains(user_text_input)]

            return df
        #########################


        #SOME ARTIFICIAL INTELIGENCE

        st.subheader('Variavél em Estudo')
        user_question_variable = st.text_input('Qual a variavel em que está interessado?')
        if user_question_variable is not None and user_question_variable !="":
            function_question_variable()

        st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input( "há algo mais que esteja interessado num dataset")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                if prompt:
                    response = llm(prompt)
                    st.write(response)
            if user_question_dataframe in ("no", "No"):
                st.write("")
        # Second Part
            if user_question_dataframe:
                st.divider()
                st.header("Mental Health Data Science Problem")
                st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our health problem into a data science problem.")
                
                prompt = st.text_input('Apresente aqui a sua Questão')
                if prompt:
                    response = llm(prompt)
                    st.write(response)

                data_problem_template = PromptTemplate(
                input_variables=['business_problem'],
                template='Convert the following Mental Health problem into a data science problem: {business_problem}.')

                data_problem_chain = LLMChain(llm=llm, prompt=data_problem_template, verbose=True)

                data_problem_chain = LLMChain(llm=llm, prompt=data_problem_template, verbose=True)

                if prompt:
                    response = data_problem_chain.run(business_problem=prompt)
                    st.write(response)

                model_selection_template = PromptTemplate(
                    input_variables=['data_problem'],
                    template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}')
                
                data_problem_template = PromptTemplate(
                    input_variables=['business_problem'],
                    template='Convert the following business problem into a data science problem: {business_problem}.'
                )

        model_selection_template = PromptTemplate(
            input_variables=['data_problem'],
            template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}.'


        )

        data_problem_chain = LLMChain(llm=lmm, prompt=data_problem_template, verbose=True)
        model_selection_chain = LLMChain(llm=llm, prompt=model_selection_template, verbose=True)
        sequential_chain = SimpleSequentialChain(chains=[data_problem_chain, model_selection_chain], verbose=True)

        if prompt:
            response = sequential_chain.run({prompt})
            st.write(response)
        
        ######################### primeira parte Criada por mim ---
        import pandas as pd
        import streamlit as st
        import numpy as np
        import plotly.figure_factory as ff
        import plotly.figure_factory as px
        import plotly.figure_factory as line
        from pandas.api.types import (
            is_categorical_dtype,
            is_datetime64_any_dtype,
            is_numeric_dtype,
            is_object_dtype,
        )
        
        

       

#df = pd.read_csv(
    #"MentalHealth.csv"
        #df = pd.read_csv('/Users/paulomonteiro/Desktop/PycharmProjects/MentalFinal/Mentalhealth3.csv')
        #st.dataframe(filter_dataframe(df))
        
        st.dataframe(df)
        col1, col2 = st.columns(2)
        with col1:
            chart_data = pd.DataFrame(
            np.random.randn( 22 , 5),
            columns=['Mulheres', 'Homens', 'Ensino Superior', 'Desempregados', 'Reformados' ])


         
            #column2=['25','50','75', '80', '100']
        with col2:
            st.area_chart(chart_data)

        # Example dataframe
        df = pd.read_csv('/Users/paulomonteiro/Desktop/PycharmProjects/MentalFinal/Mentalhealth3.csv')
        col1, col2 = st.columns(2)
        with col1:
             # plot
            st.area_chart(data = df, x= "Date1",y='Total')


        with col2:
            chart_data = pd.DataFrame(
            np.random.randn( 2 , 5),
            {

            'Date1': [2004,2008,2010,2015,2022],
            'columns':['Mulheres', 'Homens', 'Ensino Superior', 'Desempregados', 'Reformados' ] })


            #column2=['25','50','75', '80', '100']
            st.area_chart(chart_data)

        col1, col2 = st.columns(2)
        with col1:
            df = pd.read_csv('/Users/paulomonteiro/Desktop/PycharmProjects/MentalFinal/Mentalhealth3.csv')
            st.area_chart( df, x="Date1", y='Total')

            df = pd.DataFrame(
            {"Date1": [2008, 2011, 2018, 2020], "values": [0, 25, 50, 75], "values_2": [15, 25, 45, 85]}

            ).set_index("Date1")

            df_new = pd.DataFrame(
            {"steps": [4, 5, 6], "Homens": [0.5, 0.3, 0.5], "Mulheres": [0.8, 0.5, 0.3]}
            ).set_index("steps")

            df_all = pd.concat([df, df_new], axis=0)
        with col2:
            st.line_chart(chart_data, x=df.all,)
            #st.line_chart(df, x=df.index, y=["Homens", "Mulheres"])







        # Add histogram data
            x1 = np.random.randn(200) - 2
            x2 = np.random.randn(200)
            x3 = np.random.randn(200) + 2

        # Group data together
            hist_data = [x1, x2, x3]

            group_labels = ['Homens', 'Mulheres', 'Ensino superior']

        # Create distplot with custom bin_size
            fig = ff.create_distplot(
                hist_data, group_labels, bin_size=[2008, 2010, 2020])

        # Plot!
        col1,  = st.columns(1)
        with col1:
            st.plotly_chart(fig, use_container_width=True)

        
            p = open("/Users/paulomonteiro/Desktop/PycharmProjects/MentalFinal/lda.html")
            components.html(p.read(), width=1000, height=800, )








        #######################
        df = pd.read_csv('/Users/paulomonteiro/Downloads/arte-urbana - arte_urbana_fev2022.csv')
        #df = pd.read_csv('https://query.data.world/s/edvqfu4ea2ap6sbmr7ht2szxjc2mij')
        placeholder = st.empty()
        #Title
        st.title = "Real time computer Science Dashboard 1"

        job_filter = st.selectbox("A Unidade de Saúde", pd.unique(df["Localizacao"]))

        for seconds in range(200):
            df["lon"] = df["LON"] * np.random.choice(range(1, 5))
            df["lat"] = df["LAT"] * np.random.choice(range(1, 5))

            # creating KPIs
            #avg_age = np.mean(df["LON"])

            count_married = str(
                df[(df["Freguesia"] == "Freguesia")]["Freguesia"].count()
                + np.random.choice(range(1, 1000))
            )

            balance = np.mean(df["Data"])

        # with placeholder.container():
                    # create three columns
                    #kpi1, kpi2, kpi3 = st.columns(3)

                    #fill in those three columns with respective metrics or KPIs
                    ##   label="LONG ⏳",
                    #    value=round(avg_age),
                    #    delta=round(avg_age) - 10,
                # )

                # kpi2.metric(
                    #    label="LAT 💍",
                    #   value=int(Data),
                    #   delta=-10 + count_married,
                #  )

                # kpi3.metric(
                    #    label="Freguesia",
                    #    value=f"$ {round(balance, 2)} ",
                    #    delta=-round(balance / count_married) * 100,
                #  )

            import altair as alt
            fig_col1, fig_col2 = st.columns(2)

            with fig_col1:
                chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["LAT", "LON", "Freguesia"])
                st.area_chart(chart_data)
            

            with fig_col2:

                chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Freguesia", "LAT", "LON"])

            st.bar_chart(chart_data)
            #
            # st.title = "Location ID"
            # st.markdown("Location ID")
            # chart_data = pd.DataFrame(
                #  np.random.randn(20, 3),
                #  columns=['LAT', 'LON'])

            # c = alt.Chart(chart_data).mark_circle().encode(
                #    x='LAT', y='LON', size='Localizacao', color='Localizacao', tooltip=['LAT', 'LON', 'Localizacao'])

            # st.altair_chart(c, use_container_width=True)



            #with fig_col2:
            #  st.markdown("Tree Map")
            #  fig = px.density_heatmap(
            #     data_frame=df, y="ID", x="Freguesia"
            # )
            # st.write(fig)



            st.title = "Road Map of Data Science"
            st.markdown("road map")

            #df = pd.read_csv('arte-urbana - arte_urbana_fev2022.csv', usecols=['Freguesia', 'LONG', 'LAT'])
            #df = pd.DataFrame(df.columns = ['Freguesia', 'LONG', 'LAT'])
            #st.map(df)



            import streamlit as st
            import pandas as pd
            import numpy as np

        df = pd.DataFrame(
            np.random.randn(137, 2) / [50, 50] + [38.75, -9.2],
            columns=['LAT', 'LON'])

        st.map(df)

## fim de parte Criada por mim

st.subheader('Variavél em Estudo')
user_question_variable = st.text_input('Qual a variavel em que está interessado?')
if user_question_variable is not None and user_question_variable !="":
    function_question_variable()

st.subheader('Further study')

if user_question_variable:
    user_question_dataframe = st.text_input( "há algo mais que esteja interessado num dataset")
    if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
        if prompt:
            response = llm(prompt)
            st.write(response)
    if user_question_dataframe in ("no", "No"):
        st.write("")
# Second Part
    if user_question_dataframe:
        st.divider()
        st.header("Mental Health Data Science Problem")
        st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our health problem into a data science problem.")
        
        prompt = st.text_input('Apresente aqui a sua Questão')
        if prompt:
            response = llm(prompt)
            st.write(response)

        data_problem_template = PromptTemplate(
        input_variables=['business_problem'],
        template='Convert the following Mental Health problem into a data science problem: {business_problem}.')

        data_problem_chain = LLMChain(llm=llm, prompt=data_problem_template, verbose=True)

        data_problem_chain = LLMChain(llm=llm, prompt=data_problem_template, verbose=True)

        if prompt:
            response = data_problem_chain.run(business_problem=prompt)
            st.write(response)

        model_selection_template = PromptTemplate(
            input_variables=['data_problem'],
            template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}')
        
        data_problem_template = PromptTemplate(
            input_variables=['business_problem'],
            template='Convert the following business problem into a data science problem: {business_problem}.'
        )

        model_selection_template = PromptTemplate(
            input_variables=['data_problem'],
            template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}.'


        )

        data_problem_chain = LLMChain(llm=lmm, prompt=data_problem_template, verbose=True)
        model_selection_chain = LLMChain(llm=llm, prompt=model_selection_template, verbose=True)
        sequential_chain = SimpleSequentialChain(chains=[data_problem_chain, model_selection_chain], verbose=True)

        if prompt:
            response = sequential_chain.run({prompt})
            st.write(response)
        
