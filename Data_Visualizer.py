import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn import datasets
import plotly.graph_objects as go
import random
# Author: Neha Karanjkar, IIT Goa

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

def plot_3d(data, selected_x_axis, selected_y_axis, selected_function):
    fig = px.scatter_3d(data, x=selected_x_axis, y=selected_y_axis,z=selected_function, color=selected_function, color_continuous_scale=st.session_state.colorscale)
    fig.update_layout(scene = dict(aspectmode='cube'),template='plotly')
    fig.update_traces(marker_size=st.session_state.marker_size)
    
    return fig

@st.cache
def generate_data(num_rows, num_axes, num_functions):
    X,F_X = datasets.make_regression(n_samples=num_rows, n_features=num_axes, n_informative=int(num_axes/2), 
            n_targets=num_functions, effective_rank=int(num_axes/2), shuffle=True)
    column_names=["X"+str(i+1) for i in range(num_axes)]
    column_names+= ["F"+str(i+1) for i in range(num_functions)]
    df = pd.DataFrame(data=np.concatenate((X,F_X),axis=1),columns=column_names)
    return df

def to_csv(df):
    data_csv = df.to_csv(path_or_buf=None, sep=',', index=False) 
    return data_csv

@st.cache
def slice_data(data, other_axes, values):
    data_slice=data
    for i, axis in enumerate(other_axes):
        data_slice = data_slice[(data_slice[axis] >= values[i][0]) & (data_slice[axis] <= values[i][1])]
    return data_slice


def main():
    
    # Configs and sidebar
    st.set_page_config(
        page_title="DATA vis",
        page_icon="favicon.png",
        layout="wide",
        initial_sidebar_state="expanded")

    #st.sidebar.image("logo.png",width=100)
    #st.sidebar.header("Multi-dimensional Data Visualizer")
    #st.sidebar.caption("""
    #A simple tool for visualizing samples 
    #of multi-dimensional data
    #over a multi-dimensional space.
    #""")
    #st.sidebar.markdown("Made by [Neha Karanjkar](https://nehakaranjkar.github.io/)")
    st.sidebar.subheader("🚀  [About](#about)")
    st.sidebar.subheader("📃  [Data Setup](#set-up-the-data)")
    st.sidebar.subheader("📊  [Visualize](#visualize)")
    with st.sidebar:
        c1,c2 = st.columns([10,90])
        with c2:
            marker_size = st.slider("⚙️ Marker size", min_value=1, max_value=10, value=7, step=1,key="marker_size")
            colorscale = st.selectbox(' ⚙️ Color scale',px.colors.named_colorscales(),key="colorscale", index=48)
    st.sidebar.markdown("---")



    # Main page ====================
    st.image("DATA_vis.png",width=500)
    st.subheader("Multi-dimensional Data Visualizer")
    st.markdown("""
    A simple tool to visualize data points or measurements of vector functions over a multi-dimensional space.
    It is made particularly for visualizing objective function samples for black-box optimization problems.
    """
    )
    st.markdown("Made by Neha Karanjkar [(Webpage)](https://nehakaranjkar.github.io/)")
    
    # Data setup ====================
    st.markdown("""---""")
    st.subheader("Set up the data")
    
        

    c1,c2,c3=st.columns([30,10,60])
    with c1:
        data_choice = st.radio("Select a data source:",  ( 'Use randomly generated data','Upload a csv file'), index=0)
        if data_choice == 'Use randomly generated data':
            num_rows = st.slider("Number of data samples ", min_value=1, max_value=1000, value=100, step=1)
            num_axes = st.slider("Number of input dimensions (axes)", min_value=1, max_value=10, value=5, step=1)
            num_functions= st.slider("Number of output dimensions (function components)", min_value=1, max_value=10, value=3, step=1)
        else:
            st.write("The expected format is as shown below. ")
            st.image("data_format.png",width=200)
            st.markdown("""
            The first N columns should contain the N-dimensional coordinate 
            values and the last M columns should contain the values of the M function components at each point.
            The first row should contain the column names.
            """)

    with c3:
        if data_choice == 'Use randomly generated data':
            if st.button('Generate sample data'):
                st.write(f"Generating synthetic data with {num_rows} points ...")
                data = generate_data(num_rows,num_axes,num_functions)
                st.session_state.data_generated=True
                st.session_state.data=data
                with st.expander("View raw data"):
                    st.write(data)
                st.download_button('Download generated data as a CSV file', to_csv(data), 'sample_data.csv', 'text/csv')

        else:
            uploaded_file = st.file_uploader("Upload data as a csv file", type="csv")
            if uploaded_file is not None:
                data = load_data(uploaded_file)
                if data.shape[1] < 3:
                    st.error("Error: The uploaded CSV file must have at least 3 columns.")
                    return
                else:
                    st.session_state.data_generated=True
                    st.session_state.data=data
                    with st.expander("View raw data"):
                        st.write(data)
       
    if 'data' not in st.session_state:
        return
    
    #Visualize =========================
    st.markdown("""---""")
    st.subheader("Visualize")
    
    if st.session_state.data_generated:
        data = st.session_state.data
        
        column_names=list(data.columns)
        st.write("Select columns corresponding to the axis and function values in your csv file")
        if data_choice != 'Upload a csv file':
            default_axes = column_names[0:num_axes]
        else:
            default_axes = column_names[0:2]

        axes = st.multiselect("Columns containing axis values", column_names, default=default_axes)
        default_functions = [f for f in column_names if f not in axes]
        functions = st.multiselect("Columns containing function values", default_functions, default=default_functions)


        st.write("Select the components to plot as a 3D slice")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_x_axis = st.selectbox("x-axis", axes, index=0)
        with col2:
            selected_y_axis = st.selectbox("y-axis", axes, index=1)
        with col3:
            selected_function = st.selectbox("Function to plot", functions, index=0)
        
        st.markdown("""---""")
        col1, col2 = st.columns([3,7])
        with col1:
            # Get the values of the other axes from the user
            other_axes = [i for i in axes if i not in [selected_x_axis, selected_y_axis]]
            values = []
            if other_axes:
                st.write("Select the range of values for the other axes")
                for axis in other_axes:
                    min_value = float(data[axis].min())
                    max_value = float(data[axis].max())
                    values.append(st.slider(f"{axis} range", min_value=min_value, max_value=max_value, value=(min_value, max_value)))
            data_slice = slice_data(data, other_axes, values)
            st.write(f"The selected slice contains {len(data_slice)} points")

        with col2:
            fig = plot_3d(data_slice, selected_x_axis, selected_y_axis, selected_function)
            st.plotly_chart(fig, use_container_width=True)

        #st.write(data_slice)

if __name__ == "__main__":
    main()

