import streamlit as st
st.title("BASIC TITLE")
st.header("header part")
st.text("this part contains basic text")
st.markdown("***bold text** and Italic text* ")

option=st.radio('select a option:',['M','E',['N']])
st.write("you selected",option)

#dropdownbox
city=st.selectbox("select city",['Delhi','Mumbai','Bangalore'])
st.write(f"you chose {city}")

name=st.text_input("Enter your name:")
st.write("hello",name)

#dateinput
#timeinput
#

