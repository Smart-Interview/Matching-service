import streamlit as st
from pyresparser import ResumeParser
# import tempfile

# def process_resume(uploaded_file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         temp_file_path = tmp_file.name  
    
#     data = ResumeParser(temp_file_path).get_extracted_data()
    
#     return data

data = ResumeParser('C:\\Users\\Bouchra HP\\Documents\\3D_Smart_Factory\\TEST1\\Extraction_donnees_CV\\Bouchra-Benghazala-CV-Recent.pdf').get_extracted_data()

st.write(data)