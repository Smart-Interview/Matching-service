
import streamlit as st
from PIL import Image
from functions import *
import sqlalchemy


# conn = st.connection(
#     "local_db",
#     type="sql",
#     url="mysql://root:@localhost:3306/extraction_nlp_recrutement"
# )
conn = st.connection("mydb", type="sql", autocommit=True)



resumee = st.file_uploader("Veuillez télécharger les CVs", type=["pdf", "docx"], accept_multiple_files=False)
job_description = st.file_uploader("Veuillez télécharger l'offre d'emploi", type=["pdf", "docx"], accept_multiple_files=False)

if resumee and job_description:
    with st.expander("Extraction des informations"):
        with st.spinner("En cours..."):
            col_a, col_b = st.columns(2)

            requests_resumee = {
                "sexe": "Veuillez extraire juste le sexe (Femme ou Homme) à partir du CV sans mentionner d'informations supplémentaires, tu peux conclure le sexe à partir du prénom si c'est possible",
                "formation": "Veuillez extraire juste les mots clés des formations à partir du CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "localisation": "Veuillez extraire juste la localisation actuelle (plus récente) à partir du CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "competences": "Veuillez extraire juste les mots clés des compétences à partir du CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "experiences": "Veuillez extraire juste les mots clés des expériences à partir du CV sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)"
            }
            requests_job = {
                "sexe": "Veuillez extraire juste le sexe demandé pour cet emploi sans mentionner d'informations supplémentaires",
                "formation": "Veuillez extraire juste les mots clés de la formation demandée pour cet emploi sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "localisation": "Veuillez extraire juste la localisation demandée pour cet emploi sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "competences": "Veuillez extraire juste les mots clés des compétences demandées pour cet emploi sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)",
                "experiences": "Veuillez extraire juste les mots clés des expériences demandées pour cet emploi sans mentionner d'informations supplémentaires et sans inclure de titre (informations...)"
            }

            resumee_responses = {key: get_text_info_from_pdf(resumee, req) for key, req in requests_resumee.items()}
            resumee_responses["global"] = get_text_info_from_pdf(resumee, "Veuillez extraire juste les mots clés du sexe (tu peux déduire le sexe), la formation, le lieu, les compétences, expérience professionnelle mentionnées de ce CV sans écrire le titre (informations...) et sans mentionner d'informations supplémentaires, le résultat doit etre sous forme espace **nom:** espace, ne pas afficher de \n") 
            
            job_description_responses = {key: get_text_info_from_pdf(job_description, req) for key, req in requests_job.items()}
            job_description_responses["global"] = get_text_info_from_pdf(job_description, "Veuillez extraire juste les mots clés du sexe, la formation, le lieu, les compétences, expérience professionnelle demandées dans cette offre d'emploi sans écrire le titre (informations...) sans mentionner d'informations supplémentaires, le résultat doit etre sous forme espace **nom:** espace ne pas afficher de \n")

            with col_a:
                st.write("**Informations du CV**")
                st.write(resumee_responses["global"])
               

            with col_b:
                st.write("**Informations de l'offre d'emploi**")
                st.write(job_description_responses["global"])

    with st.expander("Extraction des mots-clés"):
        with st.spinner("En cours..."):
            tabs = st.tabs(["CVs", "Offre d'emploi"])

            keywords_resumee = parse_global_info(resumee_responses["global"])
            keywords_job_desc = parse_global_info(job_description_responses["global"])

            with tabs[0]:
                st.write(keywords_resumee)
                print(keywords_resumee)

            with tabs[1]:
                st.write(keywords_job_desc)

    # with st.expander("Score"):
    #     with st.spinner("En cours..."):
    #         score = rank_cvs_with_embeddings(keywords_resumee, keywords_job_desc)
    #         st.write(f"Le score de compatibilité entre le CV et l'offre d'emploi est : {score}")

    #     if conn:
    #         st.warning("Connection established!")
    #         query_candidat = f"""
    #         INSERT INTO score (id_offre, id_candidat, score)
    #         VALUES (
    #             2,
    #             2,
    #             {score}
    #         )
    #         """
    #         result=conn.query(query_candidat)
    #         result.close()
    #     else:
    #         st.warning("Oops, connection not established")


