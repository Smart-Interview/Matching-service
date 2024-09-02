
import streamlit as st
from PIL import Image
from functions import *
from sqlalchemy import create_engine, text

# conn = st.connection(
#     "local_db",
#     type="sql",
#     url="mysql://root:@localhost:3306/extraction_nlp_recrutement"
# )
# conn = st.connection("mydb", type="sql", autocommit=True)

# engine = create_engine("mysql://root:@localhost:3306/extraction_nlp_recrutement")
# conn = engine.connect()

# if conn:
#     st.warning("Connection established!")

#     select_query = text("SELECT * FROM score")
#     result = conn.execute(select_query)

#     rows = result.fetchall()

#     for row in rows:
#         st.write(f"ID Offre: {row[1]}, ID Candidat: {row[2]}, Score: {row[3]}")
# else:
#     st.warning("Oops,connection not established")



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
            resumee_responses["global"] = get_text_info_from_pdf(resumee, "Veuillez extraire juste les mots clés du sexe (tu peux déduire le sexe), la formation (tous les formations mentionnées), le lieu (donne moi un seul lieu le plus récent), les compétences, expérience (donne moi juste le nombre d'expérience en années pour la partie expérience) mentionnées de ce CV sans écrire le titre (informations...) et sans mentionner d'informations supplémentaires, le résultat doit etre sous forme espace **nom:** espace, ne pas afficher de \n") 
            
            job_description_responses = {key: get_text_info_from_pdf(job_description, req) for key, req in requests_job.items()}
            job_description_responses["global"] = get_text_info_from_pdf(job_description, "Veuillez extraire juste les mots clés du sexe (si ce n'est pas affiché écrit rien), la formation (tous les formations mentionnées), le lieu, les compétences, expérience (donne moi juste le nombre d'expérience en mois ou en années) demandées dans cette offre d'emploi sans écrire le titre (informations...) sans mentionner d'informations supplémentaires, le résultat doit etre sous forme espace **nom:** espace ne pas afficher de \n")

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

            for key in keywords_resumee:
                keywords_resumee[key] = [item.replace('\\n', '').replace('\n', '') for item in keywords_resumee[key]]
                keywords_resumee['experiences'] = [keywords_resumee['experiences'][0].replace('"}role: "model"', '').strip()]

            print(keywords_resumee)

            for key in keywords_job_desc:
                keywords_job_desc[key] = [item.replace('\\n', '').replace('\n', '') for item in keywords_job_desc[key]]
                if keywords_job_desc['experiences'] and len(keywords_job_desc['experiences']) > 0:
                    keywords_job_desc['experiences'] = [keywords_job_desc['experiences'][0].replace('"}role: "model"', '').strip()]
                else:
                    st.error("Aucune expérience extraite de l'offre d'emploi.")


            print(keywords_job_desc)


            with tabs[0]:

                st.write(keywords_resumee)

            with tabs[1]:
                st.write(keywords_job_desc)

    with st.expander("Score"):
        with st.spinner("En cours..."):
            score = similarity(keywords_resumee, keywords_job_desc)
            st.write(f"Le score de compatibilité entre le CV et l'offre d'emploi est : {score}")


            # try:
            #     query_candidat = text("""
            #     INSERT INTO score (id_offre, id_candidat, score)
            #     VALUES (:id_offre, :id_candidat, :score)
            #     """)
            #     conn.execute(query_candidat, {"id_offre": 3, "id_candidat": 3, "score": score})
            #     conn.commit()  # Si nécessaire, validez la transaction

            #     st.success("Les données ont été insérées avec succès.")
            # except Exception as e:
            #     st.error(f"Erreur lors de l'insertion : {e}")
            # finally:
            #     conn.close()  # Assurez-vous de fermer la connexion


