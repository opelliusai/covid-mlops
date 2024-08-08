'''
Créé le 08/08/2024

@author: Jihane EL GASMI - MLOps Avril 2024
@summary: Page pour la prédiction d'une Anomalie Pulmonaire (COVID ou Pneumonia Virale)
-- Gestion du compte
    -- Inscription (à venir) 
    -- Authentification (Avec Profil simple ou admin)
    -- Modification du mot de passe (à venir)
    -- Suppression du compte (à venir)
-- Prédiction
    -- Historique des prédictions de l'utilisateur (à venir)
    -- Exécution d'une prédiction et visualisation du résultat avec indice de confiance
    -- Action: Valider/Invalider/Modifier la prédiction
    Action en Backend:
    - Mettre à jour le log des predictions
    - Mettre à jour de la BDD locale pour ajouter les bonnes prédictions à une nouvelle version du dataset
'''

### Import des modules
import streamlit as st
from fastapi import UploadFile
from io import BytesIO
### Fin des imports

from src.config.log_config import logger
from src.config.run_config import init_paths

## Imports des modules internes
from src.utils import utils_data
from src.utils import utils_streamlit

def main():
    # Affichage du titre
    st.title("Détection d'une anomalie Pulmonaire")
    # Configuration de 2 colonnes : Upload d'une prédiction et affichage du résultat
    col_upload,col_res=st.columns([1, 1])
    ## COLUMN 1
    with col_upload:
        uploaded_file = st.file_uploader("Choisissez une image de Radiologie Pulmonaire...", type=["jpg", "jpeg", "png"])
        if uploaded_file :
                logger.debug(f"Type de Fichier {type(uploaded_file)}")
                #file_path = os.path.join(user_folder, uploaded_file.name)
                logger.debug(f"filename {uploaded_file.name}")
         
        ## COLUMN 2 :
        with col_res:
            if uploaded_file :
                #access_token = st.session_state.access_token
                logger.debug("Conversion en UploadFile lisible par FastAPI")
                file_like = BytesIO(uploaded_file.read())
                filename=uploaded_file.name
                #uploaded_file = UploadFile(file=fichier, filename=uploaded_file.name)
                prediction,confiance,temps_prediction,image_upload_path=utils_streamlit.lancer_une_prediction(file_like,filename)
                st.image(uploaded_file, caption=f"{prediction}, à {confiance}% en {temps_prediction} s")
                '''
                col_val,col_inval = st.columns([1, 1])
                with col_val:
                    if st.button("Valider", on_click=on_button_click):
                        st.success("La prédiction a été validée !")
                        utils_streamlit.ajout_image_classe(access_token,prediction,uploaded_file)
                        uploaded_file=None
                        st.rerun() # raffraichissement de la page
                with col_inval:
                    if st.button("Je ne sais pas", on_click=on_button_click):
                        st.success("La reconnaissance n'a pas été validée sans proposition!")
                        su.ajout_image_classe(access_token,prediction,uploaded_file)
                        uploaded_file=None
                        st.rerun() # raffraichissement de la page
                input_id = "proposition_input"
                st.text_input("Proposition", key=input_id)
                autocomplete_script = su.generate_autocomplete_script(input_id, list_birds)
                st.markdown(autocomplete_script, unsafe_allow_html=True)    
                if st.button("Soumettre", on_click=on_button_click_proposition):
                        proposition=st.session_state.input_value
                        print(f"Proposition {proposition}")
                        su.ajout_image_classe(access_token,proposition,temp_file)
                        temp_file=None
                        st.success("La proposition a été validée !")
                        uploaded_file=None
                        st.rerun() # raffraichissement de la page
                '''

    

if __name__ == "__main__":
    main()
   
