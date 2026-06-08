import streamlit as st
import os
import tempfile

from model.classifier import classify


st.markdown(
    "<h1><span style='color:red'>Mal</span><span style='color:#22cc55'>Vision</span></h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<small style='color:#888888'>by Raffaele Passaro</small>",
    unsafe_allow_html=True
)

st.write(
    "Upload a file and detect if it is malware."
)


uploaded_file = st.file_uploader(
    "Upload your file here!",
    accept_multiple_files=False
)

already_converted = st.checkbox("File is already a binary file converted to image (PNG)")

if uploaded_file is not None:

    if st.button("Analyze"):

        try:
            # salva file upload
            upload_path = os.path.join(
                tempfile.gettempdir(),
                uploaded_file.name
            )

            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # prediction + immagine malware-style
            result, image_path, confidence = classify(
                upload_path,
                "output_model.pth",
                already_converted
            )

            # mostra immagine usata dal modello
            st.subheader("Binary visualization")
            st.image(
                image_path,
                caption="Representation used by the model",
                use_container_width=True
            )

            # risultato
            st.subheader("Result")

            if result == "benign":
                st.success("The file is benign.")
            else:
                st.error(f"Malware detected: {result}")
            st.write(f"Confidence: {confidence:.2%}")

        except Exception as e:
            st.exception(e)