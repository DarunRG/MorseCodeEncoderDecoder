import streamlit as st
import tempfile
from pathlib import Path
import morse_code_encoder_decoder as mc
import extract_image_data as eid

st.set_page_config(page_title="Morse Code Auto Encoder-Decoder", page_icon="", layout="centered")
st.title("Morse Code Auto Encoder-Decoder")

tab1, tab2 = st.tabs(["✍️ Text Input", "🖼️ Image Input"])

with tab1:
    user_input = st.text_area("Enter Text or Morse Code")
    if st.button("Process Text"):
        if mc.is_morse(user_input.strip()):
            decoded_text = mc.decode_text(user_input.strip())
            st.success("Decoded Text:")
            st.code(decoded_text)
        else:
            encoded_text = mc.encode_text(user_input.upper())
            st.success("Encoded Morse Code:")
            st.code(encoded_text)

with tab2:
    uploaded_file = st.file_uploader("Upload Morse Code Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        temp_path = Path(tempfile.gettempdir()) / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(str(temp_path), caption="Uploaded Morse Code Image", use_column_width=True)

        if st.button("Process Image"):
            try:
                morse_code = eid.extract_morse_auto(str(temp_path))

                st.success("Extracted Morse Code:")
                st.code(morse_code)

                decoded_text = mc.decode_text(morse_code.strip())
                st.success("Decoded Text:")
                st.code(decoded_text)

            except Exception as e:
                st.error(f"Error while processing image: {e}")
