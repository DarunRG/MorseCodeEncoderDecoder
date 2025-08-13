import streamlit as st

alpha_morse_code = {
    "A": '.-', "B": '-...', "C": '-.-.', "D": '-..',
    "E": '.', "F": '..-.', "G": '--.', "H": '....',
    "I": '..', "J": '.---', "K": '-.-', "L": '.-..',
    "M": '--', "N": '-.', "O": '---', "P": '.--.',
    "Q": '--.-', "R": '.-.', "S": '...', "T": '-',
    "U": '..-', "V": '...-', "W": '.--', "X": '-..-',
    "Y": '-.--', "Z": '--..',
    "0": '-----', "1": '.----', "2": '..---', "3": '...--',
    "4": '....-', "5": '.....', "6": '-....', "7": '--...',
    "8": '---..', "9": '----.'
}
morse_code_alpha = {v:k for k, v in alpha_morse_code.items()}

def encode_text(input_data):
    encode_data = ''
    n = len(input_data)
    if n == 0:
        return ''
    for i in range(n):
        if input_data[i] == ' ':
            encode_data += '   '
        elif i < (n-1) and input_data[i+1] == ' ':
            encode_data = encode_data + alpha_morse_code[input_data[i]]
        else:
            encode_data = encode_data + alpha_morse_code[input_data[i]] + ' '
    return encode_data

def decode_text(input_data):
    decode_data = ''
    n = len(input_data)
    if n == 0:
        return ''
    i = 0
    while i < n:
        if i < (n - 2) and input_data[i-1:i+2] == '   ':
            decode_data += ' '
        elif input_data[i] == ' ':
            i += 1
            continue
        else:
            j = i
            morse_code = ''
            while j < n and input_data[j] != ' ':
                morse_code += input_data[j]
                j += 1
            i = j
            decode_data += morse_code_alpha.get(morse_code, '')
        i += 1
    return decode_data

def is_morse(s):
    allowed_chars = {'.', '-', ' '}
    return all(c in allowed_chars for c in s.strip())

st.set_page_config(page_title="Morse Code Auto Encoder-Decoder", page_icon="MC-AED", layout="centered")
st.title("Morse Code Auto Encoder-Decoder")

user_input = st.text_area("Enter text or Morse code:")

if st.button("Convert"):
    if user_input.strip():
        if is_morse(user_input):
            result = decode_text(user_input.strip())
            st.success("Decoded Text:")
            st.code(result)

        else:
            result = encode_text(user_input.upper())
            st.success("Encoded Morse:")
            st.code(result)
    else:
        st.warning("Please enter some text or Morse code.")
