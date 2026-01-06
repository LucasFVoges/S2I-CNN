import streamlit as st
import datetime

st.title("S2I-CNN")
st.write("An example of how the 'Spectra2Image CNN' can be used.")

st.header("Example")
checkbox_advanced_opt = st.checkbox("show advanced options", value = False)
radio_btn = st.radio("Data:", ("Simulated Data", "Spectral Data"), horizontal=True)
select_box = st.selectbox("Select a model:", ("S2I-CNN", "S2I-CNN-VGG"))
multiselect_box = st.multiselect("Select multiple conversion models:", ["RGB", "CMYK", "HLS"])
slider_compression = st.slider("Select compression:", 0, 100, 10)
inputnum_derivatives = st.number_input("Number of derivatives:", value=1, step=1, min_value=1, max_value=10)
user_comment = st.text_area("Add a comment:", datetime.datetime.now().strftime("%d/%m/%Y - %H:%M"))

btn_example = st.button("show example")

if btn_example:
    if len(multiselect_box) == 0:
        st.warning("Please select at least one conversion model!")
        st.stop()

    if checkbox_advanced_opt:
        st.write("EXAMPLE with advanced options from ", radio_btn, " with model ", select_box, "on", inputnum_derivatives, "derivatives")
    else:
        st.write("EXAMPLE from ", radio_btn, " with model ", select_box, "on", inputnum_derivatives, "derivatives")

    st.write(multiselect_box)
    st.write("Compression: ", slider_compression)
    st.write("**Comment:** ", user_comment)


form1 = st.sidebar.form("options")
with form1:
    st.write("This is a form, that does not load until clicked!")
    selections = st.multiselect("Select different things:", ["A", "B", "C", "D"])
    st.form_submit_button("Save")
st.sidebar.write(selections)