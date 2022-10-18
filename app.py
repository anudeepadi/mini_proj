import streamlit as st
from summarize import Summarizer


def main():
    st.title("Text Summarizer")
    st.subheader("Enter the text you want to summarize")
    text = st.text_area("Enter the text here")
    if st.button("Summarize"):
        summarizer = Summarizer(text)
        summary = summarizer.get_summary()
        st.write(summary[0]['summary_text'])
        st.write(f"Original text length: {len(text)}")
        st.write(f"Summary length: {len(summary[0]['summary_text'])}")
        st.write(f"Summary is {100 - round((len(summary[0]['summary_text'])/len(text))*100)}% of the original text")
        

if __name__ == "__main__":
    main()