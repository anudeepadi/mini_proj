import streamlit as st
from summarize import Summarizer


def main():
    st.title("Text Summarizer")
    st.subheader("Enter the text you want to summarize")
    text = st.text_area("Enter the text here")
    model = st.selectbox("Select the model", ["News", "Article/Blog", "Research Paper", "Stock Market"])
    if st.button("Summarize"):
        summarizer = Summarizer()
        summary = summarizer.get_summary(text, model)
        st.write("Original Text: ", summary["Original Text"])
        st.write("Summarized Text: ", summary["Summary"])
        st.write("Original Text Length:", summary["Length before Summarization"])
        st.write("Summary Length:", summary["Length after Summarization"])
        st.write("Percentage Reduction:", summary["Percentage Reduction"])
        st.write("Time Taken:", summary["Time Taken"])
        st.success("Done!")

if __name__ == "__main__":
    main()