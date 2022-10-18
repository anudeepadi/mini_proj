from transformers import pipeline

class Summarizer:

    def __init__(self, text):
        self.text = text
        self.summarizer = pipeline("summarization")
        self.summary = self.summarizer(self.text, max_length=100, min_length=30, do_sample=False)

    def get_summary(self):
        return self.summary
    
    def get_all(self):
        text = self.text
        summary = self.summary[0]['summary_text']
        per = 100 - round((len(summary)/len(text))*100)
        return text, summary, per
