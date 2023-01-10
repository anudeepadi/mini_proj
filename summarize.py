from transformers import pipeline, BertTokenizerFast, EncoderDecoderModel
import transformers
import logging
import torch
import time
import pymongo
import re
from simplet5 import SimpleT5
# import re
# from bleu import list_bleu
import summarizer

logger = logging.getLogger(__name__)


class Summarizer:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.newsum = pipeline("summarization", model='it5/it5-efficient-small-el32-news-summarization', tokenizer='it5/it5-efficient-small-el32-news-summarization')
        self.myclient = pymongo.MongoClient("mongodb+srv://m001-student:Factory2$@cluster0.qxiwj.mongodb.net/?retryWrites=true&w=majority")
        self.mydb = self.myclient["minor_project"]
        self.mycol = self.mydb["articles"]

    def get_summary(self, text, type):
        text = self.preprocess(text)
        start = time.time()
        if type == "newsum":
            summary_text = self.newsum(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        elif type == "t5":
            summary_text = self.summarize_t5(text)
        elif type == "bert":
            summary_text = self.summarize_bert(text)
        else:
            summary_text = "Invalid type"
        end = time.time()
        try:
            self.sendToMongo(text, summary_text)
        except:
            print("Error sending to Mongo")
        return {
            "Original Text": text,
            "Summary": summary_text,
            "Time Taken": end - start,
            "Length before Summarization": len(text),
            "Length after Summarization": len(summary_text),
            "Percentage Reduction": 100 - round((len(summary_text)/len(text))*100)
        }

    # def summarize_t5(text, max_length=30):
    #     model = transformers.T5ForConditionalGeneration.from_pretrained('t5-base')
    #     input_ids = transformers.tokenizer.encode(text, return_tensors='pt')
    #     output = model.generate(input_ids, max_length=max_length, num_beams=2, early_stopping=True)
    #     summary = transformers.tokenizer.decode(output[0], skip_special_tokens=True)
    #     return summary

    
    # def summarize_bert(text, max_length=30):
    #     input_ids = transformers.tokenizer.encode(text, return_tensors='pt')
    #     model = transformers.BertModel.from_pretrained('bert-base-uncased')
    #     _, output = model(input_ids)
    #     summary = transformers.tokenizer.decode(output[0], skip_special_tokens=True)
    #     return summary[:max_length]

    def sendToMongo(self, text, summary):
        mydict = {"text": text, "summary": summary}
        x = self.mycol.insert_one(mydict)
        print(x.inserted_id)

    def getFromMongo(self):
        docs = self.mycol.find()
        return {
            "collections":[
                {"text": doc["text"], "summary": doc["summary"]} for doc in docs
            ]
        }
    
    def preprocess(self, text):
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub('"','', text)
        text = ' '.join(text.split())
        return text

    # def truecasing(self, input_text):
    #     sentences = sent_tokenize(input_text, language='english')
    #     sentences_capitalized = [s.capitalize() for s in sentences]
    #     text_truecase = re.sub(" (?=[\.,'!?:;])", "", ' '.join(sentences_capitalized))
    #     doc = self.stf_nlp(text_truecase)
    #     text_truecase =  ' '.join([w.text.capitalize() if w.upos in ["PROPN","NNS"] \
    #                                                 else w.text for sent in doc.sentences \
    #                             for w in sent.words])
    #     text_truecase = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text_truecase)
    #     return text_truecase
