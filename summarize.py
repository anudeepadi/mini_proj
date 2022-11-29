from transformers import pipeline
from transformers import BertTokenizerFast, EncoderDecoderModel
import logging
import torch
import time
import pymongo
import re
from simplet5 import SimpleT5
# import re
# from bleu import list_bleu

logger = logging.getLogger(__name__)


class Summarizer:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.newsum = pipeline("summarization", model='it5/it5-efficient-small-el32-news-summarization', tokenizer='it5/it5-efficient-small-el32-news-summarization', device=self.device)
        self.newsum = pipeline("summarization", model='it5/it5-efficient-small-el32-news-summarization')
        self.myclient = pymongo.MongoClient("mongodb+srv://m001-student:Factory2$@cluster0.qxiwj.mongodb.net/?retryWrites=true&w=majority")
        self.mydb = self.myclient["minor_project"]
        self.mycol = self.mydb["articles"]

    def get_summary(self, text, type):
        text = self.preprocess(text)
        start = time.time()
        if type == "bert":
            summary_text = self.get_bert_summary(text)
        elif type == "t5":
            summary_text = self.get_t5_summary(text)
        elif type == "newsum":
            summary_text = self.newsum(text)[0]['summary_text']
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

    def preprocess(self, text):
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub('"','', text)
        text = ' '.join(text.split())
        return text

    def get_bert_summary(self, text):
        tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(tokenizer, tokenizer)
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.max_length = 142
        model.config.min_length = 56
        model.config.no_repeat_ngram_size = 3
        model.config.early_stopping = True
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=1024)
        summary_ids = model.generate(input_ids)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def get_t5_summary(self, text): 
        model = SimpleT5()
        model.from_pretrained("t5", "t5-base")
        return model.predict(text)

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
