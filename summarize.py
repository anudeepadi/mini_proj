from transformers import pipeline
from transformers import BertTokenizerFast, EncoderDecoderModel
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TFPegasusForConditionalGeneration
import logging
import torch
import time
# import re
# from bleu import list_bleu

logger = logging.getLogger(__name__)


class Summarizer:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
        self.model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization').to(self.device)
        #self.stf_nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')
        self.article_summarizer = pipeline("summarization", "pszemraj/long-t5-tglobal-base-16384-book-summary", device=0 if torch.cuda.is_available() else -1)
        self.stock_model_name = "human-centered-summarization/financial-summarization-pegasus"
        self.stock_tokenizer = PegasusTokenizer.from_pretrained(self.stock_model_name)
        self.stock_model = PegasusForConditionalGeneration.from_pretrained(self.stock_model_name) # If you want to use the Tensorflow model 
                                         


    def get_summary(self, text, type):
        start = time.time()
        if type.lower() == "article":
            summary_text = self.article_summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        else:
            inputs = self.tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            output = self.model.generate(input_ids, attention_mask=attention_mask)
            summary_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            #summary_text = self.truecasing(summary_text)
        end = time.time()
        return {
            "Original Text": text,
            "Summary": summary_text,
            "Time Taken": end - start,
            "Length before Summarization": len(text),
            "Length after Summarization": len(summary_text),
            "Percentage Reduction": 100 - round((len(summary_text)/len(text))*100)
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
