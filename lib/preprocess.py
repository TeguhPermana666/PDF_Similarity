import re
def preprocess_text(text):
    #Remove the special text 
    text = re.sub(r"[^\w\s]","",text)
    #Remove the multiple spaces 
    text = re.sub(r"\s+"," ",text)
    # make all is lower
    text = text.lower()
    return text.strip()