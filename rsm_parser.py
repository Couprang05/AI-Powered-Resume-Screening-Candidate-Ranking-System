import spacy
import PyPDF2
import io

nlp = spacy.load("en_core_web_sm")

def ext_txt_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)  # Directly use file object
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def ext_rsm_dtl(text):
    doc = nlp(text)
    dtls = {
        "skills": [],
        "experience": [],
        "education": []
    }

    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"]:
            dtls["education"].append(ent.text)
        elif ent.label_ == "DATE":
            dtls["experience"].append(ent.text)
        elif ent.label_ in ["SKILL", "LANGUAGE"]:
            dtls["skills"].append(ent.text)

    return dtls
