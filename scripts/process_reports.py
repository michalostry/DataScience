import ufal.morphodita as morphodita

# Read the text from the extracted text file
text_path = '../data/csi_reports/inspection_report.txt'
with open(text_path, 'r', encoding='utf-8') as text_file:
    text = text_file.read()

# Load the morphological model
model_path = '../models/czech-morfflex2.0-pdtc1.0-220710/czech-morfflex2.0-pdtc1.0-220710-no_dia.tagger'
tagger = morphodita.Tagger.load(model_path)

if tagger is None:
    print("Failed to load the model")
else:
    print("Model loaded successfully")

    # Split the text into sentences (basic split for demonstration)
    sentences = text.split('.')

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            forms = morphodita.Forms()
            tokenizer = tagger.newTokenizer()
            tokenizer.setText(sentence)

            while tokenizer.nextSentence(forms):
                tagged_lemmas = morphodita.TaggedLemmas()
                tagger.tag(forms, tagged_lemmas)

                for i in range(len(forms)):
                    print(f"{forms[i]} -> Lemma: {tagged_lemmas[i].lemma}, Tag: {tagged_lemmas[i].tag}")
