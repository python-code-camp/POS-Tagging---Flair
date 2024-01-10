from flair.data import Sentence
from flair.models import SequenceTagger

# Load the pre-trained POS tagging model
tagger = SequenceTagger.load("flair/pos-english")

user_input = "how to close my savings account"

# Create a 'Sentence' object
sentence = Sentence(user_input)

# predict the POS tags
tagger.predict(sentence)

print(sentence)

for entity in sentence:
    print(entity)
