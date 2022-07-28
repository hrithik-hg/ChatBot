# -*- coding: utf-8 -*-
"""BlenderBot_Chatbot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i0NFNYLc-1nfGM-ZhU-82JCogYbARNJr
"""

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

print("To end the converstaion with Bot type Bye")
while True:
  text = input("User: ")
  if text == 'Bye' or text == 'bye':
    print("Bot: Bye")
    break
  else:
    inputs = tokenizer(text, return_tensors = "pt")
    output = model.generate(**inputs)
    chatbot=tokenizer.decode(output[0])
    print("Bot: ",chatbot)