from TextPreprocessor import TextPreprocessor

test_text = """Hello World. This is a test sentence. This is a number 0 1 25 -0.343 0..4534 100km sentence </html> <p>This txt is in an html term</p>
What about a url like this https://google.com 
"""
processor = TextPreprocessor(type="lemm")
processed_text = processor._preprocess_text(test_text)
print(processed_text)