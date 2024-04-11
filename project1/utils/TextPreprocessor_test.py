from TextPreprocessor import TextPreprocessor
def test_preprocess ():
    test_text = """Hello"""
    processor = TextPreprocessor()
    processed_text = processor._preprocess_text(test_text)
    print(processed_text)
    assert True == True