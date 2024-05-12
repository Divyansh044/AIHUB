import pytesseract

from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img=Image.open('ml_models/HWR/validation/VALIDATION_41369.jpg')
extracted_text = pytesseract.image_to_string(img)
print(extracted_text)