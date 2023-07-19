#@title Default title text
from PIL import Image
import requests
from io import BytesIO

from transformers import CLIPProcessor, CLIPModel
from google.colab import files

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

uploaded_files = files.upload()
uploaded_image = list(uploaded_files.values())[0]
image = Image.open(BytesIO(uploaded_image))

inputs = processor(text=["very expensive", "expensive", "average", "inexpensive", "very inexpensive"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
labels = ["very expensive", "expensive", "average", "inexpensive", "very inexpensive"]
for i, prob in enumerate(probs[0]):
   label = labels[i]
   confidence = prob.item()
   print(f"Label: {label}, Confidence: {confidence}")

# ************************ RANK THE INDEX************************************************

print("*** Rank lables ***")
label_probs = [(label, prob.item()) for label, prob in zip(labels, probs[0])]
label_probs_sorted = sorted(label_probs, key=lambda x: x[1], reverse=True)
for label, confidence in label_probs_sorted:
    print(f"Label: {label}, Confidence: {confidence}")

# ************************ CALCULATE THE VALUE************************************************

label_values = [5, 4, 3, 2, 1]
total_index = 0
for label, confidence in label_probs_sorted:
    label_index = label_values[labels.index(label)]
    total_index += label_index * confidence
print("Label-Value Pairs:")
for label, confidence in label_probs_sorted:
    print(f"Label: {label}, Confidence: {confidence}")

print(f"\nTotal Index: {total_index}")

# ***************** COMMENT AND CLASSIFICATION ************************
print("This product is likley:")
if total_index >= 4:
  print("Very expensive")
if total_index <4 and total_index >= 3:
  print("expensive")
if total_index <3 and total_index >= 2:
  print("average")
if total_index <2 and total_index >= 1:
  print("inexpensive")
if total_index < 1:
  print("inexpensive")

# ***************** DESCRIPTION************************

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


# conditional image captioning
text = "*** --> This is :"
inputs = processor(image, text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
