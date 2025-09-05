from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("model/yoloe-11m-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt
names = ["person", ]
model.set_classes(names, model.get_text_pe(names))

# Execute prediction for specified categories on an image
results = model.predict("img/test.png")

# Show results
results[0].show()