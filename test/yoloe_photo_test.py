from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("model\yoloe-11m-seg-pf.pt")

# Execute prediction for specified categories on an image
results = model.predict("img/test.png")

# Show results
results[0].show()