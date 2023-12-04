import os.path

from flask import Flask, render_template, request, jsonify

template_dir = os.path.abspath("website/templates")
static_dir = os.path.abspath("website/static")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

SITE_TITLE = "AI-ML Project"


# Example of defining a route and acceptable methods
@app.route('/', methods=["GET", "POST"])
def view_home():
    return render_template(
        "index.html",

        # Example of passing variable to frontend from backend
        site_title=SITE_TITLE,
        classification="Waiting on Input..."
    )


@app.route('/predict', methods=["POST"])
def predict():
    image_file = request.form["imageFile"]
    model_choice = request.form["modelChoice"]

    # Get result based on model choice
    if model_choice == "pytorch":
        classification = get_pytorch_prediction(image_file)
    elif model_choice == "tensorflow":
        classification = get_tf_prediction(image_file)
    else:
        classification = get_yolo_prediction(image_file)

    # Temp until models are added
    classification = "Bird"
    return render_template("index.html",
                           site_title=SITE_TITLE,
                           classification=classification)


def get_pytorch_prediction(image_file):
    pass


def get_tf_prediction(image_file):
    pass


def get_yolo_prediction(image_file):
    pass


if __name__ == '__main__':
    debug = True
    app.debug = debug  # Toggle debugging mode
    app.run(debug=debug, port=5000)
