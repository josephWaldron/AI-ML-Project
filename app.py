from flask import Flask, render_template

app = Flask(__name__)

# Example of defining a route and acceptable methods
@app.route('/', methods=["GET", "POST"])
def view_home():
    return render_template(
        "index.html",

        # Example of passing variable to frontend from backend
        site_title="AI-ML Project"
    )


if __name__ == '__main__':
    debug = True
    app.debug = debug  # Toggle debugging mode
    app.run(debug=debug, port=5000)
