from flask import Flask, render_template
import semantic_search

app = Flask(__name__)

@app.route("/")
def search():
    
    res = semantic_search.search(("what is the cause of diseases?"))
    print(res)
    return render_template("search.html")

if __name__ == "__main__":
    app.run(debug=True)

