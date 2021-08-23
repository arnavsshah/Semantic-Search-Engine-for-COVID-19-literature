from flask import Flask, render_template, request
# import semantic_search

app = Flask(__name__)

@app.route("/search", methods=["POST","GET"])
def search():
    print(request.args.get("searchdata"))
    # res = semantic_search.search(("what is the cause of diseases?"))
    # print(res)
    data = [
      {
         "title":"hello1",
         "abstract":"hello1",
         "link":"hello1",
         "authors":" hello1"
      },
      {
         "title":"hello2",
         "abstract":"hello2",
         "link":"hello2",
         "authors":" hello2"
      },
      {
         "title":"hello3",
         "abstract":"hello3",
         "link":"hello3",
         "authors":" hello3"
      },
      {
         "title":"hello4",
         "abstract":"hello4",
         "link":"hello4",
         "authors":"hello4"
         }]
    return render_template("search.html",papers=data)

if __name__ == "__main__":
    app.run(debug=True)

