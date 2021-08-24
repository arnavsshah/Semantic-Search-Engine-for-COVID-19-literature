from flask import Flask, render_template, request
import json
from search import semantic_search

app = Flask(__name__)

@app.route("/search", methods=["GET"])
def search():

   query = request.args.get("searchdata")
   if query is not None :
      df_res = semantic_search.search(request.args.get("searchdata"), request.args.get('category'))
      json_res = df_res.to_json(orient="split")
      search_results = json.loads(json_res)

      return render_template("search.html", query=query, papers=search_results["data"], columns=search_results["columns"], category=request.args.get('category'))
   
   else :
       return render_template("search.html")

if __name__ == "__main__":
    app.run(debug=True)

