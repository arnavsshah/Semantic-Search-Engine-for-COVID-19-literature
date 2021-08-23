from flask import Flask, render_template, request
import json
import semantic_search

app = Flask(__name__)

@app.route("/search", methods=["GET"])
def search():
   #  print(request.args.get("searchdata"))
   query = request.args.get("searchdata")
   if query is not None :
      df_res = semantic_search.search(request.args.get("searchdata"))
      json_res = df_res.to_json(orient="split")
      search_results = json.loads(json_res)
   
#  "columns": [ "cord_uid", "title", "doi", "pubmed_id", "license", 
#               "abstract", "publish_time", "authors", "journal", "pdf_json_files", "url", "body_text"]

      return render_template("search.html", papers=search_results["data"], columns=search_results["columns"])
   
   else :
       return render_template("search.html")

if __name__ == "__main__":
    app.run(debug=True)

