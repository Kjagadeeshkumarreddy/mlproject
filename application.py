from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import CoustomData,PredictPipline

application=Flask(__name__)
app=application

@app.route('/',methods=["GET","POST"])
def predict():
    if(request.method=="GET"):
        return render_template("index.html")
    else:
        data=CoustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )   
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipline =PredictPipline()
        result=predict_pipline.predict(pred_df)
        print(result)
        return render_template("index.html",results=result[0])


if __name__=="__main__":
    app.run(host="0.0.0.0")