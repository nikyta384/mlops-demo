import bentoml
from bentoml.io import JSON
import os
from pydantic import BaseModel


# with open(os.getenv('VERSION_FILE'), 'r') as m:
#     model_tag = m.read()
MODEL_TAG = 'none'
model_name = "my-model"
# model_tag = '09'
# Load the model (using 'model:latest' or a specific tag)
model_ref = bentoml.sklearn.get(f"{model_name}:{MODEL_TAG}")
model_runner = model_ref.to_runner()

# Create BentoML service
svc = bentoml.Service(model_name, runners=[model_runner])

# Define input schema using Pydantic
class TweetAnalysis(BaseModel):
    text: str

# API endpoint to analyze tweets
@svc.api(input=JSON(pydantic_model=TweetAnalysis), output=JSON())
async def analyse_tweet(data: TweetAnalysis):
    input_data = [data.text]
    prediction = await model_runner.predict.async_run(input_data)
    
    # Include model name and tag in the response
    return {
        "analysed_tweet": prediction[0],
        "model_name": model_ref.tag.name,  # e.g., 'model'
        "model_tag": str(model_ref.tag),    # e.g., 'model:latest'
        "model_labels": model_ref.info.labels,
    }