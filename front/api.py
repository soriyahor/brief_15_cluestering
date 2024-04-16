from fastapi import FastAPI, Request
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import httpx


# Instance du moteur de modèles Jinja2 pour la gestion des templates HTML
templates = Jinja2Templates(directory="templates")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

URL_BACKEND = "soriyab15-fastfront.francecentral.azurecontainer.io"
# URL_BACKEND = "172.21.0.2"
#URL_BACKEND = "localhost"


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })


@app.post("/choose_model/{model_name}")
async def run_prediction(n_clusters: int, model_name = str):
    if model_name == "Kmeans":
    # Send the data to the receiver API
        async with httpx.AsyncClient() as client:
            response = await client.post(f"http://{URL_BACKEND}:8001/prediction_kmeans", params={"n_clusters": n_clusters})

            return response.text

    if model_name == "agglo":
    # Send the data to the receiver API
        async with httpx.AsyncClient() as client:
            response = await client.post(f"http://{URL_BACKEND}:8001/prediction_agglo", params={"n_clusters": n_clusters})
            
            return response.text
        
        
    
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
    
