import sys
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add module path and import queryearth
sys.path.append(r"D:\Code\esri-earth-search-engine\src")
import queryearth

app = FastAPI(title="ESRI Earth Search Engine API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qe = None

@app.on_event("startup")
def startup_event():
    """Runs once when Uvicorn starts."""
    global qe
    print("Initializing engine...")
    qe = queryearth.QueryEarth()
    qe.initialize()
    print("QueryEarth engine initialized successfully.")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    """Root health check endpoint."""
    return {"status": "online", "message": "QueryEarth API Server Running"}

@app.post("/api/init")
def api_init():
    """Health check for frontend boot sequence."""
    if qe is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    return {"status": "ok"}

@app.post("/api/predict")
def api_predict(req: QueryRequest):
    """Runs qe.predict(query) and returns GeoJSON."""
    if qe is None:
        raise HTTPException(status_code=500, detail="Engine not ready")
    
    try:
        # Run inference using your local model
        gdf = qe.find(req.query)
        print(gdf)
        
        # Calculate representative point (longitude/latitude) for mapping
        # if not gdf.empty:
        #     centroids = gdf.geometry.centroid
        #     gdf["_lon"] = centroids.x
        #     gdf["_lat"] = centroids.y
        
        geojson_data = json.loads(gdf.to_json())
        return geojson_data
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nServer stopped gracefully.")