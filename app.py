import os
import sys
import json
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# Ensure stdout flushes immediately so all prints appear in the terminal right away
sys.stdout.reconfigure(line_buffering=True)

sys.path.append(r"/home/susanket/esri-earth-search-engine/src")
import queryearth

app = FastAPI(title="ESRI Earth Search Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engines: Dict[str, queryearth.QueryEarth] = {}

class InitRequest(BaseModel):
    api_key: str

class QueryRequest(BaseModel):
    query: str
    api_key: Optional[str] = None

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/api/init")
async def api_init(req: InitRequest):
    key = req.api_key.strip() if req.api_key else ""
    if not key:
        raise HTTPException(status_code=400, detail="API key is required.")

    try:
        os.environ['GEMINI_API_KEY'] = key
        
        def _init_engine():
            print(f"\n[INIT] Initializing QueryEarth engine for key ...{key[-6:]}...", flush=True)
            engine = queryearth.QueryEarth()
            engine.initialize()
            print("[INIT] Engine initialization complete!", flush=True)
            return engine

        engines[key] = await run_in_threadpool(_init_engine)
        return {"status": "ok", "message": "Engine initialized successfully"}
    except Exception as e:
        print(f"[INIT ERROR] Failed to initialize engine: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize engine: {str(e)}")

@app.post("/api/predict")
async def api_predict(req: QueryRequest):
    key = req.api_key.strip() if req.api_key else os.environ.get('GEMINI_API_KEY', '')

    engine = engines.get(key)
    if engine is None:
        if engines:
            engine = list(engines.values())[-1]
        else:
            raise HTTPException(status_code=400, detail="Engine is not initialized.")

    # Re-apply key to environment for this thread's context
    if key:
        os.environ['GEMINI_API_KEY'] = key

    print(f"\n[QUERY START] Received prompt: '{req.query}'", flush=True)

    try:
        def _run_find():
            print(f"[PREDICT] Invoking engine.find('{req.query}')...", flush=True)
            results_gdf = engine.find(req.query)
            print(f"[PREDICT] engine.find finished. Found {len(results_gdf) if results_gdf is not None else 0} features.", flush=True)
            return results_gdf

        gdf = await run_in_threadpool(_run_find)

        if gdf is None or gdf.empty:
            print("[WARN] GeoDataFrame returned by engine.find() is empty.", flush=True)
            return {"type": "FeatureCollection", "features": []}

        geojson_data = json.loads(gdf.to_json())
        print(f"[SUCCESS] Returning {len(geojson_data.get('features', []))} GeoJSON features to frontend.", flush=True)
        return geojson_data

    except Exception as e:
        print(f"[ERROR] Exception during query prediction: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)