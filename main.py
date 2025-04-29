from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os

from processing import process_file, process_one_file, process_good_11clusters

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

USERNAME = "GKphysics"
PASSWORD = "GKphysics"

# ----------------- ROUTES --------------------

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == USERNAME and password == PASSWORD:
        return RedirectResponse("/upload", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# --- process_file (double sigmoid) ---
@app.post("/process", response_class=HTMLResponse)
async def process_upload(request: Request, 
                         file: UploadFile = File(...),
                         plot_title: str = Form("Beam Profile with FWHM and Penumbra"),
                         x_label: str = Form("Position"),
                         y_label: str = Form("Intensity")):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = process_file(file_path, plot_title, x_label, y_label)
    return templates.TemplateResponse("result.html", {"request": request, "results": results})

# --- process_one_file (single profile, simple plot) ---
@app.post("/process_single", response_class=HTMLResponse)
async def process_single(request: Request,
                         file: UploadFile = File(...),
                         plot_title: str = Form("Single Profile"),
                         x_label: str = Form("Position"),
                         y_label: str = Form("Intensity")):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = process_one_file(file_path, plot_title, x_label, y_label)
    return templates.TemplateResponse("result.html", {"request": request, "results": results})

# --- process batch of profiles ---
from typing import List
from fastapi import UploadFile, File

@app.post("/process_batch", response_class=HTMLResponse)
async def process_batch(request: Request, files: List[UploadFile] = File(...)):
    all_results = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        results = process_one_file(file_path)
        all_results.append({"filename": file.filename, "results": results})

    return templates.TemplateResponse("batch_results.html", {"request": request, "all_results": all_results})

# --- process_good_11clusters (MCU special mode) ---
@app.post("/good11", response_class=HTMLResponse)
async def good_11_handler(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = process_good_11clusters(file_path)
    return templates.TemplateResponse("good11_results.html", {"request": request, "results": results})
