from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import io
import shutil
import os
import logging
import pandas as pd
from datetime import timedelta
from io import StringIO
import sys
# Ensure local backend modules are importable when running as a package
sys.path.append(os.path.dirname(__file__))

# Import ML services (they live in this same backend package)
from ml_service import run_analysis
from ml_service_intelligent import run_intelligent_analysis

from .auth import (create_access_token, get_password_hash, verify_password, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES)
from .database import get_db, engine, SessionLocal, Base

from .models import User, Dataset, Result
from .schemas import UserCreate, Token, DatasetCreate, ResultOut
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="User Behavior Intelligence Platform", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserLogin(BaseModel):
    username: str
    password: str

@app.get("/")
async def root():
    return {"message": "User Behavior Intelligence API v2 (with Auth). Use /auth/register, /auth/login, then /analyze"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/auth/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_pw)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"msg": "User created", "user_id": db_user.id}

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/analyze", response_model=dict)
async def analyze(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files supported")
    
    # Save file
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{current_user.id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        content = open(file_path, 'rb')
        results = run_analysis(io.BytesIO(content.read()))
        
        # Save to DB
        dataset = Dataset(user_id=current_user.id, filename=file.filename, file_path=file_path)
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        result = Result(
            dataset_id=dataset.id,
            metrics=results["final_metrics"],
            clusters=results["clusters"][:100],  # Truncate for DB
            rules=results["rules"],
            insights="; ".join(results["insights"])
        )
        db.add(result)
        db.commit()
        
        results["result_id"] = result.id
        return results
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.get("/results/{result_id}", response_model=ResultOut)
async def get_result(result_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result or result.dataset.user_id != current_user.id:
        raise HTTPException(404, "Result not found")
    return result

@app.post("/analyze-intelligent", tags=["Clustering"])
async def analyze_data_intelligent(file: UploadFile = File(...)):
    """
    🔥 INTELLIGENT ANALYSIS - AutoML + Hybrid Metrics + ICSO
    
    Features:
    - Auto-selects best algorithm (KMeans, DBSCAN, or Hierarchical)
    - Calculates hybrid evaluation score
    - Computes novel ICSO metric
    - Generates cluster profiles
    - Detects anomalies
    - Provides recommendations
    """
    try:
        if not file.filename.endswith('.csv'):
            raise ValueError("Only CSV files supported")
        
        logger.info(f"Intelligent analysis started for file: {file.filename}")
        
        # Read file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode()))
        
        logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate required columns
        required_cols = ['InvoiceNo', 'user_id', 'quantity', 'price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Required: {required_cols}")
        
        # 🔥 RUN INTELLIGENT ANALYSIS
        results = run_intelligent_analysis(df)
        
        logger.info(f"Analysis complete. Algorithm: {results['best_algorithm']}")
        
        # Format response
        return {
            "status": "success",
            "message": "Intelligent analysis completed successfully",
            
            # AutoML Algorithm Selection
            "algorithm": {
                "selected": results['best_algorithm'],
                "scores": {
                    "kmeans": float(results['algorithm_scores'].get('kmeans', 0)),
                    "dbscan": float(results['algorithm_scores'].get('dbscan', 0)),
                    "hierarchical": float(results['algorithm_scores'].get('hierarchical', 0))
                }
            },
            
            # Quality Metrics
            "metrics": {
                "silhouette": float(round(results['final_metrics'].get('silhouette', 0), 4)),
                "davies_bouldin": float(round(results['final_metrics'].get('davies_bouldin', 0), 4)),
                "calinski_harabasz": float(round(results['final_metrics'].get('calinski_harabasz', 0), 4)),
                "icso_score": float(round(results['final_metrics'].get('icso_score', 0), 4))
            },
            
            # Cluster Profiles
            "clusters": [
                {
                    "id": p.get('cluster_id', idx),
                    "label": p.get('business_label', 'Unknown'),
                    "size": p.get('size', 0),
                    "percentage": float(round(p.get('percentage', 0), 1)),
                    "avg_spending": float(round(p.get('characteristics', {}).get('total_spent', {}).get('mean', 0), 2)) if 'characteristics' in p and 'total_spent' in p['characteristics'] else 0,
                    "avg_purchases": float(round(p.get('characteristics', {}).get('purchase_count', {}).get('mean', 0), 1)) if 'characteristics' in p and 'purchase_count' in p['characteristics'] else 0,
                    "avg_engagement": float(round(p.get('characteristics', {}).get('session_time', {}).get('mean', 0), 1)) if 'characteristics' in p and 'session_time' in p['characteristics'] else 0
                }
                for idx, p in enumerate(results.get('cluster_profiles', []))
            ],
            
            # Anomaly Detection
            "anomalies": {
                "count": results.get('anomalies_count', 0),
                "percentage": float(results.get('anomalies_percentage', 0))
            },
            
            # Recommendations
            "recommendations": [
                {
                    "cluster_id": int(rec.get('cluster_id', idx)),
                    "cluster_label": str(rec.get('cluster_label', 'Unknown')),
                    "recommendations": [str(r) for r in rec.get('recommendations', [])]
                }
                for idx, rec in enumerate(results.get('recommendations', []))
            ]
        }
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

