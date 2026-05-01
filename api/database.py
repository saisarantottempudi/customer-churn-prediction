"""
Database layer for persisting churn predictions using SQLAlchemy + SQLite.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./churn_predictions.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True, nullable=False)
    churn_prediction = Column(String, nullable=False)
    churn_probability = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    recommended_action = Column(Text, nullable=True)
    top_reasons = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    with Session(engine) as session:
        yield session


def log_prediction(db: Session, prediction: dict) -> PredictionLog:
    import json
    record = PredictionLog(
        customer_id=prediction["customer_id"],
        churn_prediction=prediction["churn_prediction"],
        churn_probability=prediction["churn_probability"],
        risk_level=prediction["risk_level"],
        recommended_action=prediction.get("recommended_action", ""),
        top_reasons=json.dumps(prediction.get("top_reasons", [])),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
