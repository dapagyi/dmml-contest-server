from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase): ...


class ContestInfo(Base):
    __tablename__ = "contest_info"

    contest_id: Mapped[str] = mapped_column(primary_key=True)
    contest_name: Mapped[str]
    submission_rate_limit: Mapped[int]
    train_data_url: Mapped[str]
    test_data_url: Mapped[str]
    solution_file: Mapped[str]  # <-- added path to solution CSV


class User(Base):
    __tablename__ = "user"

    user_key: Mapped[str] = mapped_column(primary_key=True)
    full_name: Mapped[str]


class SubmissionORM(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    contest_id: Mapped[str]
    display_name: Mapped[str]
    user_key: Mapped[str]
    prediction: Mapped[str]  # stored as CSV string
    score: Mapped[float]
    timestamp: Mapped[datetime]


class ContestInfoDTO(BaseModel):
    contest_id: str
    contest_name: str
    submission_rate_limit: int
    train_data_url: str
    test_data_url: str
    solution_file: str

    class Config:
        from_attributes = True


class InitInfo(BaseModel):
    full_name: str
    contest_info: ContestInfoDTO


class InitRequest(BaseModel):
    user_key: str
    contest_id: str


@dataclass
class PredictionSubmitRequest:
    display_name: str
    user_key: str
    contest_id: str
    predictions: list[float]


class SubmissionDTO(BaseModel):
    id: int
    contest_id: str
    score: float
    timestamp: datetime
    display_name: str

    class Config:
        from_attributes = True


@dataclass
class FetchSubmissionsRequest:
    contest_id: str
