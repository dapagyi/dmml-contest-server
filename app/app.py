from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from litestar import Litestar, Request, post
from litestar.config.cors import CORSConfig
from litestar.exceptions import ClientException, HTTPException, NotFoundException
from litestar.exceptions.http_exceptions import TooManyRequestsException
from litestar.plugins.sqlalchemy import SQLAlchemyAsyncConfig, SQLAlchemyPlugin
from litestar.static_files import create_static_files_router
from litestar.status_codes import HTTP_200_OK, HTTP_409_CONFLICT, HTTP_429_TOO_MANY_REQUESTS
from sklearn.metrics import f1_score
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.config import CONTESTS, USERS
from app.models import (
    Base,
    ContestInfo,
    ContestInfoDTO,
    ContestInfoRequest,
    FetchSubmissionsRequest,
    InitInfo,
    InitRequest,
    PredictionSubmitRequest,
    SubmissionDTO,
    SubmissionORM,
    User,
)

RATE_LIMIT_WINDOW = timedelta(minutes=1)


async def provide_transaction(db_session: AsyncSession):
    try:
        async with db_session.begin():
            yield db_session
    except IntegrityError as exc:
        raise ClientException(status_code=HTTP_409_CONFLICT, detail=str(exc)) from exc


async def get_contest_info(contest_id: str, transaction: AsyncSession) -> ContestInfo:
    query = select(ContestInfo).where(ContestInfo.contest_id == contest_id)
    result = await transaction.execute(query)
    try:
        return result.scalar_one()
    except NoResultFound as e:
        raise NotFoundException(detail=f"Contest {contest_id!r} not found") from e


async def get_user_info(user_key: str, transaction: AsyncSession) -> User:
    query = select(User).where(User.user_key == user_key)
    result = await transaction.execute(query)
    try:
        return result.scalar_one()
    except NoResultFound as e:
        raise NotFoundException(detail=f"User {user_key!r} not found") from e


async def check_rate_limit(user_key: str, contest: ContestInfo, transaction: AsyncSession):
    since = datetime.now(timezone.utc) - RATE_LIMIT_WINDOW
    query = (
        select(func.count())
        .select_from(SubmissionORM)
        .where(
            SubmissionORM.user_key == user_key,
            SubmissionORM.contest_id == contest.contest_id,
            SubmissionORM.timestamp > since,
        )
    )
    result = await transaction.execute(query)
    count = result.scalar_one()
    if count >= contest.submission_rate_limit:
        raise TooManyRequestsException(detail="Submission rate limit exceeded", status_code=HTTP_429_TOO_MANY_REQUESTS)


async def save_prediction(
    display_name: str,
    user_key: str,
    contest: ContestInfo,
    prediction_csv: str,
    score: float,
    transaction: AsyncSession,
):
    submission = SubmissionORM(
        display_name=display_name,
        user_key=user_key,
        contest_id=contest.contest_id,
        prediction=prediction_csv,
        score=score,
        timestamp=datetime.now(timezone.utc),
    )
    transaction.add(submission)


def evaluate_prediction(predictions: pd.DataFrame, contest: ContestInfo) -> float:
    solution_df = pd.read_csv(contest.solution_file)
    if predictions.shape != solution_df.shape:
        raise HTTPException(
            status_code=400,
            detail=f"Shape mismatch: expected {solution_df.shape}, got {predictions.shape}.",
        )
    try:
        return f1_score(solution_df.values, predictions.values)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid prediction.") from exc


@post("/contest-info", status_code=HTTP_200_OK)
async def get_contest_info_endpoint(data: ContestInfoRequest, transaction: AsyncSession) -> ContestInfoDTO:
    contest_info = await get_contest_info(data.contest_id, transaction)
    return ContestInfoDTO.model_validate(contest_info)


@post("/contest", status_code=HTTP_200_OK)
async def get_init_info(data: InitRequest, transaction: AsyncSession, headers: dict) -> InitInfo:
    contest_info = await get_contest_info(data.contest_id, transaction)
    user_info = await get_user_info(data.user_key, transaction)
    return InitInfo(
        full_name=user_info.full_name,
        contest_info=ContestInfoDTO.model_validate(contest_info),
    )


@post("/submit", status_code=HTTP_200_OK)
async def submit_prediction(request: Request, data: PredictionSubmitRequest, transaction: AsyncSession) -> dict:
    contest = await get_contest_info(data.contest_id, transaction)
    await get_user_info(data.user_key, transaction)

    await check_rate_limit(data.user_key, contest, transaction)

    try:
        predictions_df = pd.DataFrame(data.predictions)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid predictions format") from exc

    score = evaluate_prediction(predictions_df, contest)
    await save_prediction(
        data.display_name,
        data.user_key,
        contest,
        predictions_df.to_csv(index=False),
        score,
        transaction,
    )

    request.logger.info(f"User {data.user_key} scored {score} on {data.contest_id}")
    return {"score": score}


@post("/top-submissions")
async def fetch_top_submissions(data: FetchSubmissionsRequest, transaction: AsyncSession) -> list[SubmissionDTO]:
    query = (
        select(SubmissionORM)
        .where(SubmissionORM.contest_id == data.contest_id)
        .order_by(SubmissionORM.score.desc(), SubmissionORM.timestamp.asc())
        .limit(500)
    )
    result = await transaction.execute(query)
    submissions = result.scalars().all()
    return [SubmissionDTO.model_validate(s) for s in submissions]


@post("/top-submissions-per-user")
async def fetch_top_submissions_per_user(
    data: FetchSubmissionsRequest, transaction: AsyncSession
) -> list[SubmissionDTO]:
    rank_col = (
        func.row_number()
        .over(
            partition_by=SubmissionORM.display_name,
            order_by=[SubmissionORM.score.desc(), SubmissionORM.timestamp.asc()],
        )
        .label("rnk")
    )

    subq = select(SubmissionORM, rank_col).where(SubmissionORM.contest_id == data.contest_id).subquery()

    submission_alias = aliased(SubmissionORM, subq)

    query = (
        select(submission_alias).where(subq.c.rnk == 1).order_by(subq.c.score.desc(), subq.c.timestamp.asc()).limit(500)
    )

    result = await transaction.execute(query)
    submissions = result.scalars().all()
    return [SubmissionDTO.model_validate(s) for s in submissions]


cors_config = CORSConfig(
    allow_origins=[
        "http://localhost:8000",
        "https://apagyidavid.web.elte.hu",
        "https://dmml.dapagyi.dedyn.io",
    ]
)

db_config = SQLAlchemyAsyncConfig(
    connection_string="sqlite+aiosqlite:///data/dmml_contest.sqlite",
    metadata=Base.metadata,
    create_all=True,
    before_send_handler="autocommit",
)


async def on_startup(app: Litestar) -> None:
    """Load initialization data on first run."""
    async with db_config.get_session() as session, session.begin():
        for contest_id, info in CONTESTS.items():
            contest = ContestInfo(
                contest_id=contest_id,
                contest_name=info["contest_name"],
                submission_rate_limit=info["submission_rate_limit"],
                train_data_url=info["train_data_url"],
                test_data_url=info["test_data_url"],
                solution_file=info["solution_file"],
            )
            await session.merge(contest)

        for user_key, info in USERS.items():
            user = User(user_key=user_key, full_name=info["full_name"])
            await session.merge(user)

    app.logger.info("Initialization data loaded.")


app = Litestar(
    route_handlers=[
        get_contest_info_endpoint,
        get_init_info,
        submit_prediction,
        fetch_top_submissions,
        fetch_top_submissions_per_user,
        create_static_files_router(path="/static", directories=["static"]),
    ],
    dependencies={"transaction": provide_transaction},
    plugins=[SQLAlchemyPlugin(db_config)],
    cors_config=cors_config,
    on_startup=[on_startup],
)
