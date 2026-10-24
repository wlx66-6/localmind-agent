from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Chunk, Document, Source
from app.schemas import DocumentOut, ScanResult, SourceCreate, SourceOut
from app.services.ingest import SourceNotFoundError

router = APIRouter(prefix="/api/sources", tags=["sources"])


def _session_factory(request: Request) -> sessionmaker[Session]:
    factory: sessionmaker[Session] = request.app.state.db.session_factory
    return factory


def _source_out(session: Session, source: Source) -> SourceOut:
    document_count = session.scalar(
        select(func.count()).select_from(Document).where(Document.source_id == source.id)
    )
    chunk_count = session.scalar(
        select(func.count())
        .select_from(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.source_id == source.id)
    )
    return SourceOut(
        id=source.id,
        path=source.path,
        enabled=source.enabled,
        document_count=document_count or 0,
        chunk_count=chunk_count or 0,
        last_scan_at=source.last_scan_at,
    )


@router.get("", response_model=list[SourceOut])
def list_sources(request: Request) -> list[SourceOut]:
    with _session_factory(request)() as session:
        sources = session.scalars(select(Source).order_by(Source.id)).all()
        return [_source_out(session, source) for source in sources]


@router.post("", response_model=SourceOut, status_code=status.HTTP_201_CREATED)
def add_source(payload: SourceCreate, request: Request) -> SourceOut:
    raw = Path(payload.path.strip()).expanduser()
    if not raw.exists():
        raise HTTPException(status_code=400, detail=f"路径不存在：{raw}")
    path = raw.resolve()
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"不是文件夹：{path}")

    with _session_factory(request)() as session:
        existing = session.scalar(select(Source).where(Source.path == str(path)))
        if existing is not None:
            raise HTTPException(status_code=409, detail="该资料源已存在")
        source = Source(path=str(path))
        session.add(source)
        session.commit()
        session.refresh(source)
        return _source_out(session, source)


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(source_id: int, request: Request) -> None:
    store = request.app.state.store
    with _session_factory(request)() as session:
        source = session.get(Source, source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="资料源不存在")
        documents = session.scalars(select(Document).where(Document.source_id == source_id)).all()
        for document in documents:
            await store.delete_document(document.id)
        session.delete(source)
        session.commit()


@router.post("/{source_id}/scan", response_model=ScanResult)
async def scan_source(source_id: int, request: Request) -> ScanResult:
    try:
        stats = await request.app.state.ingest.scan_source(source_id)
    except SourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ScanResult(**asdict(stats))


@router.get("/{source_id}/documents", response_model=list[DocumentOut])
def list_documents(source_id: int, request: Request) -> list[DocumentOut]:
    with _session_factory(request)() as session:
        if session.get(Source, source_id) is None:
            raise HTTPException(status_code=404, detail="资料源不存在")
        documents = session.scalars(
            select(Document).where(Document.source_id == source_id).order_by(Document.rel_path)
        ).all()
        return [
            DocumentOut(
                id=document.id,
                rel_path=document.rel_path,
                size=document.size,
                indexed_at=document.indexed_at,
            )
            for document in documents
        ]
