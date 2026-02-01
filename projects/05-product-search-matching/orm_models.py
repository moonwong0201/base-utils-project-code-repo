import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(500), nullable=False)
    title = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow
    )

    milvus_primary_key = Column(Integer, nullable=True)

    __table_args__ = (
        Index('idx_title', 'title'),
    )

    def __repr__(self):
        return f"<Product(id={self.id}, title='{self.title[:50]}...', image_path='{self.image_path}')>"


def create_tables(engine):
    Base.metadata.create_all(bind=engine)

