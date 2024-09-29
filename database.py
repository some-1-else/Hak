import re

from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column


@as_declarative()
class Base:
    """Base class for all database entities"""

    @classmethod
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate database table name automatically.
        Convert CamelCase class name to snake_case db table name.
        """
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    def __repr__(self) -> str:
        attrs = []
        for c in self.__table__.columns:
            attrs.append(f"{c.name}={getattr(self, c.name)}")
        return "{}({})".format(self.__class__.__name__, ", ".join(attrs))


class Video(Base):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    model_1_e: Mapped[str | None] # Путь к эмбеддингу от модели Video
    model_2_e: Mapped[str | None] # Путь к эмбеддингу от модели Audio
    model_3_e: Mapped[str | None] # Путь к эмбеддингу от модели ViT
