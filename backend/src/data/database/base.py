"""Base class for database repositories."""

from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker


class BaseRepository:
    """Base class for database repositories."""

    def __init__(self, engine: Engine) -> None:
        """Initialise the repository with a database engine.

        Args:
            engine: SQLAlchemy engine for database connections.
        """
        self.Session = sessionmaker(bind=engine)
