"""SQLAlchemy ORM models for Hansard data."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()
EMBEDDING_DIM = 384


class Utterance(Base):
    """A single utterance or speech segment in the Hansard record."""
    __tablename__ = "utterance"

    id = Column(Integer, primary_key=True, index=True)
    xml_path = Column(Text, nullable=False)
    date = Column(Date, nullable=True)
    utterance = Column(Text, nullable=False)
    original_utterance = Column(Text, nullable=False)
    embedding_text = Column(Text, nullable=False)

    speakername = Column(String, nullable=False)
    person_id = Column(Integer, nullable=True)
    speakeroffice = Column(String, nullable=True)

    oral_heading = Column(Text, nullable=True)
    major_heading = Column(Text, nullable=True)
    minor_heading = Column(Text, nullable=True)
    speech_id = Column(String, nullable=True)
    url = Column(Text, nullable=True)
    colnum = Column(Integer, nullable=True)

    is_statement = Column(Integer, nullable=False)
    is_question = Column(Integer, nullable=False)
    is_main_question = Column(Integer, nullable=False)
    is_supplementary_question = Column(Integer, nullable=False)
    is_intervention = Column(Integer, nullable=False)
    is_answer = Column(Integer, nullable=False)
    # Statement context (for answers to statements)
    statement_text = Column(Text, nullable=True)
    statement_speaker = Column(String, nullable=True)
    original_statement_text = Column(Text, nullable=True)

    # Main question context (for answers)
    question_text = Column(Text, nullable=True)
    question_speaker = Column(String, nullable=True)
    original_question_text = Column(Text, nullable=True)

    # Supplementary/intervention context (for answers)
    context_question_text = Column(Text, nullable=True)
    context_question_speaker = Column(String, nullable=True)
    context_question_type = Column(String, nullable=True)
    original_context_question_text = Column(Text, nullable=True)

    party_at_time = Column(String, nullable=True)

    chunks = relationship(
        "UtteranceChunk", back_populates="utterance", cascade="all, delete-orphan"
    )


class UtteranceChunk(Base):
    """A chunk of an utterance for embedding-based retrieval."""

    __tablename__ = "utterance_chunk"

    id = Column(Integer, primary_key=True, index=True)
    utterance_id = Column(
        Integer, ForeignKey("utterance.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding_text = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)

    utterance = relationship("Utterance", back_populates="chunks")
    embedding = relationship(
        "Embedding", back_populates="chunk", cascade="all, delete-orphan"
    )


class Embedding(Base):
    """Vector embedding for an utterance chunk."""

    __tablename__ = "embedding"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(
        Integer, ForeignKey("utterance_chunk.id", ondelete="CASCADE"), nullable=False
    )
    embedding_type = Column(String, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM))

    chunk = relationship("UtteranceChunk", back_populates="embedding")


class Person(Base):
    """A person who has spoken in Parliament."""
    __tablename__ = "person"

    id = Column(Integer, primary_key=True)
    given_name = Column(String, nullable=True)
    family_name = Column(String, nullable=True)
    display_name = Column(String, nullable=False)


class Membership(Base):
    """A continuous parliamentary membership (party + seat) for a person."""
    __tablename__ = "membership"

    membership_id = Column(String, primary_key=True, index=True)
    person_id = Column(Integer, index=True, nullable=False)

    # Party during this membership (e.g. labour, conservative).
    party = Column(String, index=True, nullable=True)

    # Constituency (Commons) or seat (Lords).
    post_id = Column(String, index=True, nullable=True)

    # Date range this membership was valid.
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)

    # Why the membership started / ended.
    start_reason = Column(String, nullable=True)
    end_reason = Column(String, nullable=True)

    # Legacy Hansard identifier (useful for joins).
    historichansard_id = Column(String, index=True, nullable=True)


class Division(Base):
    """A parliamentary division (a vote event)."""
    __tablename__ = "division"

    id = Column(Integer, primary_key=True, index=True)
    division_key = Column(String, unique=True, nullable=False, index=True)
    vote_date = Column(Date, nullable=False)
    description = Column(Text, nullable=False)

    votes = relationship("Vote", back_populates="division", cascade="all, delete-orphan")


class Vote(Base):
    """An individual MP's vote in a parliamentary division."""
    __tablename__ = "vote"

    id = Column(Integer, primary_key=True, index=True)
    division_id = Column(
        Integer, ForeignKey("division.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_id = Column(Integer, nullable=False, index=True)
    membership_id = Column(String, nullable=False)
    vote = Column(String, nullable=False)  # 'absent', 'aye', 'no', 'abstain'

    division = relationship("Division", back_populates="votes")


class MPPolicySummary(Base):
    """Summary of an MP's voting alignment with a policy."""

    __tablename__ = "mp_policy_summary"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, nullable=False, index=True)
    policy_id = Column(Integer, nullable=False, index=True)
    period_id = Column(Integer, nullable=True)
    name = Column(String, nullable=True)
    policy_description = Column(Text, nullable=True)
    context_description = Column(Text, nullable=True)
    distance_score = Column(Float, nullable=False)
    start_year = Column(Integer, nullable=True)
    end_year = Column(Integer, nullable=True)
    num_votes_same = Column(Integer, nullable=False)
    num_strong_votes_same = Column(Integer, nullable=False)
    num_votes_different = Column(Integer, nullable=False)
    num_strong_votes_different = Column(Integer, nullable=False)
    num_votes_absent = Column(Integer, nullable=False)
    num_strong_votes_absent = Column(Integer, nullable=False)
    num_votes_abstain = Column(Integer, nullable=False)
    num_strong_votes_abstain = Column(Integer, nullable=False)
    division_ids = Column(ARRAY(Integer), nullable=True)
    mp_policy_alignment_score = Column(Float, nullable=True)
    mp_stance_label = Column(String, nullable=False)
