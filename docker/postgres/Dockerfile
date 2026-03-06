FROM postgres:17

# Install pgvector from the PostgreSQL apt repository (already in base image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        postgresql-17-pgvector \
        ca-certificates \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Install pg_textsearch v0.6.0 from GitHub release
# The zip contains a .deb: pg-textsearch-postgresql-17_0.6.0-1_amd64.deb
ARG PG_TEXTSEARCH_URL=https://github.com/timescale/pg_textsearch/releases/download/v0.6.0/pg-textsearch-v0.6.0-pg17-amd64.zip
ARG PG_TEXTSEARCH_SHA256=e27905736fb36197deda4c21a44ec6ab85dff290454c388230e72182597a6c57
RUN wget -q "${PG_TEXTSEARCH_URL}" -O /tmp/pg_textsearch.zip && \
    echo "${PG_TEXTSEARCH_SHA256}  /tmp/pg_textsearch.zip" | sha256sum -c - && \
    unzip /tmp/pg_textsearch.zip -d /tmp/pg_textsearch && \
    dpkg -i /tmp/pg_textsearch/*.deb && \
    rm -rf /tmp/pg_textsearch /tmp/pg_textsearch.zip && \
    apt-get purge -y wget unzip && apt-get autoremove -y

# Create extensions on first database init
COPY init.sql /docker-entrypoint-initdb.d/
