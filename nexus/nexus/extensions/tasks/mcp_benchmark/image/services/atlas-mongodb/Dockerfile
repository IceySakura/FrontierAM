FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://pgp.mongodb.com/server-7.0.asc | gpg --dearmor -o /etc/apt/keyrings/mongodb-server-7.0.gpg && \
    echo "deb [ arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" \
      > /etc/apt/sources.list.d/mongodb-org-7.0.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      mongodb-org-server \
      mongodb-mongosh \
      mongodb-database-tools && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/mcp-benchmarks/npm && \
    cd /opt/mcp-benchmarks/npm && \
    npm init -y >/dev/null 2>&1 && \
    npm install mongodb-mcp-server@0.2.0 && \
    npm cache clean --force && \
    rm -rf /root/.npm /tmp/npm-cache

ENV PATH=/opt/mcp-benchmarks/npm/node_modules/.bin:${PATH}

RUN mkdir -p /data/db /opt/mongo_dump

COPY .build/dump/mongo_dump_video_game_store/ /opt/mongo_dump/mongo_dump_video_game_store/

RUN mongod --bind_ip 127.0.0.1 --port 27017 --dbpath /data/db --fork \
      --logpath /tmp/mongod-build.log --pidfilepath /tmp/mongod-build.pid && \
    mongorestore --dir /opt/mongo_dump/mongo_dump_video_game_store && \
    (mongosh --quiet mongodb://127.0.0.1:27017/admin --eval 'db.adminCommand({shutdown: 1})' || true) && \
    rm -f /tmp/mongod-build.pid
