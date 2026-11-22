FROM maven:3.9-eclipse-temurin-25 AS builder
WORKDIR /app
COPY pom.xml .
COPY cutmesh-common/pom.xml cutmesh-common/pom.xml
COPY cutmesh-audit/pom.xml cutmesh-audit/pom.xml
COPY cutmesh-iam/pom.xml cutmesh-iam/pom.xml
COPY cutmesh-fleet/pom.xml cutmesh-fleet/pom.xml
COPY cutmesh-orchestration/pom.xml cutmesh-orchestration/pom.xml
COPY cutmesh-gateway/pom.xml cutmesh-gateway/pom.xml
COPY cutmesh-app/pom.xml cutmesh-app/pom.xml
RUN mvn dependency:go-offline -B
COPY cutmesh-common/src cutmesh-common/src
COPY cutmesh-audit/src cutmesh-audit/src
COPY cutmesh-iam/src cutmesh-iam/src
COPY cutmesh-fleet/src cutmesh-fleet/src
COPY cutmesh-orchestration/src cutmesh-orchestration/src
COPY cutmesh-gateway/src cutmesh-gateway/src
COPY cutmesh-app/src cutmesh-app/src
RUN mvn clean package -DskipTests -B

FROM eclipse-temurin:25-jre-noble
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /tmp/build-ctx/
RUN mkdir -p /opt/cutmesh/llamacpp /app/models /data/cutmesh/cache && \
    if [ -f /tmp/build-ctx/runtime-linux.tar.gz ]; then \
      tar -xzf /tmp/build-ctx/runtime-linux.tar.gz -C / && \
      chmod +x /usr/local/bin/cutmesh /opt/cutmesh/llamacpp/* 2>/dev/null || true; \
    fi && \
    rm -rf /tmp/build-ctx
COPY --from=builder /app/cutmesh-app/target/cutmesh-app-1.0.0.jar app.jar
ENV LD_LIBRARY_PATH="/opt/cutmesh/llamacpp:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64" \
    LLAMA_CACHE="/data/cutmesh/cache" \
    RUST_LOG="cutmesh=info"
EXPOSE 8085
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
