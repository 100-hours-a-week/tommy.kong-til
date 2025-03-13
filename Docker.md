#docker #배포 

# 🐳 Docker 개요

## Docker란?
Docker는 애플리케이션과 그 **종속성(Dependencies)** 을 컨테이너(Container)라는 단위로 패키징하여 어디서든 일관된 환경에서 실행할 수 있도록 도와주는 **컨테이너 기반 가상화 플랫폼**입니다.

Docker를 사용하면 개발 환경과 배포 환경을 일치시킬 수 있으며, **마이크로서비스 아키텍처**에서 독립적인 서비스 배포 및 확장이 가능합니다.

---

## 🏗 Docker의 핵심 개념

### 📌 Docker Engine
Docker의 **핵심(Core) 요소**로, 컨테이너를 생성, 실행 및 관리하는 역할을 합니다.

- **Server**: 컨테이너를 실행하는 **데몬 프로세스(dockerd)**.
- **REST API**: 데몬과 상호 작용하기 위한 API 인터페이스.
- **CLI (Command-Line Interface)**: 터미널에서 Docker를 제어하는 명령어 도구.

---

### 📌 Docker 이미지 (Image)
이미지는 **컨테이너 실행을 위한 읽기 전용 템플릿**입니다.

- Dockerfile을 사용하여 **이미지를 생성**할 수 있습니다.
- 애플리케이션 실행에 필요한 모든 파일과 설정을 포함합니다.
- 컨테이너는 **이미지를 기반으로 실행되는 독립적인 인스턴스**입니다.

```shell
# 로컬에 있는 모든 이미지 목록 보기
docker images

# Docker Hub에서 이미지 가져오기
docker pull [image-name]

# Dockerfile을 기반으로 이미지 빌드하기
docker build -t [image-name] .
```

---

### 📌 Docker 컨테이너 (Container)
컨테이너는 **이미지를 실행한 인스턴스**로, 애플리케이션이 동작하는 격리된 환경을 제공합니다.

- 독립적으로 실행되며, 여러 컨테이너가 동시에 실행 가능.
- 컨테이너 간 통신 가능 (네트워크 설정을 통해).
- 종료되면 기본적으로 모든 데이터가 삭제되므로, **Docker Volume**을 활용하여 데이터 지속성을 유지할 수 있음.

```shell
# 실행 중인 컨테이너 목록 보기
docker ps

# 모든 컨테이너 목록 보기 (종료된 컨테이너 포함)
docker ps -a

# 새 컨테이너 실행
docker run -d -p 8080:80 --name my-container my-image
```

---

### 📌 Dockerfile (도커파일)
**Docker 이미지를 빌드하기 위한 명령어(script)** 가 포함된 파일입니다.

```dockerfile
# Step 1: 사용할 기본 이미지 지정
FROM python:3.13-slim

# Step 2: 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# Step 3: 로컬 파일을 컨테이너로 복사
COPY . /app

# Step 4: 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: 컨테이너에서 사용할 포트 지정
EXPOSE 5000

# Step 6: 컨테이너 실행 시 실행할 명령어
CMD ["python", "app.py"]
```

```shell
# Dockerfile을 사용하여 이미지 빌드
docker build -t my-app .

# 생성한 이미지를 기반으로 컨테이너 실행
docker run -d -p 8080:80 my-app
```

---

## 📂 Docker Volume (볼륨)

Docker 컨테이너는 기본적으로 **휘발성(Volatile)** 이므로, 컨테이너가 삭제되면 내부의 데이터도 삭제됩니다.  
이를 방지하고, **데이터 지속성(Persistence)** 을 유지하기 위해 **Docker Volume**을 사용합니다.

```shell
# 새로운 볼륨 생성
docker volume create my_volume

# 볼륨 목록 확인
docker volume ls

# 특정 볼륨 정보 확인
docker volume inspect my_volume

# 사용하지 않는 모든 볼륨 정리
docker volume prune
```

### 📌 볼륨을 사용한 컨테이너 실행
```shell
docker run -d -v my_volume:/data my-image
```

---

## 📑 Docker Compose

**여러 개의 컨테이너를 한 번에 실행**할 수 있도록 도와주는 툴입니다.  
\`docker-compose.yml\` 파일을 사용하여 **멀티 컨테이너 환경을 구성**할 수 있습니다.

### 📌 예제: Flask + MySQL 애플리케이션
```yaml
version: "3.8"
services:
  app:
    build:
      context: .
    ports:
      - "5001:5000"
    depends_on:
      - db
    environment:
      - MYSQL_HOST=db
      - MYSQL_USER=root
      - MYSQL_PASSWORD=password
      - MYSQL_DATABASE=exampledb

  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: exampledb
    ports:
      - "3306:3306"
    volumes:
      - db_data:/var/lib/mysql

volumes:
  db_data:
```

```shell
# Docker Compose로 컨테이너 실행
docker-compose up -d

# 모든 컨테이너 종료 및 정리
docker-compose down --rmi all --volumes --remove-orphans
```

---

## 🎯 주요 Docker 명령어 정리

### 🔹 일반 명령어
```shell
docker --version  # Docker 버전 확인
docker info       # 시스템 정보 확인
docker help       # 명령어 도움말 확인
```

### 🔹 이미지 관리
```shell
docker images               # 로컬 이미지 목록 확인
docker pull [image-name]     # 이미지 다운로드
docker build -t my-app .     # Dockerfile을 기반으로 이미지 빌드
docker push [image-name]     # 이미지 레지스트리에 업로드
docker rmi [image-id]        # 이미지 삭제
```

### 🔹 컨테이너 관리
```shell
docker ps                   # 실행 중인 컨테이너 목록
docker ps -a                # 모든 컨테이너 목록
docker run -d my-image       # 컨테이너 실행
docker stop [container-id]   # 컨테이너 중지
docker rm [container-id]     # 컨테이너 삭제
docker logs [container-id]   # 컨테이너 로그 확인
```

### 🔹 디버깅 & 모니터링
```shell
docker inspect [object-id]   # 상세 정보 확인
docker stats [container-id]  # 실시간 자원 사용량 확인
docker top [container-id]    # 컨테이너 내부 프로세스 확인
docker events                # Docker 이벤트 로그 확인
```

---

이 문서는 Docker 개념을 정리한 Markdown 파일이며, 실제로 활용할 때는 필요에 따라 내용을 추가하면 더욱 유용할 것입니다. 🚀