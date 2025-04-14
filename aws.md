# 클라우드 서비스란?
- 물리적 자원 혹은 논리적 자원을 대여하는 것 

## EC2
- 필요한 만큼 자원을 대여해서 사용하는 것 


🔗 AWS와 Docker의 연관성

✅ Docker란?
	•	애플리케이션을 컨테이너(가벼운 가상 환경)에 포장해서 실행하는 기술
	•	개발 환경과 운영 환경이 일관성 있게 유지됨

✅ AWS란?
	•	Amazon Web Services: 클라우드 인프라 플랫폼
	•	서버, 네트워크, DB, 스토리지 등 모든 자원을 온라인으로 제공

✅ 연관성
	•	Docker 컨테이너를 AWS에서 배포하고 관리할 수 있음
	•	AWS는 Docker를 실행하기 위한 다양한 서비스를 제공함

🧭 Docker → AWS 배포 프로세스
1. Dockerize 앱 => dockerfile을 만들어 앱을 컨테이너화
2. 이미지 빌드 => docker build로 컨테이너 이미지 생성 
3. 이미지 푸시 => AWS ECR에 이미지 업로드
4. 실행 환경 준비 => EC2,ECS, 또는 EKS에 컨테이너 실행 준비
5. 배포 => docker run or AWS, ECS/EKS/EC2에서 실행
6. 스케일링 & 모니터링 => AWS에서 오토스케일링, 로드 밸런싱, CloutWatch로 모니터링 가능 

# AWS에서 Docker를 실행할 수 있는 서비스들
- EC2 => 직접 서버에 Docker를 설치하고 실행 가능
- ECS => 서버없이 컨테이너만 실행
- EKS => Kubernetes 기반 컨테이너를 오케스트레이션
- Lightsail => 간단한 Docker 컨테이너 실행용 저비용 서비스 

# EC2에서 docker 사용 
	1.	EC2 인스턴스 생성
	2.	SSH 접속 (ssh -i key.pem ec2-user@...)
	3.	Docker 설치 (sudo yum install docker)
	4.	Docker 실행 (sudo service docker start)
	5.	이미지 빌드 & 실행 (docker build, docker run)


⚙️ 1. 오토스케일링 (Auto Scaling)

🟡 개념:
	•	트래픽이 많아지면 자동으로 서버를 늘리고, 적어지면 줄이는 기능이에요.

📌 예시:

갑자기 쇼핑몰에 사람들이 몰리면 서버가 늘어나고, 한밤중에 사람 없으면 서버 줄어드는 것

🛠️ AWS 서비스:
	•	EC2 Auto Scaling
	•	ECS Service Auto Scaling

✅ 장점:
	•	비용 절감 (필요할 때만 리소스를 사용)
	•	성능 유지 (사용자 폭주에도 끄떡없음)

🌐 2. 로드 밸런싱 (Load Balancing)

🟡 개념:
	•	여러 대의 서버에 트래픽을 고르게 나눠주는 시스템

📌 예시:

음식점에 손님이 몰리면 직원들이 고르게 배정돼야 하잖아요? 그 역할을 로드 밸런서가 해요.

🛠️ AWS 서비스:
	•	Elastic Load Balancing (ELB)

✅ 장점:
	•	서버 과부하 방지
	•	고가용성 (한 서버가 죽어도 다른 서버가 응답)

🧠 3. 오케스트레이션 (Orchestration)

🟡 개념:
	•	여러 개의 컨테이너를 자동으로 배치, 시작, 중지, 모니터링, 복구하는 기술

📌 예시:

컨테이너가 100개쯤 되면, 사람 손으로 관리 못하죠. 그래서 오케스트레이터가 전체 지휘를 해주는 거예요. 마치 오케스트라 지휘자처럼요 🎻

🛠️ 대표 툴:
	•	Kubernetes (EKS in AWS)
	•	AWS ECS (Elastic Container Service)

✅ 주요 기능:
	•	컨테이너 자동 배치
	•	장애 복구
	•	롤링 업데이트
	•	네트워크 구성 자동화


