#CLI #명령어 #terminal

✅ **파일/디렉터리 관리:** pwd, ls, cd, rm, mkdir
✅ **파일 보기:** cat, less, head, tail
✅ **네트워크:** ping, curl, wget
✅ **시스템 정보:** whoami, uname -a, top, df -h
✅ **프로세스 관리:** ps aux, kill, jobs, fg, bg
✅ **압축/해제:** tar, zip, unzip

**pwd** 현재 작업 중인 디렉터리 출력 (*print working directory*)

**ls** 현재 디렉터리의 파일 및 폴더 목록 확인 (*list*)

**ls -l** 자세한 파일 정보 (권한, 크기 등) 출력

**ls -a** 숨김 파일 포함 모든 파일 표시

**cd <디렉터리명>** 특정 디렉터리로 이동 (*change directory*)

**cd ..** 상위 디렉터리로 이동 

**mkdir <폴더명>** 새 디렉터리(폴더) 생성 (*make directory*)

**rmdir <폴더명>** 빈 디렉터리 삭제 (*remove directory*)

**rm -r <폴더명>** 디렉터리 및 하위 파일 강제 삭제 (*remove*)

**rm <파일명>** 파일 삭제

**touch <파일명>** 새 파일

**cat <파일명>** 파일 내용 출력 (*concatenate*)

**less <파일명>** 긴 파일을 한 화면씩 보기 (위/아래 이동 가능)

**more <파일명>** less와 비슷하지만 위로 이동 불가

**head <파일명>** 파일의 처음 10줄 출력

**tail <파일명>** 파일의 마지막 10줄 출력

**tail -f <파일명>** 실시간 로그 확인 (파일이 업데이트될 때 계속 출력됨)

**cp <원본> <대상>** 파일 복사 (*copy*)

**cp -r <원본폴더> <대상폴더>** 디렉터리 복사

**mv <원본> <대상>** 파일 또는 폴더 이동 (이름 변경 포함,*move*)

**find . -name " .txt"** 특정 파일 찾기 (현재 디렉터리에서 .txt 파일 검색)

grep "검색어" <파일명> 특정 파일에서 특정 단어 검색

grep -r "검색어" <폴더> 특정 폴더에서 모든 파일을 대상으로 검색

whoami 현재 로그인한 사용자 확인

uname -a 시스템 정보 출력 (운영체제, 커널 버전 등)

top 실시간 CPU, 메모리 사용량 및 프로세스 확인

ps aux 실행 중인 프로세스 목록 확인

kill <PID> 특정 프로세스 종료 (PID는 ps aux로 확인 가능)

kill -9 <PID> 강제 종료 (SIGKILL)

df -h 디스크 사용량 확인

du -sh * 현재 폴더의 하위 파일/폴더 크기 확인

ping <도메인> 특정 서버에 핑(응답 시간) 테스트

curl <URL> 특정 URL의 내용 요청

wget <URL> 특정 파일 다운로드

ifconfig 또는 ip a 네트워크 인터페이스 및 IP 확인 (Linux)

ipconfig 네트워크 정보 확인 (Windows)

netstat -tulnp 현재 열려 있는 포트 확인

chmod 755 <파일> 파일 권한 변경 (rwxr-xr-x)

chown 사용자:그룹 <파일> 파일 소유권 변경

sudo <명령어> 관리자 권한으로 명령 실행

su <사용자> 다른 사용자 계정으로 변경

tar -cvf archive.tar <파일/폴더> tar 압축 파일 생성

tar -xvf archive.tar tar 압축 파일 해제

tar -czvf archive.tar.gz <파일/폴더> gzip으로 압축된 tar 파일 생성

tar -xzvf archive.tar.gz gzip tar 파일 해제

zip -r archive.zip <파일/폴더> zip 파일 생성

unzip archive.zip zip 파일 해제

command & 명령어를 백그라운드에서 실행

jobs 실행 중인 백그라운드 작업 목록 확인

fg %1 백그라운드에서 실행 중인 작업을 포그라운드로 전환

bg %1 일시 정지된 작업을 백그라운드에서 다시 실행

echo $PATH 환경 변수 PATH 확인

export VAR_NAME=value 환경 변수 설정

unset VAR_NAME 환경 변수 삭제

history 사용한 명령어 기록 확인

clear 터미널 화면 정리

alias ll='ls -lah' 명령어 단축키 설정

unalias ll 단축키 삭제
