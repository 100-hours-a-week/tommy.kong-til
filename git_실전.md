# 🧠 Git 핵심 개념 정리 (실전 편)
---

## 🧰 Git 기초 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `clone` | 원격 저장소 복사 |
| `add` | 스테이지 영역에 작업 파일 추가 |
| `commit` | 세이브, 스테이지 영역의 파일들로 커밋(=세이브) 생성 |
| `push` | 원격 저장소에 커밋 업로드 |

---

## 📂 파일의 내용 되돌리기

- 특정 파일을 마지막 커밋 상태로 되돌리려면:  
  👉 해당 파일 선택 → **"코드 뭉치 버리기"** 선택

---

## 🌿 브랜치 변경하기

- **브랜치란?**  
  기존 내용을 유지한 채 **새로운 내용을 추가**하고 싶을 때 사용

- **체크아웃**  
  특정 브랜치(또는 커밋)으로 돌아가고 싶을 때 사용

- **SourceTree에서 체크아웃**  
  브랜치 이름을 더블 클릭하면 체크아웃됨

---

## 🔀 병합하기 (Merge)

### 병합하기 1 – Fast-Forward

- 헤드 브랜치에 변경사항이 없고  
- 병합 대상 브랜치가 헤드로부터 시작된 경우  
✅ 아주 쉽게 병합 가능 (Fast-forward)

### 병합하기 2 – 진짜 병합

- 헤드 브랜치에 추가적인 커밋이 생긴 경우  
- 진짜 병합이 필요해짐  
- 충돌이 날 수도 있음 → **겁내지 말자**

---

## ⚠️ 충돌 해결하기

- 제일 중요한 점: **겁내지 말아요!**
- 같은 파일을 양쪽 브랜치에서 동시에 수정했을 경우 충돌 확률이 높음
- **에디터** 또는 **SourceTree**를 이용해 충돌 해결 가능

---

## 🕰 커밋 되돌리기

### 🔁 `reset` 사용

- **장점:** 쉬움  
- **단점 1:** 커밋이 날아감  
- **단점 2:** 강제 푸시(`--force`)가 필요

### 🌱 브랜치 만들어서 되돌리기

- reset과 달리 **내용은 사라지지 않음**
- **장점:** 쉬움  
- **단점:** 트리가 지저분해질 수 있음

### 🔄 `revert`

- 커밋은 **없어지지 않음**  
- **장점:** 가장 정석적  
- **단점:** 충돌이 날 수 있음  
- **주의:** 현재 선택한 커밋 내용을 되돌림

#### 🔂 `revert` 여러 개

- 여러 커밋을 되돌리려면 **최신 커밋부터 순서대로 revert**  
✅ 그래야 충돌을 줄일 수 있음

---

## 📝 커밋 덮어쓰기

- 이전 커밋을 수정하려면:  
```bash
  git commit --amend
```

📦 stash
	•	브랜치 변경 전에 작업 내용을 임시로 저장할 수 있는 기능
	•	👉 다른 브랜치로 체크아웃하기 전에 사용하면 유용

⸻

❗ 기타 주의 사항
	•	코드를 남기기 위해 주석 달지 말기
	•	커밋 메시지를 명확하게 잘 쓰기
	•	하나의 구현이 끝날 때마다 작은 단위로 자주 커밋하기

⸻

🔃 rebase
	•	merge처럼 두 브랜치를 합칠 때 사용
	•	현재 브랜치가 대상 브랜치 위로 올라감
	•	위험할 수 있으니 신중하게 사용

이미 push를 했다면 git push --force
---

## ✅ 1. Git push 오류: `[rejected] main -> main (fetch first)`

### 🔍 원인
- 원격 저장소에 커밋이 먼저 존재 (예: GitHub에서 README 생성)
- 로컬이 뒤처져서 `push` 실패

### 💡 해결
```bash
git pull origin main --allow-unrelated-histories
git push origin main
# 또는
git push --force


## git fetch와 git pull의 차이 
git fetch : 원격 켜밋 가져오기만 (로컬 영향x)
git pull : 원격 커밋 가져오고 로컬에 반영 죽, merge까지 같이 하는 것 

## pull –rebase vs –no-rebase
git pull --rebase 로컬 커밋을 최신 커밋 위에 재배치
git pull --no-rebase merge 방식으로 반영 

git reset 이전 커밋 상태로 완전히 되돌림(히스토리 변경)
git revert 이전 커밋을 취소하는 새 커밋 생성(히스토리 보존)

merge : 브랜치 내역을 남기면서 합침 
rebase : 브랜치 내역 없이 직선형 히스토리, 커밋 하나하나 충동 해결 필요 

### merge 충돌 발생시
git merge --abort
git add .
git commit

### rebase 충동 발생시
git add .
git rebase --continue


## squach
git rebase -i HEAD~3
# 첫 커밋은 pick, 나머지는 squash
git push --force

7. 팀 레포 주의사항
	•	GitHub에서 README, .gitignore 선택 시 → git pull 먼저 해야 함
	•	커밋 메시지 컨벤션 지키기 (feat:, fix:, docs: 등)
	•	팀 작업 중 reset이나 --force 사용 주의

## gitignore문법
file.c           # 모든 file.c
/file.c          # 최상위 폴더의 file.c
*.c              # 모든 .c 파일
!not_ignore.c    # 무시하지 않을 파일

logs             # logs 파일 또는 폴더
logs/            # logs 폴더만
logs/debug.log   # logs 폴더 내 특정 파일
logs/**/debug.log # logs 안 하위 폴더 포함

## 로컬 -> 원격 브랜치 push
git checkout -b from-local
git push -u origin from-local

# 원격 -> 로컬 브랜치 
git fetch origin
git checkout -b from-remote origin/from-remote

브랜치 전환 : git switch <branch>
새 브랜치 생성 및 전환 : git switch -c <branch>
이전 커밋의 파일 복원 : git restore --source <commit> <file>

커밋을 취소하고 히스토리를 바꾸고 싶다 : reset
커밋을 취소하되 히스토리는 유지하고 싶다 : revert 
파일을 복구하거나 스테이지에서 내리고 싶다 : restore 

git reset HEAD~1          # 마지막 커밋 삭제 (로컬만)
git reset <file>          # 파일을 staging에서 내리기
git reset --hard HEAD~1   # 커밋 + 작업 내용까지 완전 삭제

git revert abc123

git restore file.txt                   # 수정한 파일을 원래대로 복원
git restore --staged file.txt          # staging에서만 빼기
git restore --source=HEAD~1 file.txt   # 과거 커밋 상태로 파일 복원

🧠 비유로 쉽게 이해해보면…
	•	commit = 책의 각 페이지 (변경 내역)
	•	branch = 책갈피 (페이지 번호를 가리킴)
	•	HEAD = 내가 지금 읽고 있는 책갈피 (혹은 직접 책 페이지)