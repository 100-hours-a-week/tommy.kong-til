# 1. Git 기초 : add, commit

코드 짜다가 실수해서 2일 전으로 돌아가고 싶으면 어쩌죠? 
파일저장만 주구장창 했으면 다시 돌아갈 수는 없습니다. 
해결 방법이 2개 있는데
- 매일매일 손수 파일 복사본을 만들어두거나 
- **git 쓰거나** 
둘 중 선택하면 됩니다. 
git의 commit 기능을 쓰면 쓰면 파일의 현재상태를 매일매일 **기록**해둘 수 있습니다.
정확히 말하면 파일의 **스냅샷을 저장**해줍니다. 
그럼 원할 때 쉽게 되돌아가거나 그럴 수 있음 
오늘은 파일의 현재상태를 기록해줄 수 있는 git commit 명령어를 알아봅시다. 
> **일단 작업폴더에서 git을 이용하고 싶으면**
거기서 터미널을 열어서 **git init** 부터 입력하고 시작하면 됩니다. 
이제 git이 여러분이 파일생성하는거, 코드작성하는걸 추적하기 시작합니다. 
파일 하나를 생성하고 코드를 아무렇게나 짜봅시다. 
저는 test.txt 파일을 생성해서 대충 아무거나 코드짜봤습니다.
오늘 짠 코드가 맘에 들어서 따로 기록을 해두고 싶은겁니다. 
그러면 아까 설치한 git을 이용해서
"이 파일 현재상태를 기록 좀 해줘~" 라고 부탁하면 되는데 
```
git add 파일명 
git commit -m '아무메세지'
```
차례로 터미널에 입력하면 됩니다. 
이러면 방금 파일의 내용을 몰래 어딘가에 기록해줍니다. 

이럼 뭐가좋냐고요? 
이제 한참 뒤에도 이 파일상태 그대로 되돌리거나 그럴 수 있고
나중에 누가 개같이 코드짜놨는지 확인도 가능합니다.
아무튼 예로부터 조선은 기록을 중요하게 여겼기 때문에
코드짜다가 중요한 순간순간에 기록하는 습관을 들여봅시다. 

"기록"이라기보다는 **"버전생성"이라고 부르는 경우**가 더 많습니다.
심심하니까 아까 만든 파일 수정하고 버전생성하는 작업을 몇 번 더 하고옵시다. 

> **오늘의 용어정리 : staging area & repository** 
버전만들 땐 git add, git commit 차례로 하면 된다고 했습니다.
![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC1.png)

그림으로 그리자면 이런 식인데  
여기서 **가운데 부분을 staging area, 파일버전이 저장되는 곳을 repository (저장소)** 라고 합니다. 
**1. staging area는 commit을 하기 전에 commit할 파일들을 골라놓는 곳입니다.**
그리고 staging area에 파일넣는 행위를 staging이라고 합니다. 
git add 명령어로 staging 할 수 있습니다.

**2. repository는 commit된 파일의 버전들을 모아놓는 곳입니다.**
repository의 실체를 구경하고 싶으면 작업폴더안에 숨겨져 있는 .git 폴더 열어보면 됩니다. 
아무튼 staging area & repository 2개는 자주 쓰는 용어니까 잘 외워둡시다.

> **다른 명령어들** 
```
git add 파일명1 파일명2
```
이렇게 여러 파일을 동시에 스테이징할 수 있습니다.
```
git add .
```
작업폴더의 모든 파일을 전부 스테이징하고 싶으면 git add . 하면 됩니다.
```
git status
```
요즘 젊은이들은 인생이 힘들고 복잡할 때 "상태창!!"을 외치는데 
git도 마찬가지로 힘들고 복잡할 때 git status 입력하면 됩니다.  
**지금 변경된 파일, 스테이징된 파일 이런걸** 쭉 알려줍니다. 
지금 뭐 하는지 까먹었을 때도 자주 입력하게 됩니다. 
```
git restore --staged 파일명
```
**스테이징된 파일을 취소**하고 싶으면 하면 이거 입력하면 됩니다.
터미널에서 자주 알려주는 명령어라 외울 필요는 없습니다.
저기서 파일명 대신 점찍으면 어떻게 될까요

```
git commit -m '메세지'
```
commit 할 때 -m 뒤에 메세지 입력가능합니다. 
메세지에 코드에 무슨기능 추가했는지 이런거 적으면 됩니다. 

```
git log --all --oneline
git log --all --oneline --graph
```
commit 기록을 한 눈에 파악하고 싶으면 git log 명령어 입력하면 됩니다. 
--graph 옵션을 넣으면 그래프로 그려줍니다. 지금은 보잘것 없음 
다만 입력 후엔 Vim 에디터가 켜져서 j, k 키로 위아래 스크롤이 가능하고 q 키로 종료할 수 있습니다. 

**Q. 얼마나 자주 commit 하는게 좋음?**
A. ctrl + s 누르는 것 처럼 5초마다 습관적으로 할 이유는 없고 간단한 기능을 하나 추가할 때 마다 commit 하면 됩니다. 
예를 들어 웹개발시 회원가입기능을 만든다고 하면 
- 회원가입 폼 레이아웃을 만들면 commit 하고 
- 입력한 이메일이 맞는지 검증하는 기능을 만들었으면 commit 하고 
- 서버에 전송하는 기능을 만들었으면 commit 하고 
대충 이렇게 작은 작업하나 마쳤으면 commit 하는게 좋습니다. 
물론 3개 다 만들하고 commit 하는 사람들도 있습니다. 본인 맘임

# 2. Git add, commit, diff 쉽게 하는 방법

근데 요즘은 터미널에 직접 git add 이거 입력하는게 개뻘짓일 수 있습니다.
웬만한 에디터들 보면 git 기능이 내장되어있어서
그거 쓰면 터미널 켤 필요없이 편리하게 add, commit 가능합니다.
(git 기능 없는 에디터면 git 부가기능 설치하면 됩니다.)

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%981-3.png)
▲ VSCode 에디터의 경우 왼쪽 git 처럼 생긴 메뉴 들어가보면

지금 어떤 파일이 변경되고 추가되었는지 쭉 알려줍니다. 
+ 누르면 git add 한거랑 똑같고 
체크마크 누르면 git commit 한거랑 똑같습니다. 
파일이 많고 복잡하면 이거 쓰는게 더 나을 수도 있습니다. 

> **git diff 로 차이점 출력해줄 수 있음** 

commit 하기 전에 이전과 현재 코드가 어떤 차이가 있는지 알고 싶습니까. 
그럼 git diff 명령어를 쓰면 됩니다. 
**바로 전 commit과 현재 코드의 차이점**을 비교해줍니다.
코드 조금 수정해본 다음에 터미널에 git diff 입력해봅시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/git-diff2.png)

그럼 **현재 파일**이 **최근 commit**과 어떤 부분이 달라졌는지 알려줍니다.
(근데 Vim 에디터가 오픈되어서 스크롤은 j, k / 종료는 q 연타해야합니다)
하지만 터미널의 한계로 차이점보기가 힘들고
설정 안만지면 쓸데없이 엔터키나 스페이스바 변동사항도 다 알려주기 때문에 보통은 git diff를 쌩으로 사용하진 않습니다. 

우리 같이 유용한 git diff 명령어 몇개만 알아볼까요? ^0^
```
git diff 커밋id
```
최근 commit과 비교하는게 아니라 **과거의 특정 commit과 현재 파일을 비교하고 싶으면** 커밋ID를 명시해주면 됩니다.
(커밋ID는 git log --oneline 이런거 입력하면 보이는 노란 글자들입니다)
```
git diff 커밋id1 커밋id2
```
과거의 특정 commit 2개 간의 차이점 비교도 가능합니다. 

> **git difftool 이용하면 조금 더 보기좋음**

이거 쓰면 비주얼적으로 훌륭하게 차이점을 분석해줍니다. 
```
git difftool
```
입력하면 현재 파일과 최근 commit의 차이점을 비교해줍니다.

```
git difftool 커밋id
```
입력하면 현재 파일과 특정 commit의 차이점을 비교해줍니다.

```
git difftool 커밋id1 커밋id2
```
입력하면 특정 commit 2개의 차이점을 비교해줍니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/git-difftool-vim.png)
▲ 이것도 Vim 에디터가 뜨는데 

hjkl 키로 이동가능하고 :q 여러번 입력해야 나갈 수 있습니다. 아니면 :qa 입력하셈 
이것도 실은 Vim 에디터와 터미널의 한계로 그렇게 편리하진 않습니다. 
git difftool을 Vim 말고 VSCode로 열고 싶으면 **git diff 말고 에디터 부가기능 쓰는게 더 좋을 수도** 

요즘 에디터들 잘되어있는데 뭐하러 터미널에서 git difftool 입력합니까.
VSCode 에디터의 경우 좌측 Extensions 메뉴에서  Git 관련 부가기능 설치 아무거나 해주면 더 편리하게 git diff 할 수 있습니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/git-graph.png)

▲ VSCode 에디터 extension 메뉴에서 git 검색해서 아무거나 설치해봅시다.
저는 Git graph 부가기능을 설치해보겠습니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/git-diff-in-vscode2.png)
▲ 왼쪽 Git 메뉴 - Git graph 버튼 누르면
commit 내역을 한 눈에 쭉 살펴볼 수 있고 파일명 우클릭하면 git diff도 가능하니
과거 내역을 살펴보고 싶으면 이런 GUI 툴을 주로 활용해봅시다. 

**Q. git은 자고로 터미널로 조작해야지 요즘 신입들은 편한 것만 찾고 말이야 쯧쯧** 
A. 본인이 편한거 쓰는게 젤 좋습니다.

# 3. Git branch 만들기

  
 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC3-1.png)

커밋하면서 계속 코드짜다보면 갑자기 새로운 기능을 추가하거나 그래야하는 경우가 있습니다.
그럴 때는 원본파일에 코드를 추가하고 커밋해도 되겠지만
혹시나 잘못해서 지금까지 짰던 프로그램이 망가지거나 그러면 어떻게하죠? 
그럴 걱정 없이 안전하게 새로운 기능을 추가하고 싶으면
**프로젝트의** **복사본을 만들어서 거기에 먼저 개발**해보는것도 나쁘지않습니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC45.png)

git 안에선 **branch 기능을 이용해서 복사본을 쉽게 만들 수 있습니다.** 
**branch가 뭐냐면 그냥 프로젝트 복사본임** 
예시와 함께 branch 하나 만들어봅시다. 

**쇼핑몰을 만들고 있는데 새로운 기능이 필요하다**
예를 들어 지금 작업폴더에서 쇼핑몰 만드는 코드를 짜고 있다고 가정해봅시다.
근데 갑자기 쿠폰기능을 추가하고 싶은겁니다. 
근데 위험하고 복잡할 것 같아서 원래 있던 소스코드를 직접 수정하는게 아니라 
프로젝트 사본을 만들어서 거기다가 먼저 개발해보고 싶은겁니다.
그러면 branch를 하나 만들면 됩니다.  

```
git branch 브랜치이름 
```
이러면 프로젝트 사본이 하나 생성됩니다.
저는 **git branch** coupon 이라고 작명해봤습니다.

```
git switch 브랜치이름 
```
예를 들어 방금 만든 coupon 브랜치로 이동하고 싶으면 **git switch** coupon 하면 됩니다.

- 옛날엔 "git checkout 브랜치명" 입력했음 
- 다시 메인 브랜치로 되돌아가고 싶으면 **git switch main** 하면 됩니다. (**님들 설정에 따라 main 말고 master 일 수도 있음**)
- **어떤 브랜치에 와있는지 까먹었으면 git status 입력**할 수 있습니다. 
coupon 브랜치로 이동했으면 거기서 개발하고 commit 맘대로 할 수 있습니다.
**coupon 브랜치에서 새로운 파일 만들어서 코드짜고 commit 몇번 해보십시오.** 
**master/main 브랜치에서도 기존 파일들에 commit 몇번 해봅시다.** 




![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC132131.png)
▲ 지금 상황을 그림으로 그려보면 이렇습니다. 
main branch도 commit 하던 곳도 실은 하나의 branch 입니다.
main branch 또는 master branch 라고 부르고 
coupon branch에서 작업한 내용은 원래 브랜치인 main branch에 아무런 영향이 없습니다. 

![[Pasted image 20250304112356.png]]
```
git log --graph --oneline --all
```
branch 와 commit 내역을 한 눈에 그래프로 보고 싶으면 이거 입력해보면 됩니다.  
그림이 저처럼 안이쁘면 commit 이 부족할 뿐 

**Q. git log 하면 나오는 HEAD가 뭔가요?**
A. 님 현재 위치임 

> **branch 합치기**

그래서 branch에서 짰던 코드가 맘에들면 어떻게 하냐고요?
원본코드가 있는 master 또는 main 브랜치에 합치면 됩니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC11.png)

**브랜치를 합치는걸 전문용어로 merge**라고 합니다.
그럼 브랜치에서 개발했던 내용을 main 브랜치에 더해줄 수 있습니다. 
```
git switch main
git merge 브랜치명 
```

merge 하고 싶으면
1. **main/master 브랜치로 다시 이동하고**
2. **git merge 브랜치명 입력하면 합쳐집니다**.
예를 들어 git merge coupon 이러면 coupon 브랜치의 코드들이 main/master 브랜치에 합쳐집니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%985.png)
▲ merge 하고 나서 git log 이런거 해보면 이쁘게 합쳐줬다고 알려줍니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%984.png)
▲ ==근데 합칠 때 주의사항이 있는데 master 브랜치와 coupon 브랜치에서 같은 파일, 같은 줄을 수정했을 경우 merge conflict 가 발생합니다.==
이 경우 에디터로 해당 파일을 열어보면 충돌사항이 적혀있습니다. 
![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%983.png)
▲ 둘 중 어떤 코드를 적용할지 고르면 되는데 
<<<< / >>>> / ==== 이런 쓸데없는 것들은 다 지우고 원하는 코드만 남기면 됩니다.
(VSCode 에디터의 경우 Accept Incoming Change 어쩌구 버튼들을 제공해주는데 그거 누르면 편리합니다)
어떤 코드를 남길지 결정했으면 
**git add 파일명**
**git commit -m '메세지'**
입력하면 새로운 commit 을 생성해주며 merge conflict 해결 + 브랜치 합치기 완료입니다. 

> **협업시 branch 유용함**
여러 개발자들과 협업할 때도 branch를 만들어서 하면 편리합니다. 
같은 프로그램을 만드는데 10명이서 동시에 똑같은 소스코드를 수정하고 저장해버리면 난리가 나지 않을까요. 
그래서 기능을 하나 추가하고 싶으면
1. 우선 branch로 프로젝트 사본을 만들어서 거기서 먼저 개발을 조집니다.
2. 그리고 테스트해봤는데 잘 된다면 main branch 에 다시 합칩니다.

그렇게 개발하면 더 안정적으로 개발이 가능하겠군요. 

오늘 요약정리 :

브랜치 생성은 **git branch 브랜치명**

브랜치 이동은 **git switch 브랜치명** 

브랜치 합치기는 main/master 브랜치로 이동한 뒤에 **git merge 브랜치명**

브랜치마다 commit 내역을 그래프로 보고싶으면 **git log --graph --oneline --all**

브랜치 합칠 때 conflict가 발생하면 **파일열어서 수정하고 git add, git commit 하기** 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/pepe_writing.jpg)

메모해두면 다시 볼거같습니까 

그러지말고 집에서 직접 작업폴더에서 새로운 브랜치 만들어보고 합치기도 해봅시다.

# 4. 다양한 Git merge 방법

저번 시간에 브랜치를 합쳐봤는데

실은 브랜치 합치는 방법은 여러가지가 있습니다.

> **3-way merge**  

저번시간에 했던 것 처럼 

브랜치에 각각 신규 commit이 1회 이상 있는 경우 

merge 명령을 내리면 두 브랜치의 코드를 합쳐서 새로운 commit을 자동으로 생성해주는데

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/merge1.png)

이걸 3-way merge 라고 부릅니다.

이게 merge의 기본 동작방식입니다. 

> **fast-forward merge**

가끔은 새로운 브랜치에만 commit 이 있고

기준이 되는 브랜치에는 신규 commit 이 없는 경우가 있습니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC3-4.png)

이 경우 merge 하게 되면 "fast-forward merge 되었습니다" 라고 알려줍니다.

fast-forward merge가 뭐냐면 

딱히 합칠게 없어서 그냥 신규브랜치 보고

**"지금부터 니 이름은 main 브랜치여"** 하는 것입니다. 

그래도 결과는 어짜피 같지 않을까요. 

그래서 "기준이 되는 브랜치에 신규 commit이 없으면" 자동으로 fast-forward merge가 발동됩니다. 

진짜 그런지 궁금하면 직접 테스트해봅시다. 

물론 싫으면 **git merge --no-ff 브랜치명** 해서 강제로 3-way merge 할 수도 있습니다. 

> **브랜치를 삭제하려면**

3-way,  fast-forward 아무렇게나 merge 해도

브랜치를 merge 하고 나면 브랜치가 자동으로 삭제되진 않습니다.

```
git branch -d 브랜치이름
git branch -D 브랜치이름
```

둘 중 하나 사용하면 이제 필요없는 브랜치를 삭제할 수 있습니다. 

병합이 완료된 브랜치 삭제시엔 -d 이것만 해도 되는데

병합하지 않은 브랜치 삭제시엔 -D 이거 해야함 

심심하면 저번에 만든 coupon 브랜치 삭제하고 어떻게 보이는지 확인해봅시다.

> **rebase and merge** 

브랜치를 rebase 하고 나서 merge 하는 짓거리도 가능합니다. 

일단 rebase가 뭐냐면 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/merge3.png) 

rebase는 브랜치의 시작점을 다른 commit으로 옮겨주는 행위입니다. 

1. rebase를 이용해서 신규브랜치의 시작점을 main 브랜치 최근 commit으로 옮긴 다음 

2. fast-forward merge하는 것입니다. 

이런 식으로도 브랜치 합치기가 가능하겠군요. 

왜 이따구로 하냐고요?

1. 3-way merge 말고 강제로 fast-forward 하고 싶을 때

2. 브랜치 그딴거 필요없이도 코드 잘짜는 고수같은 느낌을 주고 싶을 때

3. commit 내역을 한 줄로 계속 이어서 남기고 싶을 때

그러고 싶으면 일반 3-way merge 대신 rebase & merge 해도 됩니다.

그래서 실제로 rebase and merge 하고 싶으면 

1. 새로운 브랜치로 먼저 이동해서

2. git rebase main 하면 됩니다. 

3. 그럼 브랜치가 main 브랜치 끝으로 이동하는데 그걸 fast-forward merge 하면 됩니다. 

```
git switch 새로운브랜치
git rebase main

git switch main
git merge 새로운브랜치
```

차례로 입력하면 rebase 끝입니다. 

rebase & merge를 한 줄로 쉽게 비유하자면 **강제 fast-forward merge**입니다. 

직접 새로운 브랜치 만들고 commit 몇 번 하고 rebase 해보십시오.  

당연히 main 말고 다른 브랜치끼리도 가능합니다. 

물론 단점도 있는데 

브랜치끼리 차이가 너무 많은 경우 rebase하면 충돌이 많이 발생할 수 있는데 그거 하나하나 해결하기 귀찮습니다. 

> **squash and merge 하는 경우도 있음**

님들 대충 모든 브랜치를 3-way merge 해버리면 나중에 참사가 일어날 수 있습니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/20151116_merge.png)

왜냐면 

(1) 3-way merge 된 것들은 매우 복잡해보임 

(2) main 브랜치 git log 출력해보면 3-way merge된 브랜치들의 commit 내역도 다 같이 출력되어서 더러워짐 

이런 현상이 있습니다. 

그러기 싫으면 rebase 아니면 squash and merge 하면 됩니다. 

그거 쓰면 새로운 브랜치에 있던 commit 들을 연결해주는게 아니라 똑 떼와서 main 브랜치에 붙여주기 때문에

1번과 2번걱정을 안해도 됩니다. 

rebase는 아까 배웠고

squash and merge 이거 하면 어떻게 되냐면

3-way merge처럼 선으로 이어주지 않고

새 브랜치에 있던 코드변경사항들이 **main 브랜치로** **텔레포트**합니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC2.png)

그럼 이제 main 브랜치의 git log 출력해볼 때

merge 완료된 브랜치의 commit 같은 것들은 출력되지 않습니다. 

그게 왜 좋은거냐고요? 

이런건 님들이 직접 해봐야 체감이 되는 것이기 때문에 3-way merge 많이 해보든가 하십시오. 

```
git switch main
git merge --squash 브랜치명
git commit -m '메세지'
```

squash and merge 하는 법은 그냥 --squash 옵션을 추가하면 끝입니다. 

님들이 브랜치에서 만들어놨던 많은 commit 을 다 합쳐서

하나의 commit으로 main 브랜치에 생성해줍니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/merge5.png)

▲ 그냥 merge 했을 경우

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/merge4.png)

▲ merge --squash 했을 경우 

결과는 둘 다 똑같은데 한 놈은 선으로 이어져있고 한 놈은 텔레포트했을 뿐입니다. 

결론은 :

브랜치 100개 만들어놨는데 일반 merge를 잔뜩 해놓으면 나중에 git log 그래프가 매우 복잡해질 수 있습니다.

그게 싫으면 squash 해보십시오. 또는 rebase 해도 마찬가지로 해결가능합니다. 

> **어떻게 merge 할 지 판단하기 힘들어요**

초보땐 squash 할지 말지 고민하지 말고 대충하십시오.

나중에 코딩노예로 취직하면 중요한 브랜치마다 merge 방법 가이드라인이 있습니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EB%94%B0%EB%B4%89_%ED%8C%80%EC%9B%901.png)

그런거 없으면 퇴사 ㄱ

아니면 기준같은걸 하나 만들어두면 좋습니다. 

**오늘의 숙제 :**

1. 브랜치 새로 하나 만들어서 commit 몇번 한 다음 main 브랜치에 squash and merge 해보십시오.

2. 브랜치 새로 하나 만들어서 commit 몇번 한 다음 main 브랜치로 rebase and merge 해보십시오. 

3. 잘 합쳐져서 이제 쓸모없어진 브랜치는 제거해봅시다.

# 5. 코드짜다가 실수했다 되돌아가자 (git revert, reset, restore)

commit만 주구장창 하는 사람들이 있는데 

git은 버전관리 프로그램이기 때문에 

언제든지 이전 commit으로 되돌아가거나

문제가 되는 commit 내역을 취소하거나 그럴 수 있습니다. 

git restore / git revert / git reset 명령어써서 파일 복구하는 법을 알아봅시다. 

각각 파일하나 복구, commit 복구, 시간되돌리기가 가능합니다. 

깔끔한 상태에서 시작하기 위해

새로운 작업폴더 만들어서 다시 시작합시다. 

> **일단 commit 몇 번 해보고** 

새로운 작업폴더에서 git init 하고 commit 몇 번 해봅시다. 

저는 파일 3개 만들고 만들 때 마다 commit 해봤습니다. 

git log --oneline 입력해보면 여러분의 commit 내역을 한 줄로 이쁘게 보여줍니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%983-1.png)

▲ 왼쪽에 있는 노란 문자들은 **commit의 고유 id** 입니다. 

고유 id를 이용해야 "그 고유 id로 되돌려주세요~" 이런 명령들이 가능합니다.  

> **파일 하나를 되돌리려면 git restore**

파일 하나가 잘못되었을 경우 ctrl + z 여러번 눌러도 되겠지만

수정사항이 너무 많다면 명령어 하나로 처리할 수 있습니다. 

```
git restore 파일명
```

이러면 최근 commit 된 상태로 현재 파일의 수정내역을 되돌릴 수 있습니다. 

```
git restore --source 커밋아이디 파일명
```

이러면 입력한 파일이 특정 커밋아이디 시점으로 복구됩니다. 

```
git restore --staged 파일명
```

이건 복구랑 상관없지만 이러면 특정 파일을 staging 취소할 수 있습니다. 

> **commit을 되돌리려면 git revert** 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/DV4pD0pVAAUVWFf.png)

코드 열심히 짜다가

갑자기 과거 commit 하나가 문제를 일으키면 어떻게 하죠? 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%983-1.png)

위 사진을 보면 지금 commit이 3개 있는데

여기서 b 파일이 문제가 많아서

b 파일을 만든 d874b2b commit을 취소하고 싶어진겁니다. 

commit 하나를 취소하고 싶으면 git revert 사용하면 됩니다. 

실은 없애버리는건 아니고 commit 하나를 취소한 commit을 하나 생성해줍니다. 

```
git revert 커밋아이디
```

이거 입력하면 그 커밋아이디에서 일어난 일만 취소해줍니다. 

실행하면 아마 에디터가 뜰 텐데 맘대로 커밋메세지 수정하고 닫으면 끝입니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%984-1.png)

▲ revert 명령시 가끔 Vim 에디터가 뜨는 사람들이 있을겁니다.

커밋 메세지 수정하라는건데 i 눌러서 글자수정하고 싶으면 하고 esc 눌러서 나올 수 있습니다.

그리고 :wq 누르면 커밋 메세지가 저장됩니다.

아무튼 에디터 닫고나면 새로운 커밋이 생성되고 b파일만 뿅 삭제되어있습니다.

(그 커밋id 이후에 했던 파일이나 커밋들은 영향없이 유지됨)

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%985-2.png)

git log 다시 해보면 revert 해줬다는 commit이 자동으로 생성되어있고 

작업폴더에서 a, c 파일은 있지만 b 파일은 삭제되어있군요.  

결론은 revert 명령어 쓰면 특정 커밋에서 있던 일을 지워버릴 수 있습니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC11-1.png)

▲ 그림 좋아하면 그림보쇼 

(참고)

- revert 할 때 동시에 여러개의 commit id 입력가능 

- 그냥 최근 했던 commit 1개만 revert하고 싶으면 git revert HEAD 하면 편리합니다.

- merge 명령으로 인해 새로 만들어진 commit도 revert 가능합니다. 그럼 merge가 취소되겠군요 필요하면 찾아보셈 

> **그냥 전부 시간을 되돌리고 싶으면 git reset**

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/i15371175073.png)

지옥같은 개발에 대해 아무것도 모르던 어린시절로 되돌아가고 싶습니까? 

현실에선 불가능하지만 git에선 가능합니다. 

git reset 명령어 사용하면 특정 commit 시절로 아예 모든걸 되돌릴 수 있습니다. 

```
git reset --hard 커밋아이디
```

입력하면 그 커밋이 생성될 때로 시간을 되돌려줍니다. 

작업폴더 내의 파일도 그 시절로 돌아갑니다.

작업폴더에서 직접 해봅시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC22.png)

▲ 그림으로 설명하면 이렇게 동작합니다.

commit2로 reset --hard 해버리면

commit2 이후의 미래 기억을 모두 잃습니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/pepe-son.png)

님들 지금 기억을 다 포기하고 인생을 7살로 리셋해준다고 하면 돌아갈 것입니까?

인생 망한놈들은 맘대로 갈 수 있겠지만 

인생이 어느정도 궤도에 오른 사람들은 돌아가기 힘듭니다.

마찬가지로 git reset은 그냥 프로젝트 망하면 쓰거나

아니면 짧은 거리를 돌아갈 때 쓰도록 합시다. 

(참고)

- 여러명이서 협업하는 리포지토리에는 **보통 reset 쓰면 안됩니다.** 갑자기 소스코드가 사라지는거니까요.

- untracked 파일들은 (git add 안해놓은 파일들은) 사라지지않고 유지됩니다. 

- git clean 명령어 찾아서 쓰면 untracked 파일들도 다 지울 수 있습니다. 

> **참고 : reset시 옵션 설정가능**

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%983-1.png)

▲ 아까 상황으로 다시 돌아와서 a, b, c 파일을 만들면서 각각 commit을 했다고 칩시다. 

git reset 뒤에 hard / soft / mixed 설정을 넣을 수 있는데 

```
git reset --hard d874b2b
```

이러면 a, b파일은 남아있고 c 파일이 삭제됩니다. 

```
git reset --soft d874b2b
```

이러면 a, c파일은 남아있고 b 파일은 staging area에 남아있습니다. 

이제 commit 하거나 그럴 수 있습니다. 

```
git reset --mixed d874b2b
```

이러면 a, c파일은 남아있고 b 파일은 staging 되지 않은 상태가 됩니다. 

이제 git add 하고 commit 하거나 그럴 수 있습니다. 

결론은 reset하면서 파일을 아예 지워버리는게 아니라 

검토하고 다시 commit 하고 싶으면 --soft / --mixed 사용해봅시다. 

실은 git reset 어쩌구만 하면 --mixed 옵션이 자동으로 발동됩니다. 

**오늘의 숙제 :** 

파일 만들어서 커밋 몇번 해보고 

git revert 로 원하는 commit 을 취소해보십시오  

직접 안써볼거면 배우는 의미없음 시간낭비임

# 6. Github 사용법 1. 내 코드 올릴 땐 git push

git push, pull 어쩌구 배우기 전에

원격 repository 개념과 왜 사용하는지부터 알고 지나갑시다. 

그래야 자신있게 git push 이런거 사용가능 

> **repository가 뭐냐면** 

git이라는 친구가 파일버전을 저장해두는 장소를 repository라고 합니다.

로컬 작업폴더엔 .git 폴더가 있는데 그게 repository 입니다. 

repository는 한국말로 **저장소**라고합니다. 

실제로 개발할 땐 **온라인 repository**를 많이 사용합니다.

내가 컴퓨터에 만들어 놓은 git repository를 온라인으로 저장해두는겁니다. 

그렇다면 1. EDD-202 다운받다가 컴퓨터 랜섬웨어 걸려도 안심가능 

1. 다른 사람과의 협업도 가능해집니다. 

사람들 많이 쓰는 github.com 에서 온라인 repository 하나 만들어봅시다. 

> **github 가입하고 repository (저장소) 하나 만들기**

원격 저장소를 제공해주는 github.com에 들어가서 가입하고 원격저장소를 하나 만들어봅시다.

로그인 후에 우측 상단 + 버튼 누르면 repository 하나 만들 수 있습니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%981.png)

▲ 이름 대충 짓고 확인만 잘 누르면 됩니다.

그럼 이것저것 git 명령어가 나오는데 그거 따라해도 저장소 사용이 가능한데 

싫으면 절 따라합시다. 

> **일단 작업폴더에서 git으로 commit 몇번 해보셈** 

원격저장소 왜 쓴다고 했습니까.

**내 컴퓨터에서 만들어둔 저장소를 백업**해둘 수 있다고 해서 쓴다고 하지 않았습니까.

그래서 내 컴퓨터에서 만든 로컬저장소를 원격저장소로 백업해봅시다.

일단 새로운 작업폴더에다가 git init 해서 저장소 하나 만들어보십시오. 

1. 작업폴더를 하나 만든 다음 터미널에서 열어서 git init 합니다.

그게 실은 로컬 저장소 (리포지토리) 생성하는 짓임

1. github.com은 이제 기본 브랜치 이름을 master가 아니라 main으로 사용하라고 강요합니다.

그래서 우리 로컬 작업폴더에 있는 기본 브랜치 이름도 main으로 변경해줍니다.

터미널 열어서

```
git branch -M main
```

입력하면 기본 브랜치 이름이 변경됩니다. 

안해도 될 수 있음 

1. 그 다음에 파일같은거 만들어서 commit 몇 번 해보십시오. 

> **Github에서 만든 원격 저장소에 올리기**

로컬저장소 -> 원격저장소

이렇게 업로드하고 싶으면 작업폴더에서 터미널켜서

```
git push -u 원격저장소주소 main
```

하면 됩니다.

- 로컬저장소의 main 브랜치를 원격저장소에 올리라는 뜻입니다. 다른 브랜치도 올릴 수 있음 

- github 로그인하라고 뜨면 로그인하면 됩니다. 

- 참고로 -u 옵션은 방금 입력한 주소 기억해두라는 뜻입니다. 다음부터는 주소를 길게 입력안하고 git push만 입력해도 잘됩니다.  

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%983-1.jpg)

▲ 원격 repository 주소는 이렇게 https:// 부터 시작해서 .git으로 끝납니다.

잘 찾아보십시오. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%984.jpg)

▲ 저런거 안뜨면 주소창에 있는거 그대로 복사해와서 .git만 뒤에 붙이면

그기 님들 원격 repository 접속url입니다. 

아무생각없이 제꺼 그대로 따라치는 사람들이 한 달에 10명씩 있습니다.  

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%985-1.jpg)

▲ 그랬더니 진짜로 로컬에 있던 파일과 commit 내역이 올라갔습니다. 

아무튼 결론은 원할 때 **git push 어쩌구** 하면 여러분이 작업한 파일들을 원격저장소에 업로드할 수 있습니다. 

이제 전 세계 사람들이 여러분의 부끄러운 코드 관람가능 

(참고)

github 사이트에서도 파일 수정삭제, commit 이런 것들 자유롭게 가능합니다. 

github 원격저장소는 비공개로 돌릴 수도 있음 

> **원격저장소주소 길게 입력하는게 귀찮으면** 

그니까 https://github.com/codingapple1/lesson.git 이거 매번 입력하기 귀찮으면 어떻게 하냐는 겁니다. 

그럴 땐 그 주소를 변수에 저장해서 사용할 수 있습니다. 

변수에 저장하려면 

터미널에 git remote add 변수명 저장소주소

입력하면 됩니다. 

```
git remote add origin https://github.com/codingapple1/lesson.git
```

이렇게 입력하면 "https://어쩌구" 주소가 필요할 때 마다 origin 이라는 변수명을 쓸 수 있습니다. 

아까쓰던 지랄맞게 길던 명령어를 git push -u origin main 이렇게 짧고 귀엽게 쓸 수 있겠군요. 

(참고) 실은 -u는 방금 입력한 주소를 기억하라는 뜻이라

-u 붙여서 1번 했었으면 나중엔 git push 까지만 입력해도 알아서 잘됩니다. 

진짜로 git push만 해보셈 

(참고) 변수목록을 살펴보고 싶으면 git remote -v 입력해보십쇼 

> **원격저장소에 있던거 그대로 내려받기**

돈벌어서 맥북을 샀는데 그 컴퓨터에서 갑자기 개발을 시작하고 싶은겁니다.

그럼 귀찮게 컴퓨터간 소스코드를 공유할 필요 없이 

원격저장소에 있던 내용을 그대로 내려받아서 시작하면 편리합니다. 

```
git clone https://원격저장소주소
```

하십쇼 

> **저장소에 올리지 않는 파일들은 .gitignore**

원격저장소를 효율적으로 쓰고 싶으면 쓸데없는 파일은 commit 해서 올리지 않는게 좋습니다.

.gitignore 파일을 하나 만들면 저장소에 올리지 않을 파일들을 쉽게 명시가능합니다.

거기 명시한 파일들은 git add . 해도 스테이징이 되지 않아서 편리합니다.

웹개발을 제일 많이 하니까 웹개발을 예로 들면 

node_modules 이런 폴더, 개인정보들이 들어있는 .env 파일 이런 것들은 안올립니다. 

(어짜피 package.json 파일만 잘 있으면 터미널에서 npm install 입력하면 자동으로 node_modules 폴더가 생성됩니다.)

그래서 .gitignore 파일에 명시해주면 됩니다.

작성하는 법은 필요할 때 구글찾아보십쇼

# 7.  Github 사용법 2. 타인과 협업하기 (git clone, pull)

원격저장소의 장점은 남들과 협업할 수 있다는 겁니다. 

어려운건 아니고 개발자 10명이서 각각 작업한 내용을 원격저장소에 올리면 그게 협업 아니겠습니까 

협업해봅시다. 

> **일이 너무 많아서 코딩노예 1명을 고용했습니다**

물론 여러분은 친구가 없으니까 가상의 팀원을 하나 만들어봅시다. 

코딩노예 팀원이 원격저장소에 있던 코드를 같이 짜고 싶다면 어떻게할까요.

그 친구도 똑같이 코드짜서 git push 어쩌구 하면 그게 협업 끝입니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%83%9D%EA%B0%81_%ED%8C%80%EC%9B%90.png)

**"기존 소스코드가 없는데 코드어떻게 짬?"** 

당연히 코딩노예는 기존 소스코드를 다운받아서 시작할 수 있습니다.  

다운받는 법은 github.com 가서 다운받아도 되고 

```
git clone 원격저장소주소
```

새로운 작업폴더에서 이거 입력해도 됩니다. 

그럼 원격저장소에 있던 내용을 그대로 복제해줍니다. 

새로 폴더만들어서 진짜 그런지 테스트해봅시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%982-1.png)

▲ git clone 했더니 진짜로 저번 시간에 만들었던 리포지토리가 다운받아집니다. 

참고로 필요할 땐 특정 브랜치 1개만 clone 해올 수 있습니다. 필요하면 찾아봅시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EB%94%B0%EB%B4%89_%ED%8C%80%EC%9B%901.png)

이제 팀원도 폴더 열어서 코드짜고 commit 하고 git push 하면 됩니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%981-1.png)

▲ 다만 그 팀원도 github 아이디가 있어야하고

그 팀원의 아이디를 Collaborators 메뉴에 등록해놔야 협업가능합니다.  

> **팀원이 commit 하려는데 문제가 생김** 

git push는 맘대로 할 수 있는게 아닙니다.

**갑자기 다른 놈이 만든 파일이 원격저장소에 생기면 git push 못합니다.** 

예를 들어 github.com 의 여러분 리포지토리로 들아가봅시다. 

github 홈페이지에서 이거저거 눌러서 새로운 파일 하나를 만들어봅시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%986-1.jpg)

▲ 예를 들어 저는 사이트에서 hello라는 파일을 만들고 commit 했습니다.

이런 식으로 원격저장소가 **타인에 의해 업데이트**되었다고 칩시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/dab_.png)

다른 곳에서 일하던 코딩노예 팀원도

방금 만든 파일을 원격저장소에 업로드하고 싶어진겁니다.

예를 들어 hi 라는 파일을 만들었다고 칩시다. 

그럼 이전과 같이 commit 하고 나서 git push 똑같이 하면 됩니다.

(팀원인 척 해줄 친구가 없으면 그냥 원래 코드짜던 곳에서 git push 해봅시다)

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EB%AC%BC%EC%9D%8C%ED%91%9C_%ED%8C%80%EC%9B%90.png)

**"에러나는데요"** 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%983-2.png)

▲ **원격 vs 로컬 내용이 다르다면** 로컬저장소에서 git push가 안됩니다.

왜냐면 그런 상황에서 대충 git push 해버리면 코드가 꼬이기 때문에 얘가 미리 예방해주는 것일 뿐입니다. 

> **git pull 이용하면 현재 원격저장소 내용 가져올 수 있음** 

```
git pull 원격저장소주소
```

이러면 원격저장소에 있던 모든 브랜치 내용을 가져와서 로컬저장소에 합치라는 뜻입니다.

이걸 해주면 로컬이 원격저장소 내용을 반영한 최신상태가 되기 때문에 이제 git push가 가능합니다.

결론은 변동사항이 생겼다면 git pull 하고 나서 git push 하면 됩니다.

(참고)

- git pull 원격저장소주소 브랜치명 입력하면 특정 브랜치만 가져올 수 있습니다. 

- origin이라는 변수명을 등록해놨으면 당연히 사용가능

- 예전에 -u 했었으면 git pull, git push까지만 입력해도 잘됩니다.

> **참고사항 : git pull 명령어는 git fetch + git merge 축약어임** 

git pull 입력하면 자동으로 git fetch + git merge를 해줍니다. 

git fetch는 원격저장소에 있는 commit 중에 로컬에 없는 신규 commit을 가져오라는 뜻이고

git merge는 그걸 merge 하라는 뜻입니다. 

그래서 git pull 할 때 팀원 2명이서 같은 파일을 건드리고 있을 경우 merge conflict가 날 수 있습니다.

conflict는 branch 다룰 때 다뤄봤으니 알아서 해결하면 됩니다. 

그래서 오늘의 결론은

협업시 git push 하기 전에 뭐라그러면 git pull 존나게 하면 됩니다.

# 8. Github 사용법 3. 브랜치로 협업하기 (pull request)

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC45.png)

신기능을 만들고 싶으면 main 브랜치에 코드짜다가 프로젝트 망치지말고

다른 브랜치를 만들어서 거기에 개발하는 것도 안전하고 좋다고 했습니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/branch.png)

원격 repository (저장소) 에도 브랜치를 만들 수 있습니다.

브랜치 생성하려면 1. github.com에서 브랜치 직접 만들어도 되고 

1. 아니면 로컬에서 만든 브랜치를 올려도 브랜치생성이 가능합니다. 

> **1. github 사이트에서 직접 브랜치 생성가능** 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%988.jpg)

▲ main 브랜치명 버튼 누르면 브랜치를 바꾸하거나 새로 만들 수 있습니다.

심심하면 하나 만들어보거나 하면 됩니다. 

> **2. 아니면 로컬 repository 에서도 브랜치생성가능**

로컬저장소에서 브랜치생성해서 원격저장소로 git push 해도 됩니다. 

예를 들면 지금 사이트를 하나 만들고 있는데 

사이트 방문자들 컴퓨터에 몰래 비트코인 채굴기를 심는 기능을 만든다고 칩시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%989.jpg) 

그래서 저번 시간에 git clone으로 복사해온 작업폴더에서

- 새로운 mining 브랜치를 만들고 

- 파일도 하나 새로 만들어서 commit 해봤습니다. 

그 다음에 로컬 브랜치를 원격에 올리고 싶으면

```
git push 원격저장소주소 로컬브랜치명
```

이거 하면 됩니다.  

참고로 

**git push 원격저장소주소 로컬브랜치명** 하면 특정 로컬저장소 브랜치 -> 원격저장소

**git push 원격저장소주소** 하면 모든 로컬저장소 브랜치 -> 원격저장소 입니다.

우리같은 코딩노예들은 특정 브랜치만 올리는 일이 잦습니다. 

> **Pull request 하기** 

브랜치만들면 뭐합니까 그걸 main 브랜치와 합쳐야 기능이 완성되지 않겠습니까. 

합치려면 git merge 명령어로 합치면 끝입니다. 그리고 git push 하면 끝인데

팀끼리 일하는 경우 merge 하기 전에 토론하거나 검토하거나 그래야하는 경우가 많습니다. 

그래서 github.com은 pull request 라는 기능이 있습니다. 

그냥 쉬운 말로 merge request입니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/65f69c29-6ef3-4910-83d0-4fb3c0eabf2afile_575704.png)

이거 누르면 내 브랜치좀 merge 해달라는 요청을 할 수 있고

팀원끼리 merge전에 코드검토가 가능합니다. 

github.com 웹사이트에서 pull request를 열고 싶으면

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%981-2.png)

▲ 아무나 상단 Pull requests 메뉴에서 초록버튼 누르면 pull request 생성이 가능합니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%982-2.png)

▲ 그 다음엔 어떤 브랜치를 어디에 합칠 것인지 선택하고

하단에서 commit 내역, 변경내역 잘 보고

초록버튼 누르면 pull request가 열립니다.

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%984-2.png)

▲ 그럼 Pull requests 메뉴에서 이렇게 확인가능한데 

누르면 코딩노예들이 토론할 수 있는 곳도 있습니다. 거기서 코드를 리뷰하면 됩니다. 

시니어들도 집에서 빤스바람으로 대충 읽고 Looks good to me~! 댓글 남기면 됩니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EC%BA%A1%EC%B2%985-3.png)

▲ 잘 된것 같아서 merge하기로 했으면

merge 할 때 여러가지 옵션이 있는데 택1 하면 됩니다.

사람들 요새 글 안읽으니까 그림으로 봅시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC10.png)

**create a merge commit 하면**

새로운 merge commit을 하나 생성해주는 3-way merge를 실행해줍니다. 

- main 브랜치 조회시 합쳐진 브랜치의 commit 내역도 전부 나옴  

- 터미널에 git log --oneline --graph 해보면 합쳐진 브랜치도 그림으로 나옴 

- 그래서 commit 내역이 많으면 복잡하고 더러워보일 수 있습니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC12.png)

**squash and merge 하면** 

- 합쳐질 브랜치의 commit 내역을 하나로 합쳐서 main 브랜치에 신규 commit을 생성해줍니다.

- git log --oneline --graph 해보면 합쳐진 브랜치 안나옴 

- commit을 하나로 합쳐서 main 브랜치로 순간이동 시켜주는 행위라 사람들이 깔끔하다고 좋아합니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/%EA%B7%B8%EB%A6%BC11-2.png)

**rebase and merge 하면** 

- 합쳐질 브랜치를 main 브랜치 최신 commit으로 rebase하고나서 fast-forward merge 비슷한걸 해줍니다.

- 결과는 squash and merge와 비슷한데 합쳐질 브랜치의 commit 내역이 전부 보존됩니다. 

- 얘도 git log --oneline --graph 해보셈

오늘의 결론 : 

github 등 원격 저장소에도 브랜치만들 수 있습니다. 

Pull request (merge)할 땐 3개 중 맘대로 하면 됩니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/06/peepo-shake-pepe-frog.gif)

이론은 다 배웠다고 해도

실제로 git branch, merge할 때 되면 어떻게 해야할지 손발이 덜덜 떨리는 분들이 있습니다. 

그래서 맨날 사수 찾고 사수 없으면 일 못하고 그런데  

다음 시간에 branching & merge 어떻게 하면 좋은지 전략을 알아봅시다. 

(참고)

원격저장소의 commit 내역을 과거로 되돌리고 싶으면

로컬에서 git reset --hard 이런거 쓰고 git push -f 하면 가능하긴 한데

해당 브랜치를 공동작업중인 사람들이 모두 영향받기 때문에 그러지 않는게 좋습니다.

github.com 사이트엔 revert 버튼이 있긴 한데 그거 쓰면 예전 코드로 되돌려주는 commit을 만들어주는 식으로 동작합니다.

# 9. git flow / trunk-based 브랜치 전략

개발자 10명이서 브랜치를 대충 아무렇게나 만들면 개발과정이 매우 복잡해지고 추적도 어려워서

git branch 깔끔하게 만들도록 도와주는 방법론같은게 있습니다. 

git flow, github flow, gitlab flow, trunk-based 등 다양한 것들이 있습니다.  

이런걸 적용하면

1. 브랜치관리가 쉬워지고 

2. 팀원이 아무리 많아도 개발절차가 매끄러워집니다. 

그래서 프로젝트 리드하는 사람들이 알면 좋습니다. 

시키는 것만 하는 코딩노예들은 몰라도 됩니다. 

> **안정적인 운영이 필요하면 git flow** 

님들이 만드는 프로그램이 항상 안정적인 release를 해야한다면 (예를 들면 게임개발)

git flow 전략을 쓰면 됩니다. 

git flow 전략은 크게 5개 브랜치를 운영하는데 

- main 브랜치

- develop 브랜치 (개발용)

- feature 브랜치 (develop에 기능추가용)

- hotfix 브랜치 (main 브랜치 버그해결용)

- 가끔 release 브랜치 (develop 브랜치를 main 브랜치에 합치기 전에 최종 테스트용)

를 운영합니다. 

이론만 설명하면 노잼이라 게임개발을 예시로 들어봅시다. 

이제부터 여러분은 게임개발 팀장입니다. 

지금까지는 대충 주먹구구식으로 협업해서 0.9버전까지 만들어놨다고 칩시다. 

근데 1.0 버전부터는 신기능도 많고 해서 제대로 개발을 진행하고 싶은겁니다.

그래서 이번엔 git flow를 도입해서 개발을 진행해봅시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/07/%EA%B7%B8%EB%A6%BC4.png)

**1. develop 브랜치부터 생성합니다.** 

신기능 개발해서 바로 main브랜치에 합칠 것입니까?

그래도 되겠지만 신입 개발자들을 믿을 수 없습니다. 

일단 실험용 프로젝트 사본을 만들고 거기다가 먼저 개발해봅시다. 

그러기 위해 main 브랜치에 있던 기존 프로젝트를 복사한 develop 브랜치를 생성합니다. 

이제 모든 개발은 develop 브랜치에서 진행하라고 팀원들에게 전파합니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/07/%EA%B7%B8%EB%A6%BC1.png)

**2. 신기능개발은 feature 브랜치에서 진행**

신기능을 만들고 싶으면 develop 브랜치를 복사한 feature 브랜치에서 각각 개발합니다. 

feature/guild 브랜치 만들어서 길드기능 만들고 

feature/friend 브랜치 만들어서 친구기능 만들고 하면 됩니다. 

(브랜치 작명할 때 여러 단어가 필요하면 보통 대시나 / 기호 씁니다)

- 완성되면 develop 브랜치에 merge 합니다.

- 중요한 내용이 아니면 squash and merge도 괜찮습니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/07/%EA%B7%B8%EB%A6%BC2.png)

**3. 신버전 출시 준비는 release 브랜치**

develop에서 만든 2개 기능들이 완성된 것 같습니다.

이걸 바로 main 브랜치에 합치기엔 또 불안하기 때문에

develop -> release 브랜치 이렇게 프로젝트를 복사한 다음 출시준비를 합니다. 

- 여기서 테스트나 QA같은거 진행하면 됩니다. 

- 버그를 발견하면 알아서 임시 브랜치 만들어서 수정하거나 합니다.

- release/1.0 이런 식으로 이쁘게 브랜치 이름을 짓는 경우가 많습니다.

완성된 것 같으면 main 브랜치로 merge 합니다. 그리고 그거 유저들에게 배포하면 됩니다. 

개발은 계속 진행되어야하니 완성본은 develop 브랜치에도 merge 해줍시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/07/%EA%B7%B8%EB%A6%BC5.png)

**4. hotfix 브랜치**

1.0 버전에서 갑자기 골드 무한복사 버그를 발견했습니다. 

그런 급한 것들은 main 브랜치에서 hotfix 이런 브랜치 하나 만들어서 바로바로 버그수정하면 됩니다. 

- 수정이 완료되면 main 브랜치에 직접 merge 하면 됩니다. 

- 당연히 develop 브랜치에도 merge 해줘야합니다. 

이제 유저들에겐 "잡다한 버그 수정" 공지만 올리고 점검보상 쪼금 주면 됩니다. 

게임 뿐만 아니라 웹이나 앱도 비슷하게 운영할 수 있습니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/07/%EA%B7%B8%EB%A6%BC6.png)

▲ 쓸데없이 다 합친 이쁜 그림 

출처링크 안남기고 개인 블로그에 글과 그림 그대로 복사해가는 나쁜 사람들이 많습니다. 

그래서 오늘은 워터마크를 박아봤습니다. 

**Q. 꼭 저거 따라해야하나요?**

물론 git flow 이런거 단점도 있습니다.

최근 continuous delivery 이런거 한 때 유행이었는데 그런거 할 땐 적합하지 않을 수 있습니다. 

그래서 맨날 남들이 하는거 앵무새처럼 따라할 생각하지 말고 본인 마음대로 변형해서 쓰십시오. 

예를 들면 release 브랜치 쓰지 않고 바로 main 브랜치에 merge 해서 배포하거나 그래도 됩니다. 

그 선택에 합당한 이유와 근거가 있으면 됩니다. 물론 책임도 져야합니다.

근데 책임은 언제나 전가 가능  

> **Trunk-based 전략** 

님들이 만드는게 코드짠걸 바로 대중에 배포를 해도 상관없는 프로그램이면

그리고 크게 대격변 업데이트를 안하는 안정적인 프로그램이면 

굳이 많은 브랜치를 만들 필요가 없습니다. 

그냥 main 브랜치와 기능추가용 feature 브랜치만 운영하면 됩니다. 

이게 trunk-based 전략입니다.

github flow도 이거랑 비슷합니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/07/%EA%B7%B8%EB%A6%BC7.png)

**1. 기능추가, 버그픽스가 필요하면** **main 브랜치에서 새로운 브랜치를 하나 만들어서 코드짭니다.**

브랜치마다 작명 잘하는게 중요합니다. 

**2. 기능이 완성되었으면** **main 브랜치에 합칩니다.**

이제 브랜치 쓸데없으니 삭제합니다.

**3. main 브랜치에 있는 코드를 필요할 때 마다 유저들에게 배포합니다.**

- trunk-based 개발의 장점은 코드를 한 브랜치에서만 관리하기 때문에 편리합니다. 

- 크게 개발해서 한 번에 merge 하는 것 보다 작은 단위로 merge 하는 것이 더 안전합니다. 

- 하지만 main 브랜치에 있는 코드가 뻑이나면 큰일나기 때문에 테스트나 코드리뷰를 자주해야합니다.

그래서 테스트를 자주하고 자동화해놓는 곳들이 제대로 사용가능합니다. 

> **결론**

이미 어느정도 개발이 진척이 되었거나 프로 코딩전사들로 가득한 팀이면 trunk-based 이런거 쓰는게 훨씬 편리합니다.

최근 유행한 CI/CD 이런 식으로 개발하는 곳들도 trunk-based 개발방식을 적용합니다.  

출시된 버전의 안정성이 중요한 프로그램들, 아직 뼈대가 확실하지 않아 연구식으로 개발하는 프로그램들은 git flow가 적절할 수 있습니다. 

하지만 물론 정해진 것은 없고 직접 해보고 판단하는게 좋습니다.

**Q. merge 할 때 어떤 방법 쓰는게 좋은가요?**

기록을 남겨야하는 중요한 브랜치를 merge할 땐 3-way merge

기록을 남길 필요없는 쓸데없는 브랜치를 merge할 땐 squash, rebase 쓰면 됩니다. 

취향일 뿐이고 알아서합시다.

# 10. git stash로 코드 잠깐 보관하기

방금 쓰레기같이 짜놓은 코드가 있다고 칩시다. 

그 코드를 잠깐 치워놓고 개발하고 싶으면 주석처리해도 되겠지만 

git stash 명령어를 이용해도 잠깐 코드를 치울 수 있습니다. 

> **git stash 사용해서 코드 잠깐 다른 곳에 보관하기**

```
aaaaaaaaaaaaa
```

파일 하나 만들어서 이렇게 코드를 짜서 commit 해봅시다. 

```
aaaaaaaaaaaaa
bbbbbbbbbb
```

그리고 밑에 bbbbb어쩌구 코드를 짜놨다고 칩시다.

근데 밑에 코드가 마음에 안드는겁니다.

이걸 잠깐 삭제해버리고 싶으면 git stash 명령어를 씁시다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/07/%EA%B7%B8%EB%A6%BC1-1.png)

```
git stash
```

터미널에 git stash 입력하면 방금 작성한 bbbb 어쩌구 코드는 잠깐 다른 공간에 보관됩니다. 

(그래서 파일들이 최근 commit 상태로 되돌아갑니다.)

- staging 된 것이든 안된 것이든 추적중인 파일은 다 이동됩니다.

- 새로 만든 파일인데 staging 안되었다면 이동안됩니다. 

```
git stash save "bbb 코드짰는데 망함"
```

git stash 할 때 메모도 함께 입력할 수도 있습니다. 

```
git stash list
```

git stash는 여러번 할 수 있습니다.

현재 stash 되어있는 코드 목록을 전부 출력해주는 명령어입니다. 

> **보관했던 코드 다시 불러오기** 

```
git stash pop
```

이러면 잠깐 보관했던 코드를 다시 불러옵니다. 

git stash 했던 코드가 여러개 있으면 가장 최근에 보관했던 코드부터 먼저 불러옵니다. 

![](https://codingapple-cdn.b-cdn.net/wp-content/uploads/2022/07/%EA%B7%B8%EB%A6%BC3.png)

▲ 가장 최근에 들어온 것 부터 먼저나갑니다. 

물론 현재 코드와 겹치는 부분이 있으면 conflict 나니까 알아서 해결하면 됩니다. 

> **stash 관련 여러 명령어들**

```
git stash drop 삭제할id
git stash clear
```

위에건 특정 stash 삭제, 

밑에건 모든 stash 삭제하는 명령어입니다.

삭제할 id는 git stash list 하면 보이는 0, 1, 2 이런 숫자 넣으면 됩니다. 

```
git stash -p 
```

전체 말고 일부 코드만 git stash 하고싶으면 이거 씁시다.

그럼 파일을 훑어주면서 stash 할 지 의견을 물어보는데 y/n 으로 잘 대답하면 됩니다. 

**Q. 주석처리해놓는게 더 쉬울듯** 

실은 코드를 주석처리하는거랑 용도가 비슷하긴 한데 

주석처리된 코드는 commit할 때 반영됩니다. 그렇게 되면 주석도 commit 기록에 남아서 기록이 더러워질 수 있습니다. 

주석처리한 내용을 commit 해버리기 싫을 때 git stash 쓰면 유용합니다. 

또는 기능 A, B를 만들어야하는데

기능A는 완성되었고 기능B는 반쯤 완성된 경우 

팀장이 "기능A 부분만 빨리 commit하고 merge 하라"고 하면

그럴 때 기능B를  git stash 해놔도 좋을듯요

**Q. 브랜치 새로 만들어서 거기다 코드짜놓는거랑 다를바 없는데요?**

들킴

간단히 브랜치만들어서 거기 보관하는 것도 나쁘지않습니다.
