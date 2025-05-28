## 📌 개요

이 문서는 Java Spring Boot를 기반으로 OpenAI의 `gpt-4o-mini` 모델에 요청을 보내, IT 면접 질문에 대한 심층 꼬리질문 4개를 자동 생성하는 시스템을 구축하는 방법을 설명합니다.

---

## ✅ 1. 프롬프트 템플릿

프롬프트는 꼬리 질문 4개를 JSON 배열 형식으로 생성하도록 요구합니다.

### 🔧 프롬프트 구조

```
당신은 IT 직무 면접 질문 생성 AI입니다. 주어진 정보를 바탕으로, 메인 질문에 대한 심층적인 꼬리 질문 4개를 한국어로 생성해 주세요.

[메인 질문]  
{{selected_question}}

[사용자 키워드] (주어진 경우에만 반영)  
{{keyword}}

[이전 질문 목록] (주어진 경우에만 반영)  
{{passed_questions}}

[조건]  
1. 사용자 키워드가 있다면, 각 질문에 해당 키워드를 **자연스럽게 무조건 반영**하세요.  
2. 생성되는 질문은 [메인 질문]과 **내용이 겹치면 안 됩니다.**  
3. [이전 질문 목록]이 있다면, 해당 질문들과도 **내용이 중복되지 않도록** 하세요.  
4. 각 질문은 서로 **다른 관점**을 반영해야 합니다.  
5. 모든 질문은 **완전한 문장 형태**여야 하며, `"..."`, 생략 부호, 명확하지 않은 표현은 절대 사용하지 마세요.  
6. **단답형 질문**이나 **너무 광범위한 질문**은 피해주세요.  
7. 각 질문은 **한글 기준 100자 이내**로 작성해 주세요.  
8. 결과는 반드시 아래 JSON 형식으로 출력해 주세요:

```json
[
  "질문 1",
  "질문 2",
  "질문 3",
  "질문 4"
]
```
```

---

## ✅ 2. Gradle 설정

### 📄 `build.gradle`

```groovy
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation 'com.fasterxml.jackson.core:jackson-databind'
    implementation 'org.springframework.boot:spring-boot-starter-json'
    implementation 'org.apache.httpcomponents.client5:httpclient5:5.1.3'
}
```

---

## ✅ 3. Java 코드

### 📄 `OpenAiService.java`

```java
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Service
public class OpenAiService {

    private final String OPENAI_API_KEY = "sk-xxxx"; // OpenAI 키 입력
    private final String OPENAI_URL = "https://api.openai.com/v1/chat/completions";

    public List<String> generateFollowupQuestions(String selectedQuestion, String keyword, List<String> passedQuestions) throws Exception {
        RestTemplate restTemplate = new RestTemplate();
        ObjectMapper mapper = new ObjectMapper();

        String prompt = buildPrompt(selectedQuestion, keyword, passedQuestions);

        Map<String, Object> message = Map.of("role", "user", "content", prompt);
        Map<String, Object> request = Map.of(
            "model", "gpt-4o-mini",  // gpt-4o-mini 모델 사용
            "messages", List.of(message),
            "temperature", 0.8
        );

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(OPENAI_API_KEY);

        HttpEntity<String> entity = new HttpEntity<>(mapper.writeValueAsString(request), headers);
        ResponseEntity<String> response = restTemplate.exchange(OPENAI_URL, HttpMethod.POST, entity, String.class);

        JsonNode jsonNode = mapper.readTree(response.getBody());
        String content = jsonNode.get("choices").get(0).get("message").get("content").asText();

        return mapper.readValue(content, List.class);
    }

    private String buildPrompt(String selectedQuestion, String keyword, List<String> passedQuestions) {
        StringBuilder sb = new StringBuilder();
        sb.append("당신은 IT 직무 면접 질문 생성 AI입니다. 주어진 정보를 바탕으로, 메인 질문에 대한 심층적인 꼬리 질문 4개를 한국어로 생성해 주세요.\n\n");
        sb.append("[메인 질문]\n").append(selectedQuestion).append("\n\n");

        if (keyword != null && !keyword.isBlank()) {
            sb.append("[사용자 키워드]\n").append(keyword).append("\n\n");
        }

        if (passedQuestions != null && !passedQuestions.isEmpty()) {
            sb.append("[이전 질문 목록]\n").append(String.join("\n", passedQuestions)).append("\n\n");
        }

        sb.append("[조건]\n");
        sb.append("1. 사용자 키워드가 있다면, 각 질문에 해당 키워드를 **자연스럽게 무조건 반영**하세요.\n");
        sb.append("2. 생성되는 질문은 [메인 질문]과 **내용이 겹치면 안 됩니다.**\n");
        sb.append("3. [이전 질문 목록]이 있다면, 해당 질문들과도 **내용이 중복되지 않도록** 하세요.\n");
        sb.append("4. 각 질문은 서로 **다른 관점**을 반영해야 합니다.\n");
        sb.append("5. 모든 질문은 **완전한 문장 형태**여야 하며, `\"...\"`, 생략 부호, 명확하지 않은 표현은 절대 사용하지 마세요.\n");
        sb.append("6. **단답형 질문**이나 **너무 광범위한 질문**은 피해주세요.\n");
        sb.append("7. 각 질문은 **한글 기준 100자 이내**로 작성해 주세요.\n");
        sb.append("8. 결과는 반드시 아래 JSON 형식으로 출력해 주세요:\n\n");
        sb.append("[\n  \"질문 1\",\n  \"질문 2\",\n  \"질문 3\",\n  \"질문 4\"\n]");

        return sb.toString();
    }
}

```

---

## ✅ 4. 사용 예시

```java
List<String> passed = List.of("클라우드 환경에서의 서비스 배포 경험이 있나요?");
List<String> questions = openAiService.generateFollowupQuestions(
    "Spring Boot에서 의존성 주입 방식에 대해 설명해 주세요.",
    "DI",
    passed
);
questions.forEach(System.out::println);
```

---

## ✅ 5. 추가 고려사항

- `gpt-4o-mini`는 OpenAI에서 직접 제공하는 모델인지 확인 필요 (조직 내 커스텀 모델일 수 있음)
- 응답 형식이 JSON 배열로 나오는지 항상 검증할 것
- 에러 응답(`model_not_found` 등) 발생 시 모델 이름 확인
- 향후 LangChain 또는 Langfuse 연동 시에도 프롬프트와 출력 포맷을 유지하면 일관된 결과 확보 가능

---

## ✅ 예시 결과 (예상)

```json
[
  "의존성 주입 시 생성자 방식의 장단점에 대해 설명해 주세요.",
  "Spring Bean 간의 순환 참조 문제는 어떻게 해결하셨나요?",
  "DI를 활용한 테스트 코드 작성 경험에 대해 설명해 주세요.",
  "DI가 적용된 프로젝트에서 설정 변경 시 유연성 확보 방안은 무엇인가요?"
]
```

---