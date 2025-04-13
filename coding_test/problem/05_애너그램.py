class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = collections.defaultdict(list)
        for word in strs:
            anagrams[''.join(sorted(word))].append(word)
        return list(anagrams.values())


# sorted() => 문자열도 잘 정렬하며 결과를 리스트 형태로 리턴 => 같은 길이의 문자열은 sorted을 하면 동일함
# sorted(word)는 입력받은 단어를 알파벳 순서대로 정렬한 리스트를 반환합니다.
# 예시: 'eat' → ['a', 'e', 't']
# 이를 다시 키로 사용하기 위해 join()을 사용
# 애너그램끼리 같은 키를 가지게 되고 values로 존재하게 된다.
# 올바른 키 설정
my_dict = {}
my_dict['hello'] = 'world'  # 키는 문자열 'hello'
my_dict[('a', 'b')] = 'tuple key'  # 키는 튜플 ('a', 'b') (immutable)
# 따라서, value 값을 계속 추가하기 위해 append을 하면
{'aet': ['eat', 'tea', 'ate'],
 'ant': ['nat', 'tan'],
 'abt': ['bat']}  # 이러한 형태가 된다.
# dictionary가 되는 데 이를 list형태로 values값만 빼내려면 list(anagrams.values())으로 사용해야한다.
