with open("stock_archive.py", "r", encoding="utf-8") as f:
    code = f.read()

# parse_one 내부에서 sector를 추출하는 코드가 없다면 추가
target = 'full["close"] = to_num(blind.get("current") or blind.get("현재가"))'
# 안전하게 전체 파일을 다시 정돈된 버전으로 덮어씁니다.
