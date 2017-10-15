def getWhere(location):
    # 위치에 따라 숫자인 데이터만 가져오도록 WHERE 절 생성
    if(location == 'COMMON'):
        query = "\n    AND D_PALDANG REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n    AND D_36T REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n    AND D_GWANGAM REGEXP '^[0-9]+\\.?[0-9]*$'" \
                + "\n    AND D_YANGJECHEON REGEXP '^[0-9]+\\.?[0-9]*$'"
    elif(location == 'HONGTONG'):
        query = "\n    AND D_HONGTONG REGEXP '^[0-9]+\\.?[0-9]*$'"
    elif(location == 'SEONGSAN_GIMPO'):
        query = "\n    AND D_SEONGSAN_GIMPO REGEXP '^[0-9]+\\.?[0-9]*$'"
    return query

def getSelect(location):
    if(location == 'HONGTONG'):
        query = "CONVERT(D_PALDANG, DOUBLE) + CONVERT(D_36T, DOUBLE) - (CONVERT(D_GWANGAM, DOUBLE) + CONVERT(D_HONGTONG, DOUBLE) + CONVERT(D_YANGJECHEON, DOUBLE))"
    elif(location == 'SEONGSAN_GIMPO'):
        query = "CONVERT(D_PALDANG, DOUBLE) + CONVERT(D_36T, DOUBLE) - (CONVERT(D_GWANGAM, DOUBLE) + CONVERT(D_SEONGSAN_GIMPO, DOUBLE) + CONVERT(D_YANGJECHEON, DOUBLE))"
    return query