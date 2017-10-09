from math import sqrt

# Pearson correlation coefficient
def sim_pearson(prefs, p1, p2):
    #  같이 평가한 항목들의 목록을 구함
    si = dict()

    for item in prefs[p1]:
        if item in prefs[p2]: si[item] = 1

    # 공통 항목 개수
    n = len(si)

    # 공통 항목이 없으면 0 리턴
    if n==0: return 0

    # 모든 선호도를 합산
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])

    # 제곱의 합을 계산
    sum1Sq = sum([(prefs[p1][it])**2 for it in si])
    sum2Sq = sum([(prefs[p2][it])**2 for it in si])

    # 곱의 합을 계산
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    # 피어슨 점수 계산
    num = pSum - (sum1*sum2/n)
    den = sqrt((sum1Sq-pow(sum1,2)/n) * (sum2Sq-pow(sum2,2)/n))
    if den==0: return 0

    r = num/den

    return r