import urllib.request
from bs4 import BeautifulSoup

def setWord(str):

    request = urllib.request.Request("http://m.endic.naver.com/search.nhn?searchOption=all&query=" + (str.replace(' ','%20')))
    #공백 %20으로 치환
    print("http://m.endic.naver.com/search.nhn?searchOption=all&query=" + (str.replace(' ','%20')))
    data = urllib.request.urlopen(request).read().decode() #UTF-8 encode
    bs=BeautifulSoup(data,'lxml')

    return bs


if __name__ == "__main__" :

    inputStr = "consist"

    bsStart = setWord(inputStr)

    entry = bsStart.find_all('div',attrs={'class':'entry_search_word top'})

    print(entry)

    for eachEntry in entry:

        word = eachEntry.find('a',attrs={'class':"h_word"})
        wordMean = eachEntry.find_all('li')

        print(word.text.strip()) #단어
        for eachMean in wordMean:
            print(eachMean.text.strip())

        if eachEntry.find('p',attrs={'class':'example_mean'}):
            wordEx = eachEntry.find('p',attrs={'class':'example_stc'}).find_all('a')
            wordExMean = eachEntry.find('p',attrs={'class':'example_mean'})

            print()

            for eachwordEx in wordEx: #예문은 하나씩 조합해야됨
                if eachwordEx.find('em',attrs={'class':"u_hc"}):
                    continue
                print(eachwordEx.text, end=' ')

            print()

            print(wordExMean.text.strip())

            print()

    print()