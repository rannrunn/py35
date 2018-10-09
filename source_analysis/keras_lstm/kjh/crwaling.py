import urllib

from bs4 import BeautifulSoup

import pandas as pd



def getStockPrice(code, page):

    pageContents = []

    for i in range(1, page+1):

        pageAddress = "http://finance.daum.net/item/quote_yyyymmdd_sub.daum?page="+str(i)+"&code="+str(code)+"&modify=1"

        pageUrl = urllib.request.urlopen(pageAddress)

        Soup = BeautifulSoup(pageUrl, "lxml")

        Profile = Soup.findAll('td', attrs={'class':'num'})

        pageDate = Soup.findAll('td', attrs={'class':'datetime2'})



        for j in range(len(pageDate)):

            rowContents = []

            rowDate = pageDate[j].text.strip()

            rowOpen = Profile[7*j].text.strip().replace(",", "")

            rowHigh = Profile[7*j+1].text.strip().replace(",", "")

            rowLow = Profile[7*j+2].text.strip().replace(",", "")

            rowClose = Profile[7*j+3].text.strip().replace(",", "")

            rowChange = Profile[7*j+4].text.strip().replace(",", "")

            rowReturn = Profile[7*j+5].text.strip().replace("%", "")

            rowVolume = Profile[7*j+6].text.strip().replace(",", "")



            rowContents.append(rowDate)

            rowContents.append(rowOpen)

            rowContents.append(rowHigh)

            rowContents.append(rowLow)

            rowContents.append(rowClose)

            rowContents.append(rowChange)

            rowContents.append(rowReturn)

            rowContents.append(rowVolume)





            pageContents.append(rowContents)



    my_df = pd.DataFrame(pageContents)

    my_df.to_csv(code+'.csv')





getStockPrice('005930', 1)


