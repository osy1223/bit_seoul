from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
from urllib.parse import quote_plus
from selenium import webdriver
import time

flatfrom=input('사이트 : ')
search=input('검색 : ')
n=int(input('마지막 이미지 번호 : '))+1

if '네이버' == flatfrom:
    url=f'https://search.naver.com/search.naver?where=image&sm=tab_jum&query={quote_plus(search)}'


    driver=webdriver.Chrome('./homework/project1/chromedriver.exe')
    driver.get(url)
    # for i in range(20) :
    #     driver.execute_script("window.scrollBy(0,10000)")
    #     time.sleep(0.5)

    html=driver.page_source
    soup=BeautifulSoup(html)
    img=soup.select('._img')
    imgurl=[]

    for i in img :
        try :
            imgurl.append(i.attrs["src"])
        except :
            imgurl.append(i.attrs["data-src"])

    for i in imgurl :
        urlretrieve(i, "./homework/project1/tmp/ball_origin/"+str(n)+".jpg")
        n+=1
        print(imgurl)



elif '구글' == flatfrom:
    url=f'https://www.google.com/search?q={quote_plus(search)}&rlz=1C1CHZN_koKR926KR926&sxsrf=ALeKk02D62tpvik4raiNBDuxquZbgn-LJQ:1606378974764&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjNgamB5J_tAhXEFogKHfMHBU4Q_AUoAXoECAkQAw&biw=1920&bih=969#imgrc=upw4mNbY0uWHOM&imgdii=xhE_tdSVkbu9iM'

    driver=webdriver.Chrome('./homework/project1/chromedriver.exe')
    driver.get(url)
    for i in range(20) :
        driver.execute_script("window.scrollBy(0,10000)")
        time.sleep(0.5)

    html=driver.page_source
    soup=BeautifulSoup(html)
    img=soup.select('.rg_i.Q4LuWd')
    imgurl=[]

    for i in img :
        try :
            imgurl.append(i.attrs["src"])
        except :
            imgurl.append(i.attrs["data-src"])

    for i in imgurl :
        urlretrieve(i, "./homework/project1/tmp/ball_origin/"+str(n)+".jpg")
        n+=1
        print(imgurl)

driver.close()