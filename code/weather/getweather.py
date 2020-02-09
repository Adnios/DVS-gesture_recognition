import json
import io
import sys
import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import urllib.error
import csv
import time
import random
import socket

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码, 防止控制台打印乱码


def get_soup(url):
    try:
        #socket.setdefaulttimeout(20)  # 设置socket默认的等待时间，在read超时后能自动往下继续跑
        head = {}
        head['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36'
        req = urllib.request.Request(url, head)
        # proxy='47.112.214.45:8000'
        # # proxies = ['118.24.246.249:80', '120.79.193.230:8000', '106.85.128.2:9999', '118.25.13.185:8118']
        # # proxy = random.choice(proxies)
        # proxy_support = urllib.request.ProxyHandler({'http': proxy})  # 使用代理之后会出现其他的图，不是想要的
        # opener = urllib.request.build_opener(proxy_support)
        # opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36')]
        # urllib.request.install_opener(opener)

        response = urllib.request.urlopen(url)
        soup = BeautifulSoup(response, 'lxml')
        response.close()
        return soup
    except:
        print(urllib.error.URLError)
        time.sleep(3)
        get_soup(url)
    # except urllib.error.URLError:
    #     return "URL Error"
    time.sleep(1)


def get_data():
    file = open("./city.txt", 'rb')
    filepath = './Weather_info.csv'
    with open(filepath, 'a', encoding='gbk') as f:
        row = ['日期', '天气状况', '气温', '风力风向','城市']
        write = csv.writer(f, lineterminator='\n')
        write.writerow(row)
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        print(dic)
        papers.append(dic)

    for line in papers:
        print(line)
        # for i in range(2011,2020):
        #     for j in range(1,13):
        i=2020
        j=1
        date=""
        date=202001
        url = "http://www.tianqihoubao.com/lishi/"+str(line[0])+"/month/202001.html"
        soup = get_soup(url)
        try:
            all_weather = soup.find("div", class_="wdetail").find("table").find_all("tr")
        except AttributeError as e:
            time.sleep(1)
            soup = get_soup(url)
            all_weather = soup.find("div", class_="wdetail").find("table").find_all("tr")
        if len(all_weather)!=1:
            data = list()
            tr=all_weather[1]
            td_li = tr.find_all("td")
            for td in td_li:
                s = td.get_text()
                data.append("".join(s.split()))
            data.append(line)
            res = np.array(data).reshape(-1, 5)
            print(res)
            result_weather = pd.DataFrame(res)
            result_weather.to_csv(filepath, mode='a', encoding='gbk',index=0,header=0)
            #index=0不使用行索引，header=0不使用列名
            # print('Save '+str(line[0])+ date+' weather success!')
                # else:
                #     list1=[]
                #     if j in (1,3,5,7,8,10,12):
                #         days=31
                #     else:
                #         days=30
                #     date_list = []
                #     for k in range(1,days+1):
                #         date_str=str(i)+'年'+str(j)+'月'+str(k)+'日'
                #         date_list.append(date_str)
                #     date_list = np.array(date_list)
                #     array1 = np.array([['缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据'],
                #                        ['缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据'],
                #                        ['缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据', '缺失数据',
                #                         '缺失数据']])
                #     array1 = array1.reshape(10, 3)
                #     array2 = np.insert(array1, 0, values=array1, axis=0)
                #     array3 = np.insert(array2, 0, values=array1, axis=0)
                #     if j in (1, 3, 5, 7, 8, 10, 12):
                #         array3 = np.insert(array3, 0, values=['缺失数据', '缺失数据', '缺失数据'], axis=0)
                #     array4 = np.insert(array3, 0, values=date_list, axis=1)
                #     empty = np.array([range(1, days + 1)])
                #     res = np.insert(array4, 0, values=empty, axis=1)
                #     empty_weather = pd.DataFrame(res)
                #     empty_weather.to_csv(filepath, mode='a', encoding='gbk', index=0,header=0)
                #     print(str(line[0]) + date + '缺失数据')

        # print('Save '+str(line[0])+' weather success!')

if __name__ == '__main__':
    get_data()