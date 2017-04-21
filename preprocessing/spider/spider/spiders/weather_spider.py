# -*- coding: utf-8 -*-
import scrapy,datetime,calendar,re,urllib,os
from scrapy.selector import HtmlXPathSelector

wrt_file_path1 = '/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/weather.csv'

def add_months(dt,months):
    month = dt.month - 1 + months
    year = dt.year + month / 12
    month = month % 12 + 1
    day = min(dt.day,calendar.monthrange(year,month)[1])
    return dt.replace(year=year, month=month, day=day)

class WeatherSpider(scrapy.Spider):

    name = "weather"
    allowed_domains = ["lishi.tianqi.com"]
    start_urls = []
    start_date = datetime.date(year = 2016, month=1,day=1)

    for inc in range(14):
        date_now = add_months(start_date, inc)
        date_str = date_now.strftime("%Y%m")
        url_str = 'http://lishi.tianqi.com/beijing/'+date_str+'.html'
        start_urls.append(url_str)
    if os.path.exists(wrt_file_path1):
       os.remove(wrt_file_path1)
    def parse(self, response):

       hxs = HtmlXPathSelector(response)#创建查询对象
       out_file = open(wrt_file_path1,"a")
       # 如果url是 http://www.xiaohuar.com/list-1-\d+.html
       if re.match('http://lishi.tianqi.com/beijing/\d+.html', response.url): #如果url能够匹配到需要爬取的url，即本站url
           base_xpath = '//div[@class="tqtongji2"]/ul'
           items = hxs.xpath(base_xpath) #select中填写查询目标，按scrapy查询语法书写
           item_len = len(items)
           for i in range(2, item_len+1):
               #第一行是title
               dt = hxs.xpath(base_xpath +'[' + str(i) +']/li[1]/a/text()').extract()[0]
               ht = hxs.xpath(base_xpath +'[' + str(i) +']/li[2]/text()').extract()[0]
               lt = hxs.xpath(base_xpath +'[' + str(i) +']/li[3]/text()').extract()[0]
               weather = hxs.xpath(base_xpath +'[' + str(i) +']/li[4]/text()').extract()[0]
               wind_scale = hxs.xpath(base_xpath +'[' + str(i) +']/li[6]/text()').extract()[0]
               out_str = u"%s\t%s\t%s\t%s\t%s\n" % (dt,ht,lt,weather,wind_scale)
               out_file.write(out_str.encode('utf-8'))
       out_file.close()
