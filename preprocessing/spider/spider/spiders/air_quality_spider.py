# -*- coding: utf-8 -*-
import scrapy,datetime,calendar,re,urllib,os
from scrapy.selector import HtmlXPathSelector
from weather_spider import add_months

wrt_file_path = '/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/air.csv'

class AirSpider(scrapy.Spider):
    name = "air"
    allowed_domains = ["lishi.tianqi.com"]
    start_urls = []
    start_date = datetime.date(year = 2016, month=1,day=1)

    for inc in range(14):
        date_now = add_months(start_date, inc)
        date_str = date_now.strftime("%Y%m")
        url_str = 'http://lishi.tianqi.com/pm25/beijing_'+date_str+'.html'
        start_urls.append(url_str)
    if os.path.exists(wrt_file_path):
       os.remove(wrt_file_path)
    def parse(self, response):
       hxs = HtmlXPathSelector(response)#创建查询对象
       out_file = open(wrt_file_path,"a")
       # 如果url是 http://www.xiaohuar.com/list-1-\d+.html
       if re.match('http://lishi.tianqi.com/pm25/beijing_\d+.html', response.url): #如果url能够匹配到需要爬取的url，即本站url
           base_xpath = '//div[@class="kqzl_box2"]/ul'
           items = hxs.xpath(base_xpath) #select中填写查询目标，按scrapy查询语法书写
           item_len = len(items)
           for i in range(2, item_len+1):
               #第一行是title
               dt = hxs.xpath(base_xpath +'[' + str(i) +']/li[1]/a/text()').extract()[0]
               aqi = hxs.xpath(base_xpath +'[' + str(i) +']/li[2]/text()').extract()[0]
               pm25 = hxs.xpath(base_xpath +'[' + str(i) +']/li[3]/text()').extract()[0]
               out_str = u"%s,%s,%s\n" % (dt,aqi,pm25)
               #print out_str
               out_file.write(out_str.encode('utf-8'))
       out_file.close()
