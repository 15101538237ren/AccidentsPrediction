# -*- coding: utf-8 -*-
import scrapy,datetime,calendar,re,urllib,os
from scrapy.selector import HtmlXPathSelector
from weather_spider import add_months

wrt_file_path = '/Users/Ren/PycharmProjects/AccidentsPrediction/preprocessing/data/air_houbao.csv'

class Air_HoubaoSpider(scrapy.Spider):
    name = "air_houbao"
    allowed_domains = ["tianqihoubao.com"]
    start_urls = []
    start_date = datetime.date(year = 2016, month=1,day=1)

    for inc in range(14):
        date_now = add_months(start_date, inc)
        date_str = date_now.strftime("%Y%m")
        url_str = 'http://www.tianqihoubao.com/aqi/beijing-'+date_str+'.html'
        start_urls.append(url_str)
    if os.path.exists(wrt_file_path):
       os.remove(wrt_file_path)
    def parse(self, response):
       hxs = HtmlXPathSelector(response)#创建查询对象
       out_file = open(wrt_file_path,"a")
       # 如果url是 http://www.xiaohuar.com/list-1-\d+.html
       if re.match('http://www.tianqihoubao.com/aqi/beijing-\d+.html', response.url): #如果url能够匹配到需要爬取的url，即本站url
           base_xpath = '//div[@class="api_month_list"]/table/tr'
           items = hxs.xpath(base_xpath) #select中填写查询目标，按scrapy查询语法书写
           item_len = len(items)
           print "item_len: %d" % item_len
           for i in range(2, item_len+1):
               #第一行是title
               dt = hxs.xpath(base_xpath +'[' + str(i) +']/td[1]/text()').extract()[0].strip()
               aqi = hxs.xpath(base_xpath +'[' + str(i) +']/td[3]/text()').extract()[0].strip()
               pm25 = hxs.xpath(base_xpath +'[' + str(i) +']/td[5]/text()').extract()[0].strip()
               out_str = u"%s,%s,%s\n" % (dt,aqi,pm25)
               #print out_str
               out_file.write(out_str.encode('utf-8'))
       out_file.close()
