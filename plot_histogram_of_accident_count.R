require(ggplot2)
require(Cairo)
library("scales")
base_dir = '/Users/Ren/PycharmProjects/AccidentsPrediction';
setwd(base_dir)
data_fp = paste(base_dir, 'static','accident_count3.tsv',sep = '/')
accidents = read.csv(data_fp,header = T, na.strings = "-1",sep = '\t',stringsAsFactors = F)
#acc_ts = data.frame(Time=c(as.POSIXct(as.Date(accidents$time, format = "%Y/%m/%d %H:%M"))),Cnt=c(accidents$cnt))
#plot1 = ggplot(acc_ts,aes(x=Time,y=Cnt)) + geom_line(colour = 'blue') + xlab('Time') + ylab('Accident Count')+scale_x_datetime(breaks = date_breaks("2 days"),labels = date_format("%m-%d"))
#ggsave("ts_of_acc.pdf", plot1, width  = 15, height = 3.15) 
res = 300
wid = 15
heit = 3
pointsz = 5
png(file = "ts_of_acc3.png",width = wid*res,height = heit*res, res = res,pointsize = pointsz)
barplot(accidents$cnt,col = "blue",border = "white")
dev.off()