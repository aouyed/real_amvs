from datetime import datetime 

MONTH=datetime(2020,1,1)
#MONTH=datetime(2020,7,1)
month_string=MONTH.strftime("%B").lower()
QC='noqc'
THRESHOLDS=[4,10,100]
GEODESICS={'swath':[(-47.5, -60), (45, -30),'latitude'],
                'equator':[(6.5, -149.5),(6.5, 4.5),'longitude']}
#GEODESICS={'swath':[(-50, -130), (46, -91),'latitude'],
 #               'equator':[(6.5, -149.5),(6.5, 4.5),'longitude']}
PRESSURES=[850, 700, 500, 400]