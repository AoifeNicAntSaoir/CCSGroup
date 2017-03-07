import inline as inline
import pandas
#pandas library
result = pandas.read_csv("RecentPageload.csv")

print "\t\t\tprint result"
print result
print "--------------------------------------------------------------------"
print "\t\t\tresult['Location']"
print result['Location']
print "-----------------------------------------------------------------------"
print "\t\t\tresult.columns"
print(result.columns)
print "----------------------------------------------------------------------"
print "\t\t\tresult.shape"
print result.shape
print "----------------------------------------------------------------------"
print "\t\t\tresult.iloc[10:20]"
someresults = result.iloc[10:20]
print someresults

print "---------------------------------------------"
print result[["Location", "Host Name", "IP Address", "Time"]]

#reviews["score"].mean()
print "result.count()"
print result.count()
print "result[""Host Name""].count()"
print result["Host Name"].count()
print result.max()
print result.min()
print result.median()
print result.std

#reviews["score"] / 2

lecFilter =  (result["Host Name"] == "EOBO Limited") & (result["IP Address"] == "88.87.168.242")
print lecFilter



