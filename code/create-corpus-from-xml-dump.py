import pandas as pd
import bs4 as bs
import csv

# ebsco database search returns xml file with full bibliographic metadata records
# need to extract the year and abstract for each record

# load xml file as beautiful soup (bs4) object
file = open("data/7b73469b-082f-4414-a5a0-fd79c5e7fa33.xml", "r")
contents = file.read()
soup = bs.BeautifulSoup(contents, 'xml')

# iterate through each record, append abstract and year data to lists
year = []
abs_text = []
for record in soup.findAll('rec'):
	if record.select('ab'):
		abTag = record.select('ab')
		for ab in abTag:
			absText = ab.get_text()
			abs_text.append(absText)
	else:
		abs_text.append("")
	for pubdate in record.select('dt'):
		pubyear = pubdate["year"]
		year.append(pubyear)

# create tuple merging the abstract and year data
tuples = [(key, value)
          for i, (key, value) in enumerate(zip(year, abs_text))]

# covert tuple to dataframe and export to csv
# note: after saving, open abstracts-with-date.csv and sort by year. the data needs
# to be in chronological in order to extract the 4-year sub-corpora.
# see articles-by-year.txt for which rows to extract for each 4-year increment.
# i removed the year and id columns and saved the remaining data (just the abstracts)
# as abstracts-cleaned.csv, which is used as the input for preprocessing
df = pd.DataFrame(tuples, columns=['year', 'abstract'])
df.to_csv("abstracts-with-date.csv")