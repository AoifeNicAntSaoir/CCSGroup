from bs4 import BeautifulSoup
import  requests

page = requests.get("https://www.rte.ie/news/markets/euroexchangerates/")
soup = BeautifulSoup(page.content, 'html.parser')
table = soup.find_all('table', class_='business-shares')
print(table[0])

