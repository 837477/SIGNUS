from bs4 import BeautifulSoup
from selenium import webdriver
import datetime
from modules.crawler.login import everytime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time

from modules.crawler.etc.driver_agent import chromedriver
from modules.crawler.list.url_list import List
from modules.crawler.list.date_cut import date_cut
from modules.crawler.etc.post_wash import post_wash
from modules.crawler.etc.img_size import img_size



#게시판 page_url 을 받으면, 그 페이지의 포스트 url 들을 반환
def Parsing_list_url(URL, page_url, driver):
	List = []

	#만약 driver이 켜져있으면 끄고, 없으면 그냥 진행
	try:
		driver.quit()
	except:
		pass

	driver = chromedriver()
	List.append(page_url)
	
	data = (driver, List)
	return data



#포스트 url을 받으면, 그 포스트의 정보를 dictionary 형태로 반환
def Parsing_post_data(driver, post_url, URL, recent_post):
	post_data_prepare = []
	domain = Domain_check(URL['url'])
	end_date = date_cut(URL['info'])
	now_num = 0

	driver.get(post_url)
	post_driver = chromedriver()
	#비교를 하기위해서 make
	last_posts = [0]
	while 1:
		if (now_num > 0) and (now_num % 100 == 0):
			print("post_driver를 재시작 합니다.")
			post_driver.close()
			post_driver = chromedriver()	# 포스트 페이지를 위한 드라이버
		
		driver.find_element_by_tag_name("body").send_keys(Keys.END)
		time.sleep(1)

		html = driver.page_source
		bs = BeautifulSoup(html, 'html.parser')

		posts = bs.find("div", {"class": 'list'}).findAll("div", {"class": "item"})
		#더이상 내릴 수 없으면 break
		if len(last_posts) == len(posts):
			break
		else:
			last_posts = posts

		for post in posts[now_num:]:
			try:
				post_data = {}
				url = post.find("a")['href']
				url = domain + url
				try:
					post_driver.get(url)
				except:
					if len(post_data_prepare) == 0:
						recent_post = None
					else:
						recent_post = post_data_prepare[0]['title']
					data = (post_data_prepare, recent_post)
					return data
				try:
					WebDriverWait(post_driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "img"))) #a.item을 발견하면 에이작스 로딩이 완료됬다는 가정
				except Exception as e:
					print(e)
				except:
					if len(post_data_prepare) == 0:
						recent_post = None
					else:
						recent_post = post_data_prepare[0]['title']
					data = (post_data_prepare, recent_post)
					return data				
				html_post = post_driver.page_source
				bs_post = BeautifulSoup(html_post, 'html.parser')
				title = bs_post.find("div", {"class": "content"}).find("h1").get_text(" ", strip = True)
				date =  bs_post.find("div", {"class": "section"}).find("p", {"class": "indent"}).text.strip()
				date = date.split("~")[1]
				date = date.split("(")[0]
				date = date[1:]
				#마감을 뜻한다.
				if date == "0년 0월 0일":
					break
				now_year = datetime.datetime.now().strftime("%Y")
				try:
					date = now_year + "년 " + date + " 00:00:00"
					date = str(datetime.datetime.strptime(date, "%Y년 %m월 %d일 %H:%M:%S"))
				except:
					date = date[6:].strip()
					date = str(datetime.datetime.strptime(date, "%Y년 %m월 %d일 %H:%M:%S"))
				phrase = bs_post.find("article", {'class': "description"}).get_text(" ", strip = True)
				phrase = post_wash(phrase)		#post 의 공백을 전부 제거하기 위함
				if bs_post.find("div", {"class": "poster"}) is None:
					img = 8
				else:
					try:
						img = bs_post.find("div", {"class": 'poster'}).find("img")['src']
						if 1000 <= len(img):
							img = 8
						else:
							if img.startswith("http://") or img.startswith("https://"):		# img가 내부링크인지 외부 링크인지 판단.
								pass
							elif img.startswith("//"):
								img = "http:" + img
							else:
								img = domain + img
					except:
						img = 8
				if img != 8:
					if img_size(img):
						pass
					else:
						img = 8

				post_data['title'] = title.upper()
				post_data['author'] = ''
				post_data['date'] = date
				post_data['post'] = phrase.lower()
				post_data['img'] = img
				post_data['url'] = url

				print(date, "::::", title)

				if (date < end_date) or (title.upper() == recent_post):
					break
				else:
					post_data_prepare.append(post_data)
			except:
				continue
		now_num = len(posts)
		print("now_num : ", now_num)
		if (date <= end_date) or (title.upper() == recent_post) or date == "0년 0월 0일":
			break
	if len(post_data_prepare) == 0:
		recent_post = None
	else:
		recent_post = post_data_prepare[0]['title']
	data = (post_data_prepare, recent_post)
	post_driver.close()
	return data



#url을 받으면 url 그대로 반환해준다. => Page number이 필요하지 않는 url
def Change_page(url, page):

	return url


#입력된 url의 도메인 url 반환
def Domain_check(url):
	domain = url.split('/')[0] + '//' + url.split('/')[2]	#도메인 url 추출

	return domain