import datetime

from modules.crawler.list.url_list import List
from modules.crawler.list.date_cut import date_cut_dict
from modules.crawler.etc.post_wash import post_wash
from modules.crawler.etc.img_size import img_size
#게시판 n페이지를 받으면, 그 페이지의 포스트 url 리스트 반환
def Parsing_list_url(URL, bs):
	List = []
	domain = Domain_check(URL['url'])
	if bs.find("div", {"class": "list"}) is None:
		return List
	posts_all = bs.find("div", {"class": "list"}).findAll("div", {"class": "list-item"})
	posts_closed = bs.find("div", {"class": "list"}).findAll("div", {"class": "list-item closed"})
	posts = list(set(posts_all) - set(posts_closed))
	for post in posts:	
		url = post.find("a")['href']
		url = domain + url
		List.append(url)
			
	return List



#포스트 url을 받으면, 그 포스트의 정보를 dictionary 형태로 반환
def Parsing_post_data(bs, post_url, URL):
	return_data = []
	post_data = {}
	domain = Domain_check(URL['url'])

	title = bs.find("div", {"id": "ModuleBoardView"}).find("div", {"class": "title"}).find("h5").get_text(" ", strip = True)
	author = bs.find("span", {"rel": "author"}).text.strip()
	date = bs.find("li", {"class": "date"}).find("time").text.strip()
	date = str(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
	post = bs.find("div", {"class": "content"}).get_text(" ", strip = True)
	post = post_wash(post)		#post 의 공백을 전부 제거하기 위함
	if bs.find("article", {"data-role": "post"}).find("img")['src'] is None:
		img = 1
	else:
		try:
			img = bs.find("article", {"data-role": "post"}).find("img")['src']
			if 1000 <= len(img):
				img = 1
			else:
				if img.startswith("http://") or img.startswith("https://"):		# img가 내부링크인지 외부 링크인지 판단.
					pass
				elif img.startswith("//"):
					img = "http:" + img
				else:
					img = domain + img
		except:
			img = 1
	if img != 1:
		if img_size(img):
			pass
		else:
			img = 1

	post_data['title'] = title.upper()
	post_data['author'] = author.upper()
	post_data['date'] = date
	post_data['post'] = post.lower()
	post_data['img'] = img
	post_data['url'] = post_url

	return_data.append(post_data)
	return_data.append(title)
	return_data.append(date)
	return return_data



#url을 받으면 Page를 변환시켜서, 변환된 url 반환
def Change_page(url, page):
	url_done = url + str(page)

	return url_done


#입력된 url의 도메인 url 반환 :::: sj10 에 한해서 /bbs/ 까지 뽑힘
def Domain_check(url):
	domain = url.split('/')[0] + '//' + url.split('/')[2]	#도메인 url 추출

	return domain