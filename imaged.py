
from selenium import webdriver as wd
from bs4 import BeautifulSoup as Bsoup
from urllib import request as req
import os
import time
import random
import datetime

class ImageDownloader :
    def __init__(self, animal_name, scroll_count, path, url="https://unsplash.com/s/photos/") :
        self.animal_name = animal_name
        self.scroll_count = scroll_count
        self.path = path
        self.url = url
        
        self.is_plurality = True if str(type(animal_name)) == "<class 'list'>" else False
        self.is_dir = os.path.isdir(path)

        self.wd_path = "C:/Users/muns3/Downloads/chromedriver_win32 (1)/chromedriver.exe"

    def __setDir(self, name) :
        dir_path = os.path.join(self.path, name)

        if not os.path.exists(dir_path) :
            os.makedirs(dir_path)

        return dir_path

    def __getImagesUrl(self, name) :
        url = self.url + name
        
        print(url)

        driver = wd.Chrome(self.wd_path)
        driver.get(url)

        scroll_value = self.scroll_count

        for i in range(20) :
            driver.execute_script("window.scrollTo(0, " + str(scroll_value) + ")")
            scroll_value += self.scroll_count
            time.sleep(1)

        soup = Bsoup(driver.page_source, "html.parser")
        images = soup.find_all("img", attrs={"class" : "_2UpQX"})

        urls_arr = []

        for image in images :
            if image.get("src") != None :
                url = image.get("src")
                urls_arr.append(url)

        print(urls_arr)

        return urls_arr

    def saveImages(self) :
        if not self.is_plurality :
            name_list = list([self.animal_name])

        else :
            name_list = self.animal_name

        for name in name_list :
            dir_path = self.__setDir(name)
            urls_arr = self.__getImagesUrl(name)

            for url in urls_arr :
                file_name = datetime.datetime.now().strftime("%y%m%d%H%M%S")
                file_name += str(random.randrange(1, 1000))
                file_name += str(random.randrange(1, 1000))
                
                jpg_name = file_name + ".jpg"
                png_name = file_name + ".png"

                read = req.urlopen(url).read()

                with open(os.path.join(dir_path, jpg_name), "wb") as save :
                    try :
                        save.write(read)              

                    except :
                        pass

                with open(os.path.join(dir_path, png_name), "wb") as save :
                    try :
                        save.write(read)              

                    except :
                        pass

path = "C:/Users/muns3/OneDrive/Desktop/python-project/learning-program/animal_analyzer/data_related"
Idr = ImageDownloader(["dog", "cat"], 500, path)
Idr.saveImages()