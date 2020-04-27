#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:46:10 2020

@author: yrgg
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from opencage.geocoder import OpenCageGeocode
from opencage.geocoder import InvalidInputError, RateLimitExceededError, UnknownError

import dotenv
import os

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

import time

class SchoolCrawler:
    """Take list of cities urls and create pandas DataFrame.
        Can save as CSV or store in Database
    """
    def __init__(self, cities=[]):
        """Create instance
        
        Args:
            urls (List): list of urls to scrape
        """
        self.search_url = 'https://www.greatschools.org/search/search.page'
        self.base_url = 'https://www.greatschools.org'
        self.cities = cities
        self.cities_soup = self.search_all_cities()
        self.schools = self.parse_all_cities()
        self.df = pd.DataFrame(self.schools)

    def to_csv(self, path='./data/raw/school_ratings.csv'):
        self.df.to_csv(path)

    def parse_soup(self, school_soup):
        """Turn BS4 into dictionary

        Args: 
            school_soup (BS4): tree to parse
        Returns:
            parsed_school (dict): results
        """
        parsed_school = { 
            'name': '',
            'rating': 0,
            'street': '',
            'city': '',
            'zip': 0,
            'lat': 0,
            'long': 0,
            }
        parsed_school['name'] = school_soup.find('a', class_='name').get_text()
        parsed_school['rating'] = school_soup.find('div', class_='circle-rating--small').get_text().split('/')[0] if school_soup.find('div', class_='circle-rating--small') else 0
        if int(parsed_school['rating']) < 9:
            return
        address = school_soup.find('div', class_='address').get_text()
        parsed_school['lat'], parsed_school['long'] = self.lat_long(address)
        address = address.split(',')
        parsed_school['street'] = address[0]
        parsed_school['city'] = address[1]
        parsed_school['state'] = address[2]
        parsed_school['zip'] = address[3]

        return parsed_school
    
    def parse_all_cities(self):
        results = []
        for soup in self.cities_soup:
            for li in soup:
                result = self.parse_soup(li)
                if result:
                    results.append(result)
                    print(result)
        return results
    
    def get_schools_html(self, city):
        """Get first 25 'li' elements by city
        """
        city = city.replace(' ', '%20')
        search_string = f'{self.search_url}?q={city}&sort=rating'
        try:
            with webdriver.Safari() as browser:
                browser.get(search_string)
                WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "school-list")))

                search_results_html = BeautifulSoup(browser.page_source, features="lxml")
        
            schools = search_results_html.find_all('li', class_='unsaved')
            return schools
        except:
            print(f'error fetching: {search_string}')

    def search_all_cities(self):
        results = []
        for city in self.cities:
            soup = self.get_schools_html(city)
            results.append(soup)
        return results

    def lat_long(self, address):
        key = os.getenv("GEO_CODE")
        geocoder = OpenCageGeocode(key)

        try:
            results = geocoder.geocode(address)

            lat = results[0]['geometry']['lat']
            long = results[0]['geometry']['lng']
            time.sleep(1)

            return (lat, long)
        except RateLimitExceededError as ex:
            print(ex)
 
cities = ['Seattle', 'Kenmore', 'Sammamish', 'Federal Way', 'Maple Valley',
       'Bellevue', 'Duvall', 'Auburn', 'Kent', 'Redmond', 'Issaquah',
       'Renton', 'Kirkland', 'Mercer Island', 'Snoqualmie', 'Enumclaw',
       'Bothell', 'Fall City', 'North Bend', 'Vashon', 'Woodinville',
       'Carnation', 'Black Diamond', 'Medina']

sc = SchoolCrawler(cities)
print(len(sc.schools))
sc.to_csv()