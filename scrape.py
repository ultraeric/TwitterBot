import os
from TwitterAPI import TwitterAPI
import json
import time
import configparser
import gzip
import shutil


#Set up constants

config = configparser.ConfigParser()
config.read('./config/api_auth.cf')
consumer_key = config['AUTH']['consumer_key']
consumer_secret = config['AUTH']['consumer_secret']
access_token_key = config['AUTH']['access_token_key']
access_token_secret = config['AUTH']['access_token_secret']

config.read('./config/conf.cf')
write_file_name = config['FILE LOCS']['dirty_dataset_dir'] + '/dirty' 
#Get auth tokens, etc., initialize Twitter API connections
api = TwitterAPI(consumer_key, consumer_secret, access_token_key, access_token_secret)

#Get from the specified endpoint
req = api.request('statuses/sample', {})

#Function to scrape, used to help restart scrapinging in case of error
time_start = time.time()
num_file = int(float(config['DATA INFO']['start_num']))
def scrape_stuff():
	counter = 0
	time_file = time.time()
	global num_file
	write_file = open(write_file_name + str(num_file) + '.txt', 'a+')
	time_elapsed = time.time()

	#Iterates over items given in request
	for item in req:
		if 'delete' in item or ('lang' in item and item['lang'] != 'en'):
			continue	
	
		#Log progress
		if counter % 100 == 0:
			print('Seconds since start/last file/last 100: ' + str(time.time() - time_start) + ' // ' + str(time.time() - time_file) + ' // ' + str(time.time() - time_elapsed))
			print('Current progress: ' + str(counter))
			time_elapsed = time.time()
		counter += 1
		write_file.write(json.dumps(item) + '\n')

		#Every 100,000 good tweets transfer them to a zip file
		if counter % 100000 == 0 and counter > 0:
			write_file.close()
			with open(write_file_name + str(num_file) + '.txt', 'rb') as f_in:
				with gzip.open(write_file_name + str(num_file) + '.txt.gz', 'wb') as f_out:
					shutil.copyfileobj(f_in, f_out)

			#Remove uncompressed
			os.remove(write_file_name + str(num_file) + '.txt')
			num_file += 1
			write_file = open(write_file_name + str(num_file) + '.txt', 'a+')
			time_file = time.time()

#Loop to ensure continuous scraping
while True:
	try:
		print('scraping')
		scrape_stuff()
	except Exception:
		req = api.request('statuses/sample', {})
		continue
