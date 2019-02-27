# coding: utf-8
# **Thesis title:** Who’s afraid of the big bad terrorist?
# 
# **Subtitle:** Threat perception from Islamist and right-wing terrorism and the consequences in the United Kingdom and Germany.
# 
# **Submission Date:** April 2018

# # Preparation

### IMPORTS ###
import urllib.request
import urllib.error
import json
import datetime as dt
import csv
import time
import re
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import numpy as np
import sklearn as skl
import sklearn.linear_model as lm
reg = lm.LinearRegression()
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


### SET FOLDERS FOR WORKING DIRECTORY ###
folder_Webscraping      = 'replace with path'#### ANONYMIZED ####
folder_scraped_statuses = 'replace with path'#### ANONYMIZED ####
os.chdir(folder_Webscraping) # set wd to folder 'Webscraping'


### FACEBOOK ACCESS ###
app_id = "replace with ID" #### ANONYMIZED ####
app_secret = "replace with secret" #### ANONYMIZED ####
access_token = app_id + "|" + app_secret

### HELPER FUNCTIONS ###
def request_until_succeed(url):
    """ helper function to catch HTTP error 500"""
    req = urllib.request.Request(url)
    success = False
    
    while success is False:
        try: 
            response = urllib.request.urlopen(req)
            if response.getcode() == 200:
                success = True
        except Exception as e:
            print(e)
            time.sleep(5)
            print("Error for URL")
    return response.read()

def testFacebookPageData(page_name, access_token=access_token):
    """ get page's numeric information """  
    # construct the URL string
    base = "https://graph.facebook.com/v2.4"
    node = "/" + page_name
    parameters = "/?access_token=%s" % access_token
    url = base + node + parameters
    # retrieve data    
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode()) 
    #print(data)
    return data


# In[3]:


## [IDs OF] PAGES OF INTEREST ##
pages = [ # UK newspapers (broadsheet)
    'financialtimes', 'dailytelegraph', 'theguardian', 
    'TELEGRAPH.CO.UK', 'timesandsundaytimes',
    'northwesteveningmail',
    # UK newspapers (tabloids)
    'dailymail', 'DailyExpress', 'SundayExpres', 'thesun',
    'dailymirror', 'MirrorPolitics', 'thesundaypeople',
    'thedailystar', 'morningstaronline',
    'eveningstandard', 'MetroUK', 'cityam',
    # UK TV broadcasting
    'bbcnews', 'itv',
    'itvnews', 'uktvnow', 'Channel4', 
    'Channel4News', 'skynews',
    #25
    # Germany Newspapers (broadsheet and tabloid)
    'bild', 'faz', 'FrankfurterRundschau', 
    'handelsblatt',
    'jungefreiheit',
    'junge.welt', 'ihre.sz', 'spiegelonline', 
    'taz.kommune', 'welt',
    #'zeitonline'
    # Germany regional newspapers (> 200.000 circulation)
    'abendblatt',  'AugsburgerAllgemeine', 'freiepresse',
    'HNA', 'ksta.fb', 'NeueWestfaelische', 'rheinpfalz',
    'rponline', 'szonline', 'suedwestpresse', 'waz',

    # public broadcast
    'ARD', 'ZDF', 'ZDFheute',  'radiobremen', 'WDR',
    'monitor.wdr', 'hessischerrundfunk', 'SRonline.de',
    'SWRAktuell', 'bayerischer.rundfunk', 'fernsehen.rbb',
    'rbb24.de', 'NDR.de',
    # private broadcast
    'sat1tv', 'meinRTL', 'ProSieben', 'kabeleins'
    # 39
    ] #print(pages)

### BUILD DICTS FOR NAME OF PAGE AND NUMERIC ID ###
pages_ids_dict = {} # empty dictionary to store results
for page in pages:
    pages_ids_dict[testFacebookPageData(page, access_token)['name']] 
                    = testFacebookPageData(page, access_token)['id']

pages_ids_dict_backup = pages_ids_dict
#build inverse dictionary
inv_page_ids_dict = {v: k for k, v in pages_ids_dict.items()} 


# # Functions




def UNIX_ts_from_iso_8601_with_tz_offset(iso_8601):
    """ Convert ISO 8601 with a timezone offset to unix timestamp """
    # input format: ISO 8601 compliant, preferably 'YYYY-MM-DDTHH:MM:SS+XXXX'
    utc_at_epoch = dt.datetime(1970, 1, 1)
    if 'T' in iso_8601: iso_8601_dt = dt.datetime.strptime(iso_8601[:-5], '%Y-%m-%dT%H:%M:%S')
    else: iso_8601_dt = dt.datetime.strptime(iso_8601, '%Y-%m-%d')
    epoch_without_tz_offset = (iso_8601_dt - utc_at_epoch).total_seconds()
    if '+' in iso_8601 or iso_8601[-5] == '-': tz_offset = 60 * (60 * int(iso_8601[-4:-2]) + int(iso_8601[-2:]))
    else: tz_offset = 0
    if iso_8601[-5] == '-': tz_offset = -tz_offset
    return int(epoch_without_tz_offset - tz_offset)
#---------------------------------------------------------------------------------#
def queryFB_def_time (page_id, from_inp_perpcat_fatal_inj, duration_days=7, access_token=access_token, posts_lim=100, comm_lim=100, comments=False, get_data=True):
    # former name: getFacebookPageFeedData
    """retrieves (IF SELECTED) the statuses, (COMMENTS IF SELECTED,) reaction counts and share counts of one page for a specified time range (OTHERWISE DISPLAYS URL TO DO RETRIEVE THIS DATA)"""
    # don't use 'print(queryFB(...)')
    # parameter explanation: page_id       = the page's numeric ID
    #                        from_inp     = format "[YYYY]-[MM]-[DD]"T"[HH]:[MM]:[SS]+[XXXX]" time in UTC (XXXX = time zone difference to UTC)
    #                        duration_days = format "XXXX" (duration of time window in days)
    #                        access_token  = [app_id] + "|" + [app_secret]
    #                        posts_lim     = limit for number of posts
    #                        comm_lim      = limit for number of comments retrieved
    #                        comments      = True - the comments are included in the query; no comments are retrieved
    #                        get_data      = True - the query is executed; False - the URL is displayed as link
       
    # build URL
    base = "https://graph.facebook.com/v2.11"
    node = "/"
    since = "since=" + str(UNIX_ts_from_iso_8601_with_tz_offset(from_inp_perpcat_fatal_inj[:10]))
    plus = "&"
    until = "until=" + str(UNIX_ts_from_iso_8601_with_tz_offset(from_inp_perpcat_fatal_inj[:10])+86400*duration_days)
    fields = "fields=id,created_time,type,link,name,message,likes.limit(0).summary(total_count),reactions.type(LOVE).summary(total_count).limit(0).as(love),reactions.type(WOW).summary(total_count).limit(0).as(wow),reactions.type(HAHA).summary(total_count).limit(0).as(haha),reactions.type(SAD).summary(total_count).limit(0).as(sad),reactions.type(ANGRY).summary(total_count).limit(0).as(angry),shares.limit(0).summary(True)"
    comm_number = ",comments.limit(0).summary(True)"
    commfield = ",comments.limit(%s).summary(True)" % comm_lim
    token = "access_token=%s" % access_token
    url_p1 = base + node + page_id + node + "posts?" + since + plus + until + plus + fields
    if comments == True: url = url_p1 + commfield + plus + token
    else: url = url_p1 + comm_number + plus + token
    
    if get_data == False:    # not: return url
        print(from_inp_perpcat[-13:], ':', url)       # sample URL = https://graph.facebook.com/v2.11/228735667216/posts?since=2017-05-22T00:00:00+0000&until=2017-05-26T00:00:00&fields=id,created_time,updated_time,message,likes.summary(True)&access_token=107542820013006|ee6JFonBTXB51TmAGmZxCI1o7A8
                                        # THIS WORKS! The output - when entering that URL into the browser - is a JSON which is still a bit confusing but seems to work!
    else: # retrieve data
        data = json.loads(request_until_succeed(url))
        #return json.dumps(data, indent=4, sort_keys=True)
        return data
#---------------------------------------------------------------------------------#
def processFBStatus(status, from_inp_perpcat_fatal_inj, UTCplusX = 0): # status = JSON.data[X] // IB
    # former name: processFacebookPageFeedStatus
    ''' translates the JSON input from [queryFB] into a tuple with the values of interest '''
    # The status is a Python dictionary (JSON), so for top-level items, we can call the key. As some items may not exist, we must check their existence first
    
    status_id = 'missing_value' if 'id' not in status.keys() else status['id']
    status_message = 'missing_value' if 'message' not in status.keys() else status['message'].encode('utf-8')
    link_name = 'missing_value' if 'name' not in status.keys() else status['name'].encode('utf-8')
    status_type = 'missing_value' if 'type' not in status.keys() else status['type']
    status_link = 'missing_value' if 'link' not in status.keys() else status['link']
    perpcat = from_inp_perpcat_fatal_inj[:13]
    fat = from_inp_perpcat_fatal_inj[14:18]
    inj = from_inp_perpcat_fatal_inj[19:]
    
    # Time needs special care since a) it's in UTC and b) it's not easy to use in statistical programs.
    status_published = '' if 'created_time' not in status.keys() else dt.datetime.strptime(status['created_time'],'%Y-%m-%dT%H:%M:%S+0000')
    status_published = status_published + dt.timedelta(hours = + UTCplusX) # to local time from UTC
    status_published = status_published.strftime('%Y-%m-%d %H:%M:%S') # best time format for spreadsheet programs
    
    # Nested items require chaining dictionary keys.
    num_likes = "-99" if 'likes' not in status.keys() else status['likes']['summary']['total_count']
    num_love = "-99" if 'love' not in status.keys() else status['love']['summary']['total_count']
    num_wow = "-99" if 'wow' not in status.keys() else status['wow']['summary']['total_count']
    num_haha = "-99" if 'haha' not in status.keys() else status['haha']['summary']['total_count']
    num_sad = "-99" if 'sad' not in status.keys() else status['sad']['summary']['total_count']
    num_angry = "-99" if 'angry' not in status.keys() else status['angry']['summary']['total_count']
    num_comments = "-99" if 'comments' not in status.keys() else status['comments']['summary']['total_count']
    num_shares = "-99" if 'shares' not in status.keys() else status['shares']['count']
    
    # return a tuple of all processed data
    return (status_id, status_published, perpcat, status_message, link_name, status_type,
            status_link, num_likes, num_love, num_wow, num_haha, num_sad, num_angry,
            num_comments, num_shares, inj, fat)
#---------------------------------------------------------------------------------#
filenames_list = [] # this list is needed for the function as empty list under that name
def scrapeFB (page_id, from_inp_perpcat_fatal_inj, duration_days=7, access_token=access_token, limit=100, posts_lim=100, comm_lim=100, comments=False, get_data=True, UTCplusX=0):
    # former name: scrapeFacebookPageFeedStatus, based on: https://github.com/minimaxir/facebook-page-post-scraper/blob/master/examples/how_to_build_facebook_scraper.ipynb
    ''' takes the posts of one page (page_id) and scrapes them. As of now (2018-01-05), time limitation from the queryFB function does not work yet
        1. Query each page of Facebook Page Statuses (100 statuses per page) using getFacebookPageFeedData.
        2. Process all statuses on that page using processFBStatus and writing the output to a CSV file.
        3. Navigate to the next page, and repeat until no more statuses
        This function implements both the writing to CSV and page navigation.'''
    os.chdir(folder_scraped_statuses)
    with open('%s_%s_facebook_statuses.csv' % (from_inp_perpcat_fatal_inj[:13], page_id), 'w') as file:
        filenames_list.append('%s_%s_facebook_statuses.csv' % (from_inp_perpcat_fatal_inj[:13], page_id))
    #with open('%s_facebook_statuses.csv' % page_id, 'w') as file: # backup
        w = csv.writer(file)
        w.writerow(["status_id", "publication datetime", "perpetrator_categorized", "status_message", "link_name", "status_type",
            "status_link", "num_likes", "num_love", "num_wow", "num_haha", "num_sad", "num_angry",
            "num_comments", "num_shares", "fatalities", "injured"])
        
        has_next_page = True
        num_processed = 0   # keep a count on how many we've processed
        scrape_starttime = dt.datetime.now()
        
        print("Scraping Facebook Page: %s for %s and the following week. Scraping starts at: %s" % (inv_page_ids_dict[page_id], from_inp_perpcat_fatal_inj[:-13], scrape_starttime))
        
        statuses = queryFB_def_time(page_id, from_inp_perpcat_fatal_inj, duration_days, access_token, posts_lim, comm_lim, comments, get_data) # limit deleted ### queryFB_from_until would need to go here if used
        while has_next_page == True:
            for status in statuses['data']:
                w.writerow(processFBStatus(status, from_inp_perpcat_fatal_inj, UTCplusX=0))
                
                # output progress occasionally to make sure code is not stalling
                num_processed += 1
                if num_processed % 100 == 0:
                    print("%s Statuses Processed: %s" % (num_processed, dt.datetime.now()))
            # if there is no next page, we're done.
            if 'paging' in statuses.keys() and 'next' in statuses['paging'].keys():
                statuses = json.loads(request_until_succeed(statuses['paging']['next']))
            else:
                has_next_page = False
                print("Scraping Facebook Page %s done! %s Statuses Processed in %s \n\n" % (page_id, num_processed, dt.datetime.now() - scrape_starttime))
                global total_number # total_number need to be defined globally to def available in all functions
                total_number += num_processed
                os.chdir(folder_Webscraping)
#---------------------------------------------------------------------------------#
def scrapingFB_nested (ids_list, ls_date_perpabbr_fat_inj, duration=7, comments=False):
    ''' implements the scraping, nested with pages and time windows to scrape [duration in days]'''
    global total_number
    total_number = 0
    # input is a (str) list of the pages' IDs that need to be scraped and the beginnings of the time windows from which onwards one week will be scraped per instant
    starting_time_nested_scraping = dt.datetime.now()
    idx = int()
    for iterator in ls_date_perpabbr_fat_inj:   # execute the scraping for every time window in the list 'parseddates_ls'
        for idx in ids_list:              # execute the scraping for every page in the list 'pages_ids_ls'
            scrapeFB(idx, iterator, duration, access_token, 100, 100, comments) #, UTCplusX=0)
    print('\n\n\n________________________________________________LOOP DONE. SCRAPING COMPLETED!________________________________________________\nScraped %s statuses in %s' % (total_number, (dt.datetime.now() - starting_time_nested_scraping)))
#---------------------------------------------------------------------------------#


# # Data on attacks




# Get the dates of the attacks from the GTD output
# the data was retrieved from: http://www.start.umd.edu/gtd/search/ with the following parameters:
            # WHEN                2001-09-12 to 2016-12-31 [the last possible date included in the data base]
            #                         (the data is downloaded in two chunks, from 2001-09-12 to 2009-12-31 and 2010-01-01 to 2016-12-31)
            # REGION              Western Europe
            # COUNTRY             [All]
            # PERPETRATOR GROUP, WEAPON TYPE, ATTACK TYPE, TARGET TYPE  [All]
            # TERRORISM CRITERIA  Require Criteria I-III, exclude ambiguous cases, exclude unsuccessful attacks
            # CASUALTIES          Casualty type: both injured and fatalities; Number of casualties: any number
            # in addition, the the files can be found in the online appendix 

# load data
csv_attacks = "replace with path\#MPhil_thesis_Appendix\2001-09-12_2016-12-31_GTD-Export_Western_Europe.csv" #### ANONYMIZED ####
attacks_df = pd.read_csv(csv_attacks, delimiter=',', header=0)
attacks_df.insert(0, 'DATE_STR', attacks_df.DATE)
attacks_df.DATE = pd.to_datetime(attacks_df.DATE) # convert date to datetime
attacks_df.set_index('DATE', inplace=True) # set datetime index 'DATE'
del csv_attacks # clean up temporary data sets

### CLEAN AND CATEGORIZE PERPETRATORS ###
# define categorization
dict_categorization = dict_no_hash = {
    "Protestant extremists": "SEPARATIST (IRE, LOY)",
    "Basque Fatherland and Freedom (ETA)": "SEPARATIST (NON-UK)",
    "Revolutionary Solidarity": "LEFT",
    "Unknown": "UNKNOWN",
    "Ulster Volunteer Force (UVF)": "SEPARATIST (IRE, LOY)",
    "Irish National Liberation Army (INLA)": "SEPARATIST (IRE, REP)",
    "First of October Antifascist Resistance Group (GRAPO)": "LEFT",
    "Revolutionary People's Struggle (ELA)": "LEFT",
    "Terra Lliure": "IRRELEVANT TO RQ",
    "Ulster Freedom Fighters (UFF)": "SEPARATIST (IRE, LOY)",
    "November 17 Revolutionary Organization (N17RO)": "LEFT",
    "Corsican National Liberation Front (FLNC)": "SEPARATIST (NON-UK)",
    "Iparretarrak (IK)": "IRRELEVANT TO RQ",
    "Red Army Faction (RAF)": "LEFT",
    "Armed Falange": "IRRELEVANT TO RQ",
    "Serbian extremists": "IRRELEVANT TO RQ",
    "People's Rebellion": "RIGHT",
    "Serbian Nationalists": "IRRELEVANT TO RQ",
    "Iranian extremists": "IRRELEVANT TO RQ",
    "Serbian guerrillas": "IRRELEVANT TO RQ",
    "Resistenza": "LEFT",
    "Irish People's Liberation Organization (IPLO)": "SEPARATIST (IRE, REP)",
    "Neo-Nazi extremists": "RIGHT",
    "Action Group for the Destruction of the Police State": "LEFT",
    "Iranian exiles": "UNKNOWN",
    "Movement for the Protection of Jerusalem": "RIGHT",
    "Kurdistan Workers' Party (PKK)": "SEPARATIST (NON-UK)",
    "Mafia": "MAFIA",
    "Anti-Iran Government Exiles": "UNKNOWN",
    "Latvian Republic Volunteer Troops": "IRRELEVANT TO RQ",
    "Albanian exiles": "IRRELEVANT TO RQ",
    "French National": "IRRELEVANT TO RQ",
    "White extremists": "RIGHT",
    "Hungarian Skin Head Group": "RIGHT",
    "Iranians": "IRRELEVANT TO RQ",
    "Polish Skinheads": "RIGHT",
    "Red Commandos": "SEPARATIST (IRE, LOY)",
    "Jewish Extremists": "RELIGIOUS (JEWISH)",
    "Greek Anarchists' Union": "LEFT",
    "NaN": "UNCATEGORIZED",
    "Left-Wing Militants": "LEFT",
    "Youths": "UNKNOWN",
    "Animal Rights extremists": "ENVIRONMENTALIST",
    "Right-wing extremists": "RIGHT",
    "Right-Wing Youths": "RIGHT",
    "Bavarian Liberation Army": "IRRELEVANT TO RQ",
    "Belarusian Liberation Army": "IRRELEVANT TO RQ",
    "Israeli extremists": "IRRELEVANT TO RQ",
    "Armed Islamic Group (GIA)": "ISLAMIST",
    "Algerian Moslem Fundamentalists": "ISLAMIST",
    "Al-Gama'at al-Islamiyya (IG)": "ISLAMIST",
    "International Justice Group (Gama'a al-Adela al-Alamiya)": "ISLAMIST",
    "Corsican Separatists": "SEPARATIST (NON-UK)",
    "Turkish Revenge Brigade": "IRRELEVANT TO RQ",
    "Fighting Guerrilla Formation": "IRRELEVANT TO RQ",
    "Serbs": "SEPARATIST (NON-UK)",
    "Loyalist Volunteer Forces (LVF)": "SEPARATIST (IRE, LOY)",
    "Continuity Irish Republican Army (CIRA)": "SEPARATIST (IRE, REP)",
    "Revolutionary Nuclei": "IRRELEVANT TO RQ",
    "Orange Volunteers (OV)": "IRRELEVANT TO RQ",
    "Red Hand Defenders (RHD)": "SEPARATIST (IRE, LOY)",
    "Combat 18/ White Wolves (UK)": "RIGHT",
    "Loyalists": "SEPARATIST (IRE, LOY)",
    "Red Brigades Fighting Communist Party (BR-PCC)": "LEFT",
    "Hells Angels/ Nationalsocialistisk Front (NSF)": "RIGHT",
    "Breton Liberation Front (FLB)": "SEPARATIST (NON-UK)",
    "Irish Republican Extremists": "SEPARATIST (IRE, REP)",
    "Former Soldiers/Police": "IRRELEVANT TO RQ",
    "National Socialist Underground": "RIGHT",
    "Real Irish Republican Army (RIRA)": "SEPARATIST (IRE, REP)",
    "Anti-Semitic extremists": "RIGHT",
    "Revolutionary Perspective": "LEFT",
    "Animal Liberation Front (ALF)": "ENVIRONMENTALIST",
    "Armata Corsa": "IRRELEVANT TO RQ",
    "Haika": "IRRELEVANT TO RQ",
    "Revolutionary Proletarian Initiative Nuclei (NIPR)": "LEFT",
    "Anti-Imperialist Territorial Nuclei (NTA)": "LEFT",
    "Revolutionary Violence Units": "SEPARATIST (NON-UK)",
    "Anarchist Liberation Brigade": "LEFT",
    "Group of Carlo Giuliani": "IRRELEVANT TO RQ",
    "Anarchist Squad": "LEFT",
    "Neo-Fascists": "RIGHT",
    "Catholic Reaction Force": "IRRELEVANT TO RQ",
    "Association Totalement Anti-Guerre (ATAG)": "IRRELEVANT TO RQ",
    "Red Hand Defenders (RHD)/ Ulster Freedom Fighters (UFF)": "IRRELEVANT TO RQ",
    "Red Brigades Fighting Communist Union (BR-UCC)": "LEFT",
    "Popular Resistance (Laiki Antistasi)": "RIGHT",
    "Sardinian Autonomy Movement": "IRRELEVANT TO RQ",
    "New Revolutionary Popular Struggle (NELA)": "LEFT",
    "Rabid Brothers of Giuliani": "IRRELEVANT TO RQ",
    "Democratic Iraqi Opposition of Germany": "IRRELEVANT TO RQ",
    "CCCCC": "IRRELEVANT TO RQ",
    "Resistenza Corsa": "IRRELEVANT TO RQ",
    "Supporters of Johnny Adair": "IRRELEVANT TO RQ",
    "Proletarian Nuclei for Communism": "LEFT",
    "Anti IRQ War": "UNKNOWN",
    "Revolutionary Struggle": "LEFT",
    "Loyalist Action Force": "SEPARATIST (IRE, LOY)",
    "Informal Anarchist Federation": "LEFT",
    "Anarchists": "LEFT",
    "Abu Hafs al-Masri Brigades": "ISLAMIST",
    "Moroccan extremists": "IRRELEVANT TO RQ",
    "Resistance Cell": "LEFT",
    "Hofstad Network": "ISLAMIST",
    "Global Intifada": "ISLAMIST",
    "Secret Organization of al-Qaida in Europe": "ISLAMIST",
    "Al-Qaida Organization for Jihad in Sweden": "ISLAMIST",
    "Anti-State Justice": "LEFT",
    "Revolutionary Action of Liberation": "LEFT",
    "Solidarity with imprisoned members of Action Directe (AD)": "IRRELEVANT TO RQ",
    "Athens and Thessaloniki Arsonist Nuclei": "IRRELEVANT TO RQ",
    "Al-Qaida in Iraq": "ISLAMIST",
    "Dissident Republicans": "SEPARATIST (IRE, REP)",
    "Muslim extremists": "ISLAMIST",
    "Real Irish Republican Army (RIRA)/ Irish National Liberation Army (INLA)": "SEPARATIST (IRE, REP)",
    "Anti-Independence extremists": "IRRELEVANT TO RQ",
    "Conspiracy of Cells of Fire": "IRRELEVANT TO RQ",
    "Forbidden Blockade (Greece)": "UNKNOWN",
    "Anti-Democratic Struggle": "UNKNOWN",
    "LW": "UNKNOWN",
    "Hutu extremists": "IRRELEVANT TO RQ",
    "Oglaigh na hEireann": "IRRELEVANT TO RQ",
    "Gangs of Conscience": "IRRELEVANT TO RQ",
    "Popular Will (Greece)": "RIGHT",
    "Incendiary Committee of Solidarity for Detainees": "UNKNOWN",
    "Sect of Revolutionaries (Greece)": "LEFT",
    "Revolutionary Liberation Action (Epanastatiki Apelevtherotiki Drasi) - Greece": "IRRELEVANT TO RQ",
    "Nihilists Faction": "IRRELEVANT TO RQ",
    "Basque Separatists": "SEPARATIST (NON-UK)",
    "Alexandros Grigoropoulos Anarchist Attack Group": "LEFT",
    "Armed Revolutionary Action (ENEDRA)": "IRRELEVANT TO RQ",
    "Attack Teams for the Dissolution of the Nation (Greece)": "UNKNOWN",
    "Deniers of Holidays": "IRRELEVANT TO RQ",
    "Zero Tolerance": "IRRELEVANT TO RQ",
    "Illuminating Paths of Solidarity": "IRRELEVANT TO RQ",
    "Militant Forces Against Huntingdon": "IRRELEVANT TO RQ",
    "Council for the Destruction of Order": "IRRELEVANT TO RQ",
    "Hoodie Wearers": "IRRELEVANT TO RQ",
    "Anarchist Action (CA / United States)": "LEFT",
    "Black and Red Anarchist and Anti-Authoritarians Initiative (Greece)": "LEFT",
    "Paramilitary members": "IRRELEVANT TO RQ",
    "Revolutionary Continuity": "LEFT",
    "Sisters in Arms": "IRRELEVANT TO RQ",
    "Rebellious Group Lambros Foundas": "IRRELEVANT TO RQ",
    "Groups for Dissemination of Revolutionary Theory and Action": "LEFT",
    "Real Ulster Freedom Fighters (UFF) - Northern Ireland": "SEPARATIST (IRE, LOY)",
    "Iraqi extremists": "IRRELEVANT TO RQ",
    "Animal Rights Militia": "ENVIRONMENTALIST",
    "Provisional RSPCA": "IRRELEVANT TO RQ",
    "Hekla Reception Committee-Initiative for More Social Eruptions": "IRRELEVANT TO RQ",
    "Sunni Muslim extremists": "ISLAMIST",
    "Friends of Loukanikos": "LEFT",
    "Republican Action Against Drugs (RAAD)": "IRRELEVANT TO RQ",
    "Jihadi-inspired extremists": "ISLAMIST",
    "International Revolutionary Front": "RIGHT",
    "Hezbollah": "IRRELEVANT TO RQ",
    "Irish Republican Army (IRA)": "SEPARATIST (IRE, REP)",
    "The New Irish Republican Army": "SEPARATIST (IRE, REP)",
    "Militant Minority (Greece)": "IRRELEVANT TO RQ",
    "Militant Minority (Greece)/ Circle of Violators/Nucleus Lovers of Anomy": "IRRELEVANT TO RQ",
    "People's Fighter Group (Band of Popular Fighters)": "IRRELEVANT TO RQ",
    "Wild Freedom/ Instigators of Social Explosion": "IRRELEVANT TO RQ",
    "Wild Freedom": "IRRELEVANT TO RQ",
    "Angry Brigade": "IRRELEVANT TO RQ",
    "Real Ulster Freedom Fighters (UFF) - Northern Ireland/ Loyalist Volunteer Forces (LVF)": "SEPARATIST (IRE, LOY)",
    "Informal Anarchist Federation/ Int'l Revolutionary Front": "LEFT",
    "Overall Deniers of Joining the Existing": "IRRELEVANT TO RQ",
    "Epanastatiki Anatropi (Revolutionary Overthrow)": "IRRELEVANT TO RQ",
    "Anti-Clerical Pro-Sex Toys Group": "IRRELEVANT TO RQ",
    "Proletariat Self-defense Groups": "LEFT",
    "English Defense League (EDL)": "RIGHT",
    "Anti-Muslim extremists": "RIGHT",
    "Borderless Solidarity Cell (BSC)": "LEFT",
    "Detonators of Social Uprisings": "IRRELEVANT TO RQ",
    "Comite d'Action Viticole": "IRRELEVANT TO RQ",
    "Angry Foxes Cell/ All Coppers Are Bastards": "IRRELEVANT TO RQ",
    "Powers of the Revolutionary Arc": "IRRELEVANT TO RQ",
    "Mateo Morral Insurrectionist Commandos": "LEFT",
    "Resistencia Galega": "IRRELEVANT TO RQ",
    "Militant People's Revolutionary Forces": "LEFT",
    "Corsican Nationalists": "IRRELEVANT TO RQ",
    "Jewish Defense League (JDL)": "IRRELEVANT TO RQ",
    "Group of Popular Fighters": "IRRELEVANT TO RQ",
    "The Justice Department": "IRRELEVANT TO RQ",
    "National Liberation Front of Provence (FLNP)": "IRRELEVANT TO RQ",
    "Islamic State of Iraq and the Levant (ISIL)": "ISLAMIST",
    "Organization for Revolutionary Self Defense": "LEFT",
    "German Resistance Movement": "RIGHT",
    "Random Anarchists": "LEFT",
    "Left-wing extremists": "LEFT",
    "Proletarian Solidarity": "LEFT",
    "The Irish Volunteers": "SEPARATIST (IRE, REP)",
    "Yazidi extremists": "IRRELEVANT TO RQ",
    "Anarchist Anti-Capitalist Action Group": "LEFT",
    "Proletarian Assault Group": "LEFT",
    "Informal Anarchist Federation/ Earth Liberation Front": "LEFT",
    "Free Network South (Freies Netz Sued)": "RIGHT",
    "Al-Qaida in the Arabian Peninsula (AQAP)": "ISLAMIST",
    "Anarchist Commando Nestor Makhno Group": "LEFT",
    "Anti-Immigrant extremists": "RIGHT",
    "Patriotic Europeans against the Islamization of the West (PEGIDA)": "RIGHT",
    "Nihilistic Patrol and Neighborhood Arsonists": "LEFT",
    "International Revolutionary Front/ Informal Anarchist Federation": "RIGHT",
    "Earth Liberation Front (ELF)": "ENVIRONMENTALIST",
    "Revolutionary Cells Network (SRN)": "IRRELEVANT TO RQ",
    "The Third Way (Der III. Weg)": "RIGHT",
    "Rubicon (Rouvikonas)": "IRRELEVANT TO RQ",
    "Freital Group": "RIGHT",
    "Informal Anarchist Federation/ International Revolutionary Front": "LEFT",
    "Anarchist Cell Acca (C.A.A.)": "LEFT",
    "Ramiro Ledesma Social Centre": "IRRELEVANT TO RQ",
    "Irish National Liberation Army (INLA) / New IRA": "SEPARATIST (IRE, REP)",
    "Les Casseurs": "IRRELEVANT TO RQ",
    "Unrepentant Anarchists": "LEFT",
    "Lone Wolves of Radical, Autonomous, Militant National Socialism": "RIGHT",
    "Alde Hemendik Movement": "IRRELEVANT TO RQ",
    "No Borders Group": "LEFT",
    "Bahoz": "SEPARATIST (NON-UK)",
    "Nordic Resistance Movement": "RIGHT",
    "Angry Foxes Cell": "LEFT"
    }

dict_abbreviation_cat = {
    'RIGHT' : '_RI', 'LEFT': '_LE', 'SEPARATIST (NON-UK)': '_SE', 
    'SEPARATIST (IRE, REP)': '_SR', 'SEPARATIST (IRE, LOY)': '_SL',
    'ISLAMIST': '_IS', 'UNKNOWN' : '_UN', 'ENVIRONMENTALIST': '_EN',
    'IRRELEVANT TO RQ': '_IR'
    }

PERP = pd.Series(attacks_df['PERPETRATOR 1'])
PERP_CATEGORIZED = PERP.map(dict_categorization)
if 'PERPETRATOR CATEGORIZED' not in attacks_df.columns: attacks_df.insert(2, 'PERPETRATOR CATEGORIZED', PERP_CATEGORIZED)
#attacks_df.head(5)

CAT = pd.Series(attacks_df['PERPETRATOR CATEGORIZED'])
CAT_2L = CAT.map(dict_abbreviation_cat)
if 'ABBREVIATED CATEGORIZATION' not in attacks_df.columns: attacks_df.insert(0, 'ABBREVIATED CATEGORIZATION', CAT_2L)

ls_categories = ['RIGHT', 'LEFT', 'SEPARATIST (NON-UK)', 
                 'SEPARATIST (IRE, REP)', 'SEPARATIST (IRE, LOY)',
                 'ISLAMIST', 'UNKNOWN', 'ENVIRONMENTALIST',
                 'IRRELEVANT TO RQ']
del dict_abbreviation_cat

### CLEAN DATAFRAME ###
attacks_df = attacks_df[attacks_df['PERPETRATOR 1'] != 'Unknown'] # drop line if perpetrator is unknown
attacks_df['STR_FATALITIES'] = attacks_df['FATALITIES'].copy() # create "FATALITIES" column as string
attacks_df['STR_INJURED'] = attacks_df['INJURED'].copy() # create "INJURED" column as string
attacks_df['FATALITIES'] = pd.to_numeric(attacks_df['FATALITIES'], errors='coerce')
attacks_df['INJURED'] = pd.to_numeric(attacks_df['INJURED'], errors='coerce')
# drop columns
attacks_df.drop(attacks_df.columns[14:25], axis=1, inplace=True) #drop columns that are not of interest
attacks_df.drop(attacks_df.columns[7:12], axis=1, inplace=True) #drop columns that are not of interest
attacks_df.drop(attacks_df.columns[6], axis=1, inplace=True) #drop perpetrator 1, too
#attacks_df.drop(attacks_df.columns[1], axis=1, inplace=True) #drop DATE_STR done later, after filtering
attacks_df['FAT & INJ'] = attacks_df['FATALITIES'] + attacks_df['INJURED']
# reformat strings
attacks_df['STR_FATALITIES'] = attacks_df['STR_FATALITIES'].apply('{:0>4}'.format) # fill up with leadin zeros to 4 numbers
attacks_df['STR_INJURED'] = attacks_df['STR_INJURED'].apply('{:0>4}'.format) # fill up with leadin zeros to 4 numbers


### FILTERING THE DATA SET ###
attacks_of_interest = attacks_df[           # only attacks of RIGHT/ LEFT/ ISLAMIST/ SEPPARATIST (UK) AND WITH AT LEAST ONE CASUALTY
        (
            (attacks_df['PERPETRATOR CATEGORIZED'] == 'blank') # placeholder: impossible condition for coding convenience
            | (attacks_df['PERPETRATOR CATEGORIZED'] == 'RIGHT')
            | (attacks_df['PERPETRATOR CATEGORIZED'] == 'LEFT')
            | (attacks_df['PERPETRATOR CATEGORIZED'] == 'SEPARATIST (IRE, REP)')
            | (attacks_df['PERPETRATOR CATEGORIZED'] == 'SEPARATIST (IRE, LOY)')
            | (attacks_df['PERPETRATOR CATEGORIZED'] == 'ISLAMIST')
            #| (attacks_df['PERPETRATOR CATEGORIZED'] == 'SEPARATIST (NON-UK)')
            #| (attacks_df['PERPETRATOR CATEGORIZED'] == 'ENVIRONMENTALIST')
            #| (attacks_df['PERPETRATOR CATEGORIZED'] == 'UNKNOWN') 
            #| (attacks_df['PERPETRATOR CATEGORIZED'] == 'IRRELEVANT TO RQ')
        )
        &
        (
            (attacks_df['FAT & INJ'] >= 100000) # placeholder: impossible condition for coding convenience
            #| (attacks_df['FAT & INJ'] >= 2)
            #| (attacks_df['INJURED'] >10)
            | (attacks_df['FATALITIES'] > 0)
            #& (attacks_df['FAT & INJ'] > 1)
        )
        ]
del attacks_df, CAT, CAT_2L # clean up

### GENERATE DATE AND ABBREVIATION FOR PERPETRATOR ###
# EXTRACT DATES, ABBREVATED PERPETRATOR, FATALITIES, AND INJURED #
attacks_of_interest.sort_index(inplace=True)
dates_ls = attacks_of_interest['DATE_STR'].tolist()
perpcat_abbr_list = attacks_of_interest['ABBREVIATED CATEGORIZATION']
inj_ls = attacks_of_interest['STR_INJURED']
fat_ls = attacks_of_interest['STR_FATALITIES']

parseddates_ls = []
for i in dates_ls: # parse dates
    parseddate = dt.datetime.strftime(dt.datetime.strptime(i[0:10],'%d/%m/%Y'),'%Y-%m-%d')
    parseddates_ls.append(str(parseddate) + i[10:])
    del parseddate
assert len(parseddates_ls) == len(perpcat_abbr_list) & len(perpcat_abbr_list) == len(inj_ls) & len(inj_ls) == len(fat_ls)
date_perp_abbr_ls = [x+y for x,y in zip(parseddates_ls,perpcat_abbr_list)] # redundant; only as backup
ls_date_perpabbr_fat_inj = [w + x + '_' + y + '_' + z for w,x,y,z in zip(parseddates_ls,perpcat_abbr_list, fat_ls, inj_ls)]
assert len(date_perp_abbr_ls) == len(parseddates_ls) == len(perpcat_abbr_list) # redundant; only as backup
assert len(ls_date_perpabbr_fat_inj) == len(parseddates_ls)

print('There are ' + str(len(date_perp_abbr_ls)) + ' attacks of interest, for which the week following the attack will be scraped.') # print(parseddates_ls[0:])

pages_ids_ls = []
for k in pages_ids_dict:
    list_item = pages_ids_dict[k]
    pages_ids_ls.append(list_item)
assert len(pages_ids_dict) == len(pages_ids_ls) # ensure the list contains as many items as the dictionary contains key/ value pairs


# # Scraping Facebook




# analysis of the posts scrapted for the entire time period shows that the first posts on terrorist attacks were in 2010; therefore, the searched time frame is restriced to this time frame to save bandwidth

### SCRAPE FB WITH THE PAGES AND DATES SPECIFIED ###
scrapingFB_nested(pages_ids_ls, ls_date_perpabbr_fat_inj[28:], duration=7, comments=True) #temp_date_perp_abbr_ls) # this command executes the scraping and takes about 5min to execute


# # Importing and cleaning




### LOAD DATA INTO PANDAS ###
posts = pd.DataFrame() # empty pandas data frame
for filename in filenames_list_man: # imports the files one by one into one data frame
    os.chdir(folder_scraped_statuses)
    df = pd.read_csv(filename, delimiter=',', converters={'link_name': str, 'status_message': str}, na_values=['-99'],
                     encoding = "ISO-8859-1") # other values (e.g. 'missing_value', as assigned) commented out to deal with missing values in another way
    pageid = filename.split('_')[2] # take 'filename' from 'filename_list', take what's before the '_'
    df.insert(0, 'page_id', str(pageid)) # append this as the new first column called "page_id" to the data frame
    #print(df.head())#
    posts = posts.append(df)
    os.chdir(folder_Webscraping)
    
### CLEAN AND ORDER DATA ###
print('The initial dimensions of the DataFrame are:    ' + str(posts.shape) + ".") #The initial dimensions of the DataFrame are:    (5847, 16)
posts['status_id'] = posts['status_id'].str.replace('\d+_', '').to_frame() # delete any number of digits before the '_' in the column post['status_ids']
posts['attack_date'], posts['attacker_categorized'] = posts['perpetrator_categorized'].str.split('_', 1).str # make attack date_perpetrator into two columns

# include page name in addition to page ID to ensure better readability
if 'page_name' in posts.columns: del posts['page_name'] # if code is run multiple times
posts.insert(0, 'page_name', posts['page_id']) # append this as the new first column called "page_id"
posts["page_name"].replace(inv_page_ids_dict, inplace=True) # include name of page instead of numeric ID
#print(posts.columns)
posts_reordered = posts[['page_name', 'page_id', 'status_id', # reorder columns
                         'attack_date', 'attacker_categorized', 'publication datetime',
                         'status_type', 'status_message', 'link_name', 'status_link',
                         'num_likes', 'num_love', 'num_wow', 'num_haha', 'num_sad', 'num_angry', 
                         'num_comments', 'num_shares', 'perpetrator_categorized', 
                         'fatalities', 'injured']]
posts_reordered = posts_reordered.rename(columns={'publication datetime': 'publication_datetime_str'}) #rename column
assert posts.shape == posts_reordered.shape
del posts_reordered['perpetrator_categorized'], posts #clean up

# DELETE DUPLICATES AND MISSING VALUES #
posts_reordered.drop_duplicates(keep='first', inplace=True) #if only selecting on the status ID: 'status_id', keep='first', inplace=True) [not advisable since one post might belong to two instances]

#del miss_values
posts_reordered.dropna(thresh=14, inplace=True) # it seems i can only drop all... need to fix this    
posts_reordered['status_message'].dropna(inplace=True) # inplace needs to be True if not assigning anew, otherwise needs False
posts_reordered['link_name'].dropna(inplace=True) # inplace needs to be True if not assigning anew, otherwise needs False

## INDEX ##
dict_remap = {'RI': 'RIGHT', 'LE': 'LEFT', 'SL': 'SEP-LOY', 'SR': 'SEP-REP', 'IS': 'ISLAMIST'}
posts_reordered['attacker_categorized'] = posts_reordered['attacker_categorized'].map(dict_remap)
del dict_remap # clean up
posts_reordered['attack_date'] = pd.to_datetime(posts_reordered['attack_date'], errors='ignore') # convert column to datetime
posts_reordered['status_id'] = pd.to_numeric(posts_reordered['status_id'], errors='ignore') # convert column to numeric
posts_reordered.sort_values(by=['attack_date', 'status_id'], ascending=True, inplace=True, na_position='first')

posts_reordered.reset_index(inplace=True)
if 'index' in posts_reordered.columns: del posts_reordered['index']
if 'level_0' in posts_reordered.columns: del posts_reordered['level_0'] # this indexing was chosen to facilitate the string search below that runs into difficulties (lexsort) with multilevel indexes

### INSPECTING AND CLEANING DATA ###
# convert data types
cols_to_convert = ['page_id', 'num_likes', 'num_love', 'num_wow', 'num_haha', 
                   'num_sad', 'num_angry', 'num_comments', 'num_shares', "injured", "fatalities"]
posts_reordered[cols_to_convert] = posts_reordered[cols_to_convert].apply(pd.to_numeric, errors='ignore', axis=1) # convert to numeric
posts_reordered['publication_datetime_dt'] = pd.to_datetime(posts_reordered['publication_datetime_str'], errors='ignore') # convert column to datetime
del cols_to_convert # clean up

print("After cleaning, the DataFrame's dimensions are: " + str(posts_reordered.shape) + '. This should be [initial number of columns] + 3 (split of attack_date and attacker_categorized, added page_name and datetime_publication_dt)')

#define keywords for which to search the posts
keywords = ['terror',
            'attack', 'Angriff', 'Anschlag', 'Anschläge',
            'shooting','Schießerei',
            'knife','Messer',
            'Islamist', 'jihad', 'Dschihad'
            'Islam',
            'Extremist'
           ]

negative_keywords = ['cyber', # topics
                     'boko', # groups
                     'Chad', 'Niger', 'Cameroon', 'Tunis', 'Iraq', 'Afghanistan',
                     'Tschad', 'Kamerun', 'Tunesien', 'Irak',
                     "Weltkrieg", "World War"
                    ]
posts_filtered = posts_reordered[((posts_reordered['status_message'].str.contains('|'.join(keywords), case=False))
                                  | (posts_reordered['link_name'].str.contains('|'.join(keywords))))
                               & (~posts_reordered['status_message'].str.contains('|'.join(negative_keywords), case=False))
                               & (~posts_reordered['link_name'].str.contains('|'.join(negative_keywords), case=False))
                                ]
print("Dimensions of filtered posts: " + str(posts_filtered.shape)) # (1316, 21)
posts_filtered.reset_index(inplace=True) # reset index
posts_filtered.fillna(0, inplace=True) # fill missing values with 0

posts_filtered.sort_values(by=['attack_date', 'status_id'], ascending=True, inplace=True, na_position='first')
posts_filtered.drop([0, 1, 2], inplace=True) # the first three posts were dated back by the site owners (published post-2010, dated to 2005)
posts_filtered.reset_index(inplace=True)





#insert new columns for dummies
cols = [ # columns to create
    "islamist_attack", "left_attack", "right_attack", "sep_loy_attack", "sep_rep_attack", "separatist", #attacks
    "photo", "video", "link", "type_ord",#status types
    "YYYYMM",
    "terror_mentioned"]
dict_perps = {'ISLAMIST': "islamist_attack", 'LEFT': "left_attack", 'RIGHT': "right_attack", 'SEP-LOY': "sep_loy_attack", 'SEP-REP': "sep_rep_attack"}
dict_islamist = {"JeSuisCharlie": "islamist_attack", "Hebdo": "islamist_attack", "ParisAttacks": "islamist_attack",
                 "Anschlag in Paris": "islamist_attack", "Paris Attacks": "islamist_attack", "Je suis charlie": "islamist_attack", "Paris terror attack": "islamist_attack", "shootings in Paris": "islamist_attack",
                 "attack in Paris": "islamist_attack",  "Paris beginnt der Schweigemarsch": "islamist_attack", "Anschlag von Paris": "islamist_attack", "attacks in Paris": "islamist_attack",
                 
                 "Islam": "islamist_attack", "jihad": "islamist_attack"}
dict_status = {"photo": "photo", "link": "link", "video": "video"}
for col in cols:
    if col in posts_filtered.columns: del posts_filtered[col]
    posts_filtered[col] = pd.Series(0, index=posts_filtered.index)

# assign attacker categories and status types dummies
for k in dict_perps:
    posts_filtered.loc[posts_filtered['attacker_categorized'] == k, dict_perps[k]] = 1

for k in dict_islamist:
    posts_filtered.loc[posts_filtered['status_message'].str.contains(k, case=False), 'islamist_attack'] = 1 # check that posts refering to charlie hebdo are labelled as "islamist"
    
for k in dict_status:
    posts_filtered.loc[posts_filtered['status_type'] == k, dict_status[k]] =  1
    
posts_filtered.loc[posts_filtered['attacker_categorized'].str.contains('SEP', case=False), "separatist"] = 1

dict_status_type = {"link": 0,  "video": 1, "photo": 2}
for key in dict_status_type:
    posts_filtered.loc[posts_filtered['status_type'] == key, 'type_ord'] = dict_status_type[key]

# check that islamist attacks are labelled only as such
ls_attack_dummies_without_islamist = ['sep_loy_attack', 'sep_rep_attack', "right_attack", "left_attack"]
for i in ls_attack_dummies_without_islamist:
    posts_filtered.loc[posts_filtered["islamist_attack"]==1, i] = 0
    
posts_filtered["YYYYMM"] = posts_filtered['publication_datetime_str'].str[0:4] + posts_filtered['publication_datetime_str'].str[5:7] # assign YYYYMM to column
posts_filtered["YYYYMM"] = pd.to_numeric(posts_filtered["YYYYMM"])

# assign "terrorism mentioned" dummy
posts_filtered.loc[posts_filtered['status_message'].str.contains("terror", case=False), "terror_mentioned"] = 1 

# combine 'reactions'
if 'reactions_comb' in posts_filtered.columns: del posts_filtered['reactions_comb'] #if code is run multiple times
posts_filtered = posts_filtered.assign(reactions_comb = #create column with combined reactions
                     pd.Series(posts_filtered['num_likes']
                             + posts_filtered['num_love']
                             + posts_filtered['num_wow']
                             + posts_filtered['num_haha']
                             + posts_filtered['num_sad']
                             + posts_filtered['num_angry']
                             + posts_filtered['num_comments'] # not sure about this
                             + posts_filtered['num_shares'] # not sure about this
                            ))
# drop previously combined columns
posts_filtered_dropped = posts_filtered.drop(
    ['num_likes', 'num_love', 'num_wow', 'num_haha', 'num_sad', 'num_angry',
     'num_comments', 'num_shares', 
     'page_id', 'status_id', 'attacker_categorized', 'status_type', 'status_message', 'link_name', 'status_link'], axis=1
    )

del cols, dict_perps, dict_status, ls_attack_dummies_without_islamist, dict_islamist, dict_status_type


# # Transform and export data




posts_only = posts_filtered.drop(['level_0', 'index', 'page_name', 'page_id', 'status_id', 'publication_datetime_str', 
                                  'status_type', 'status_message', 'link_name', 'status_link',
                                  'num_likes', 'num_love', 'num_wow', 'num_haha', 'num_sad', 'num_angry',
                                  'num_comments', 'num_shares', 'publication_datetime_dt', 'photo', 'video', 'link',
                                  'reactions_comb'], axis=1)

posts_only["att_date_perpcat"] = posts_only["attack_date"].map(str) + "_" + posts_only["attacker_categorized"] # prepare column to of attack and perpetrator (in case multiple attacks by different groups happened on the same day)
posts_only['att_date_perpcat'] = posts_only['att_date_perpcat'].str.replace(' 00:00:00', '') # remove " 00:00:00"
posts_only.sort_values(by=['att_date_perpcat'], ascending=True, inplace=True, na_position='first') # sort

### VERSION 1 ### #prepare df with meta data on posts -- version 1: don't change!
I_post_count = posts_only.groupby('att_date_perpcat').first() # create result df and group by attack_date column; keep first occurence
I_post_count['count'] = posts_only['att_date_perpcat'].value_counts() # count values of attack_date in the column counts
I_post_count.reset_index(inplace=True) # reset index

### VERSION 2 ### #prepare df with meta data on posts -- version 2: don't change!
II_post_count = posts_only.groupby('att_date_perpcat').sum() # create result df and group by attack_date column; keep sum of occurences
II_post_count['count'] = posts_only['att_date_perpcat'].value_counts() # count values of attack_date in the column counts
II_post_count.reset_index(inplace=True) # reset index

### Add 1 column (terror_mentioned) from V2 to replace that column in V1
I_post_count['terror_mentioned'] = II_post_count['terror_mentioned']





# Export post analysis dataframe
#posts_filtered.to_csv("posts_export_v1.csv", na_rep="-99")
posts_filtered.to_csv("posts_export.csv", na_rep="-99")

# Export post count dataframe
#I_post_count.to_csv("post_count_export_v1.csv", na_rep='-99')
I_post_count.to_csv("post_count_export.csv", na_rep='-99')
#II_post_count.to_csv("post_count_export2.csv", na_rep='-99')

