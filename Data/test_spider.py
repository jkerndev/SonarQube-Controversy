import scrapy
import time
import re
import json
import statistics
from collections import defaultdict, Counter
from scrapy.http import TextResponse
from bs4 import BeautifulSoup
from datetime import datetime
from playwright.async_api import Page
from scrapy_playwright.page import PageMethod

# This is a test spider for testing a single-complex sonar community post page

# TEST URLS
#test_url = "https://community.sonarsource.com/t/sonarqube-sonarcloud-and-the-log4j-vulnerability/54721" # Complex, long post with lots of JavaScript for testing PlayWright
#test_url = "https://community.sonarsource.com/t/ca1859-versus-s3242/105248" # Tests condition 1
#test_url = "https://community.sonarsource.com/t/non-cap-group-w-o-quantifier-specified-by-s5850-and-proscribed-by-s6395/110258" # Tests condition 2
#test_url = "https://community.sonarsource.com/t/semantic-provided-for-runtimeexception-is-against-java-guideline-in-rule-java-112/112282" # Tests condition 3

#test_url = "https://community.sonarsource.com/t/false-positive-of-csharpsquid-s1116/106204"
#test_url = "https://community.sonarsource.com/t/sonarlint-vs-extension-performance/54423"
test_url = "https://community.sonarsource.com/t/await-should-not-be-used-redundantly/29074"


scrolling_script = """
    const scrollInterval = setInterval(() => {
    window.scrollTo(0, document.body.scrollHeight)

    const replies = document.querySelector("div.timeline-replies").innerText;
    const replies_array = replies.split(" ");
    const numerator = parseInt(replies_array[0])
    const denominator = parseInt(replies_array[2])

    // All replies loaded
    if (numerator === denominator) {
      clearInterval(scrollInterval)
    }
  }, 100);
"""

# Avoid spider-tailored pages
custom_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"

def language_dict():
    with open('languages.json') as f:
        languages = json.load(f)
    
    return languages

def language_regex_scan():
    languages = language_dict()
    
    language_keys = list(languages['language'].keys())

    languages = "|".join([f"(?:\s+){re.escape(lang)}(?:\s+)" for lang in language_keys]) # Example: python

    return languages

def language_regex_rulescan():
    languages = language_dict()
    
    language_keys = list(languages['language'].keys())

    # Conditions, see outline document
    condition4 = f"(?:\s+)RSPEC-\d{{1,4}}(?:\s+)|(?:\s+)SPEC-\d{{1,4}}(?:\s+)|(?:\s+)RSPEC\d{{1,4}}(?:\s+)|(?:\s+)RSPEC\d{{1,4}}(?:\s+)" # Example: RSPEC-4326, SPEC-4326, RSPEC111, SPEC111 surrounded by whitespace 
    condition3 = f"(?:\s+)[sS]\d{{1,4}}(?:\s+)|(?:\s+)squid:[sS]\d{{1,4}}(?:\s+)" # Example: s1234, squid:S1604 surrounded by whitespace
    condition2 = "|".join([f"(?:\s+){re.escape(lang)}:[sS]\d{{1,4}}(?:\s+)|(?:\s+){re.escape(lang)}squid:[sS]\d{{1,4}}(?:\s+)" for lang in language_keys]) # Example: java:S6395, javasquid:S6395 surrounded by whitespace
    condition1 = "|".join([f"(?:\s+){re.escape(lang)}:\d{{1,4}}(?:\s+)|(?:\s+){re.escape(lang)}squid:\d{{1,4}}(?:\s+)" for lang in language_keys])  # Example: java:112, javasquid:112 surrounded by whitespace
    return condition1, condition2, condition3, condition4

LANGUAGE_CONDITIONS = language_regex_rulescan()
LANGUAGE_CONDITIONS_COMBINED = f"(?:{LANGUAGE_CONDITIONS[0]})|(?:{LANGUAGE_CONDITIONS[1]})|(?:{LANGUAGE_CONDITIONS[2]})|(?:{LANGUAGE_CONDITIONS[3]})"
LANGUAGE_CONDITIONS_COMBINED_COMPILE = re.compile(LANGUAGE_CONDITIONS_COMBINED, re.IGNORECASE | re.MULTILINE)
LANGUAGE_REGEX = language_regex_scan()
LANGUAGE_REGEX_COMPILE = re.compile(LANGUAGE_REGEX, re.IGNORECASE)
LANGUAGE_CONDITION1_COMPILE = re.compile(LANGUAGE_CONDITIONS[0], re.IGNORECASE)
LANGUAGE_CONDITION2_COMPILE = re.compile(LANGUAGE_CONDITIONS[1], re.IGNORECASE)
LANGUAGE_CONDITION3_COMPILE = re.compile(LANGUAGE_CONDITIONS[2], re.IGNORECASE)
LANGUAGE_CONDITION4_COMPILE = re.compile(LANGUAGE_CONDITIONS[3], re.IGNORECASE)
LANGUAGE_LOOKUP = language_dict()['language']
SONAR_POST_DATE_PATTERN = re.compile(r'Created: (\w+ \d{1,2}, \d{4} \d{1,2}:\d{2} \w{2})\nLatest: (\w+ \d{1,2}, \d{4} \d{1,2}:\d{2} \w{2})')

class SonarCommunitySpider(scrapy.Spider):
    name = 'sonar_community_spider_test'
    allowed_domains = ['community.sonarsource.com']
    url = 'https://community.sonarsource.com/top?period=all&page={}&per_page=50'
    custom_settings = {
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "DOWNLOAD_HANDLERS": {
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        }
    }
    pages_exist = True

    def start_requests(self):
        url = test_url
        yield scrapy.Request(url=url, 
                                  callback=self.parse_post,
                                  headers={'User-Agent': custom_user_agent},
                                  meta=dict(
                                    playwright=True,
                                    playwright_include_page=True,
                                    playwright_context_kwargs={
                                        "user_agent": custom_user_agent
                                    },
                                    playwright_page_methods=[
                                        PageMethod("evaluate_handle", scrolling_script),
                                        PageMethod("wait_for_timeout", 2500), 
                                    ],
                                ))
        
    async def parse_post(self, response):
        page: Page = response.meta["playwright_page"]
        self.logger.warning("ALERT 2")
        try:
            # Wait until the entire page content is loaded
            scrapyResponse = TextResponse(body=await page.content(), encoding='utf-8', url=page.url)
        finally:
            # Ensure the page closes even if there’s an error
            await page.close()

        # Write to html file to validate PlayWright HTML
        with open("post2.html", "w", encoding="utf-8") as file:
            file.write(scrapyResponse.text)

        #self.logger.warning("-------------------- HTML START ---------------------")
        #self.logger.warning(scrapyResponse.text)
        #self.logger.warning("-------------------- HTML END ---------------------")
        
        post_context = scrapyResponse.css('div[id="main-outlet"]')
        title_context = post_context.css('div[id="topic-title"] div.title-wrapper')
        # Look for references to the rule in the title, posts, or comments
        posts = post_context.css('div.post-stream')
        main_post = posts.css("div.topic-post article#post_1 div.row div.topic-body div.regular.contents div.cooked")
        rule, is_rule_related = self.rule_scan(title_context, main_post)
        if not is_rule_related:
            return
        post_stream = posts.css('div.topic-post')
        
        like_count, mean_like, stddev_like, user_count, code_references_count = self.get_post_stream_metadata(post_stream)
        item = {
            "id": title_context.css('h1::attr(data-topic-id)').get(),
            "datetimes": self.get_datetimes(post_context.css('td.num.topic-list-data.age.activity::attr(title)').getall()),
            "url": response.url,
            "rule": rule,
            "title": self.clean_content(title_context.css('h1').get()),
            "link_count": posts.css('div.topic-map section.topic-map__contents div.topic-map__stats button.topic-map__links-trigger span.number::text').get(default=0),
            "user_count": user_count,
            "main_post_likes": int(posts.css('article#post_1 div.only-like span.reactions-counter::text').get(default=0)),
            "code_references": code_references_count, 
            "mean_comment_likes": mean_like,
            "stddev_comment_likes": stddev_like,
            "comment_likes": like_count
        }
        yield item
        

    def get_datetimes(self, datetimes):
        correct_match = None
        for datetime_str in datetimes:
            match = re.search(SONAR_POST_DATE_PATTERN, datetime_str)

            if match:
                correct_match = match
                break
    
        created_str = correct_match.group(1)
        latest_str = correct_match.group(2)

        date_format = "%b %d, %Y %I:%M %p"
    
        created_date = datetime.strptime(created_str, date_format)
        latest_date = datetime.strptime(latest_str, date_format)
        
        # ISO 8061 format
        initial_post = created_date.isoformat()
        latest_update = latest_date.isoformat()
        
        return {
            "created": initial_post,
            "latest": latest_update
        }
    
    def rule_scan(self, title, main_post):
        """Greedy Rule Scanner"""
        hints = defaultdict(int)
        title_content = title.getall()
        main_post_content = main_post.getall()
        html_text = ''.join(title_content) + ''.join(main_post_content)
        clean_text = self.clean_content(html_text)

        matches = re.findall(LANGUAGE_CONDITIONS_COMBINED_COMPILE, clean_text)

        if len(matches) == 0:
            self.logger.warning(f"----- CONDITIONS START -----")
            self.logger.warning(LANGUAGE_CONDITIONS_COMBINED)
            self.logger.warning(f"----- CONDITIONS END -----")
            self.logger.warning(f"----- NO MATCH FOUND START -----")
            self.logger.warning(clean_text)
            self.logger.warning(f"----- NO MATCH FOUND END -----")
        else:
            for i, match in enumerate(matches):
                self.logger.warning(f"----- MATCHES START {i}-----")
                self.logger.warning(match)
                self.logger.warning(f"----- MATCHES END {i}-----")

        for match in matches:
            hints[match] += 1

        # Process by most frequent match
        while hints:
            max_key = self.get_intdict_max(hints)

            # Check if it matches condition 2 (preferred full match with language and 'S')
            if lang_match := re.search(LANGUAGE_CONDITION2_COMPILE, max_key):
                split_string = lang_match.group(0).strip().lower().replace("squid", '').split(':')
                standard_string = f"{split_string[0]}:{split_string[1].upper()}"
                return standard_string, True
            
            # Check if it matches condition 1 (language and code number, no 'S')
            if lang_match := re.match(LANGUAGE_CONDITION1_COMPILE, max_key): 
                split_string = lang_match.group(0).strip().lower().replace("squid", '').split(':')
                standard_string = f"{split_string[0]}:S{split_string[1]}"
                return standard_string, True
            
            # Check if it matches condition 3 (S/squid: + code number only)
            condition3_match = re.match(LANGUAGE_CONDITION3_COMPILE, max_key)
            if condition3_match:
                language_matches = re.findall(LANGUAGE_REGEX_COMPILE, clean_text)

                if len(language_matches) == 0:
                    self.logger.warning(f"----- LANGUAGE CONDITIONS START -----")
                    self.logger.warning(LANGUAGE_REGEX)
                    self.logger.warning(f"----- LANGUAGE CONDITIONS END -----")
                    self.logger.warning(f"----- NO LANGUAGE MATCH FOUND START -----")
                    self.logger.warning(clean_text)
                    self.logger.warning(f"----- NO LANGUAGE MATCH FOUND END -----")
                else:
                    self.logger.warning("----- LANGUAGE MATCHES DICT START -----")
                    self.logger.warning(json.dumps(LANGUAGE_LOOKUP, indent=4))
                    self.logger.warning("----- LANGUAGE MATCHES DICT END -----")
                    self.logger.warning("----- LANGUAGE MATCHES MAX_KEY START -----")
                    self.logger.warning(max_key)
                    self.logger.warning("----- LANGUAGE MATCHES MAX_KEY END -----")
                    for i, match in enumerate(language_matches):
                        self.logger.warning(f"----- LANGUAGE MATCHES START {i}-----")
                        self.logger.warning(match)
                        self.logger.warning(f"----- LANGUAGE MATCHES END {i}-----")
                if len(language_matches) >= 1:
                    # Count occurrences of each language match
                    match_counter = Counter(LANGUAGE_LOOKUP[language.strip().lower()] for language in language_matches)
                    most_common_language = match_counter.most_common(1)[0][0]
                    code = max_key.strip().lower().replace("squid:", '')[1:]
                    standard_string = f"{most_common_language}:S{code}"
                    return standard_string, True
                
            # Check if it matches condition 4 (RSPEC-num/SPEC-num/RSPECnum/SPECnum)
            condition4_match = re.match(LANGUAGE_CONDITION4_COMPILE, max_key)
            if condition4_match:
                language_matches = re.findall(LANGUAGE_REGEX_COMPILE, clean_text)

                if len(language_matches) >= 1:
                    # Count occurrences of each language match
                    match_counter = Counter(LANGUAGE_LOOKUP[language.strip().lower()] for language in language_matches)
                    most_common_language = match_counter.most_common(1)[0][0]
                    code = max_key.strip().lower().replace("rspec-", '').replace("spec-", '').replace("rspec", '').replace("spec", '') 
                    standard_string = f"{most_common_language}:S{code}"
                    return standard_string, True
                
            # If condition 3/4 has no language matches, try next most-frequent
            hints.pop(max_key)
        return None, False

    def clean_content(self, html_content):
        """Clean and standardize posts"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Normalize code blocks (you could wrap them in special markers or format them)
        for code_block in soup.find_all('pre'):
            code_block.insert_before('CODEBLOCK_START\n')
            code_block.insert_after('\nCODEBLOCK_END\n') 

        # Get all text, strip extra whitespace, and join lines
        description = soup.get_text(separator='\n')
        description = re.sub(r'[\r|\n|\r]+', '\n', description)
        description = description.strip()

        return description
    
    def get_post_stream_metadata(self, post_stream):
        """Return total like count, mean of the like distribution, std dev of the like distribution, unique user count, code reference (block) count"""
        likes = 0
        likes_list = []
        users = set()
        code_references = 0

        for post in post_stream:
            if (user := post.css('a::attr(data-user-card)').get()):
                users.add(user)
            # Skip the main post
            if not post.css('article#post_1'):
                if (like := post.css('div.only-like span.reactions-counter::text').get()):
                    like_count = int(like)
                    likes_list.append(like_count)
                    likes += like_count
                if (code_ref := post.css('pre.codeblock-buttons').getall()):
                    code_references += len(code_ref)
        
        self.logger.warning("----- COMMENT DIST START -----")
        self.logger.warning(likes_list)
        self.logger.warning("----- COMMENT DIST END -----")

        # Like Distribution Metrics
        mean_likes = statistics.mean(likes_list) if likes_list else 0
        std_dev_likes = statistics.stdev(likes_list) if len(likes_list) > 1 else 0

        return likes, mean_likes, std_dev_likes, len(users), code_references

    def get_intdict_max(self, dict):
        return max(dict, key=dict.get)