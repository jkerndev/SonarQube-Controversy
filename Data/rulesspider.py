import scrapy
from bs4 import BeautifulSoup
import re

class RulesSpider(scrapy.Spider):
    name = 'rules_spider'
    allowed_domains = ['rules.sonarsource.com']
    start_urls = ['https://rules.sonarsource.com/']

    def parse(self, response):
        language_ruleset_links = response.css('ul[class^="LanguagesListstyles__Ul-sc"]')
        for li in language_ruleset_links.css('li'):
            link = li.css('a::attr(href)').get()
            yield response.follow(link, self.parse_rule_set)

    def parse_rule_set(self, response):
        language_rule_links = response.css('ol[class^="RulesListstyles__StyledOl-sc"]')
        for li in language_rule_links.css('li'):
            link = li.css('a::attr(href)').get()
            yield response.follow(link, self.parse_rule)

    def parse_rule(self, response):
        rule_url = response.url.split('/')
        rule = response.css('div[class^="RuleDetailsstyles__StyledContainer-sc"]')
        item = {
            'language': rule_url[-3],
            'code': ''.join(['S', rule_url[-2].split('-')[-1]]),
            'name': rule.css('h1::text').get(),
            'attributes': self.remove_comments(rule.css('div[class^="RuleDetailsstyles__StyledCleanCodeAttribute-sc"]::text').getall()),
            'impacts': self.remove_comments(rule.css('div[class^="Impactstyles__StyledContainer-sc"]::text').getall()),
            'type': rule.css('div[class^="RuleDetailsstyles__StyledType-sc"]::text').get(),
            'severity': self.remove_comments(rule.css('div[class^="RuleDetailsstyles__StyledSeverity-sc"]::text').getall())[0],
            'quick_fix': True if rule.css('div[class^="Quickfixstyles__StyledContainer-sc"]::text').getall() else False,
            'tags': self.remove_comments(rule.css('ul[class^="RuleDetailsstyles__StyledMetadataTags-sc"] li a::text').getall()),
            'description': self.clean_description(rule.css('section[class^="RuleDetailsstyles__StyledDescription-sc"]').get())
        }
        yield item

    def remove_comments(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        #print("Precleaned text:", repr(text))
        cleaned_text = text.replace('\xa0', '')
        cleaned_text = re.sub(r'<!--.*?-->', '', cleaned_text, flags=re.DOTALL)
        #print("Cleaned text:", repr(cleaned_text)) 
        split = cleaned_text.split()
        cleaned_text = [text.strip() for text in split if re.match(r'[a-zA-Z]+', text.strip()) or re.match(r'[^\s]+-[^\s]+', text.strip())]
        return cleaned_text

    def clean_description(self, html_content):
        """Clean and standardize the description text."""
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