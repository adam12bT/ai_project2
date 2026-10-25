"""Curated questions; source passages resolved from downloaded documents, never invented."""
import json
from librarian.corpus import DATA

# page | answer locator | two independently phrased questions
CASES = '''csv|DictReader|How do I read CSV rows as dictionaries?|Which CSV reader maps column names to row values?
csv|DictWriter|How do I write dictionaries to a CSV file?|Which CSV class writes rows from dictionaries?
json|loads|How can I decode a JSON string?|Which function parses JSON from a string?
json|dumps|How do I encode an object as a JSON string?|Which JSON function returns a serialized string?
pathlib|read_text|How do I read a text file using pathlib?|Which Path method returns the contents of a text file as a string?
pathlib|mkdir|How do I create a directory with pathlib?|Which Path method creates a new directory?
re|findall|How do I find all matches of a regular expression?|Which regex function returns all non-overlapping matches?
re|sub|How do I replace text using a regular expression?|Which regex function substitutes matching text?
datetime|strptime|How do I parse a date string using a format?|Which datetime method parses a string according to a date format?
datetime|strftime|How do I format a datetime as a string?|Which date method produces formatted date text?
collections|Counter|How do I count occurrences of items in a list?|Which collections class stores counts for hashable objects?
collections|defaultdict|How do I give dictionary keys automatic default values?|Which collections dictionary calls a factory for missing values?
itertools|product|How do I compute a Cartesian product of iterables?|Which itertools function replaces nested loops over inputs?
itertools|combinations|How do I generate combinations of a fixed size?|Which itertools function selects subsequences without repetition?
functools|lru_cache|How do I cache the results of a function?|Which functools decorator provides a bounded memoization cache?
functools|partial|How do I pre-fill some arguments of a function?|Which functools helper freezes function arguments?
math|isclose|How do I compare floating point numbers approximately?|Which math function checks whether two values are close?
math|sqrt|How do I compute a square root?|Which math function returns a square root?
random|choice|How do I choose a random element from a sequence?|Which random function selects one sequence element?
random|sample|How do I sample elements without replacement?|Which random function returns unique selections from a population?
statistics|median|How do I calculate the median of data?|Which statistics function returns the middle value?
statistics|mean|How do I calculate an arithmetic average?|Which statistics function computes the arithmetic mean?
decimal|Decimal|How do I represent decimal numbers exactly?|Which decimal class supports decimal arithmetic?
fractions|Fraction|How do I represent a rational number?|Which class represents a numerator and denominator?
dataclasses|dataclass|How do I generate methods for a data class?|Which decorator adds generated special methods to a class?
enum|IntEnum|How do I create enumeration members that are integers?|Which enum type is also a subclass of int?
argparse|add_argument|How do I define a command line argument?|Which ArgumentParser method registers an argument?
argparse|parse_args|How do I parse command line options?|Which parser method processes command line argument strings?
logging|basicConfig|How do I configure basic logging?|Which logging function sets up the root logger?
logging|getLogger|How do I get a named logger?|Which logging function retrieves a logger by name?
unittest|assertRaises|How do I test that an exception is raised?|Which unittest assertion checks for an expected exception?
sqlite3|connect|How do I open a SQLite database?|Which sqlite3 function creates a database connection?
sqlite3|commit|How do I save a SQLite transaction?|Which connection method commits a pending transaction?
subprocess|run|How do I run an external command and wait for it?|Which subprocess function runs a command and returns a CompletedProcess?
shutil|copytree|How do I copy an entire directory tree?|Which shutil function recursively copies directories?
shutil|rmtree|How do I delete a directory tree?|Which shutil function removes a directory recursively?
tempfile|TemporaryDirectory|How do I create a temporary directory that cleans itself up?|Which tempfile class creates a managed temporary directory?
glob|iglob|How do I iterate over wildcard path matches?|Which glob function returns an iterator rather than a list?
copy|deepcopy|How do I copy nested objects recursively?|Which copy function makes a deep copy?
pprint|pformat|How do I format an object for readable printing as a string?|Which pprint function returns formatted text?
textwrap|dedent|How do I remove common leading indentation from text?|Which textwrap function removes common whitespace prefixes?
heapq|nlargest|How do I get the largest n elements of an iterable?|Which heapq function returns the n largest elements?
bisect|bisect_left|How do I find the left insertion point in a sorted list?|Which bisect function inserts before existing equal entries?
queue|Queue|How do I create a synchronized FIFO queue?|Which queue class supports multiple producers and consumers?
threading|Lock|How do I synchronize threads with a lock?|Which threading primitive provides acquire and release methods?
contextlib|contextmanager|How do I turn a generator into a context manager?|Which contextlib decorator creates a context manager from a generator?
gzip|open|How do I open a gzip compressed file?|Which gzip function opens compressed files in text or binary mode?
zipfile|ZipFile|How do I work with ZIP archives?|Which zipfile class reads and writes ZIP files?
hashlib|sha256|How do I compute a SHA256 hash?|Which hashlib constructor creates a SHA-256 digest object?
secrets|token_urlsafe|How do I generate a secure URL-safe token?|Which secrets function returns random URL-safe text?
urllib.parse|urlencode|How do I encode a mapping as URL query parameters?|Which urllib.parse function creates an encoded query string?
configparser|ConfigParser|How do I read INI configuration files?|Which configparser class manages configuration sections?
tomllib|loads|How do I parse a TOML string?|Which tomllib function reads TOML from a string?
warnings|warn|How do I emit a warning?|Which warnings function issues a warning message?
inspect|signature|How do I inspect the signature of a callable?|Which inspect function returns a Signature object?'''

def main():
    chunks=[json.loads(line) for line in (DATA/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[]
    for entry in CASES.splitlines():
        page,term,*questions=entry.split('|'); page='library/'+page
        matches=[c for c in chunks if c['page']==page and term in c['text']]
        if not matches: raise ValueError(f'Missing source: {page} {term}')
        # Prefer the section defining the symbol; locator matches are weak labels until reviewed.
        matches.sort(key=lambda c:(term.lower() not in c['section'].lower(),len(c['text'])))
        source=matches[0]
        relevant=[c['id'] for c in matches]
        for number,question in enumerate(questions):
            rows.append(dict(id=f'q{len(rows)+1:03}',question=question,answer=term,source_passage=source['text'],source_url=source['url'],relevant_ids=relevant,answerable=True,split='dev' if number==0 else 'test',review_status='needs_human_review',topic=page+':'+term))
    for question in ['Who won the 2026 football World Cup?','What is the weather in Tunis today?','What is my university tuition fee?','How do I install a PostgreSQL cluster on Kubernetes?','What are the side effects of aspirin?','How do I train a convolutional network in PyTorch?','What is the capital of Tunisia?','What are the Django authentication settings?','How do I create a React component?','What is the price of Bitcoin today?']:
        rows.append(dict(id=f'q{len(rows)+1:03}',question=question,answer="I don't know",source_passage=None,source_url=None,relevant_ids=[],answerable=False,split='test',review_status='needs_human_review',topic='out_of_scope'))
    (DATA/'eval_questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    print(f'Wrote {len(rows)} questions. Weak source labels require human review; dev/test paraphrases share topics.')

if __name__=='__main__': main()
