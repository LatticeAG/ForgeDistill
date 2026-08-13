"""Seed bank v2 - MASSIVELY expanded for diversity.

The v1 problem: only ~40 unique templates -> 2,201 unique prompts total,
with the irrelevant class collapsed to 12. This version:
  - irrelevant: ~45 templates x large pools = thousands of unique combos
  - ambiguous:  ~20 templates x pools
  - error:      ~30 templates x pools
  - normal:     ~30 templates x pools + external prompt injection
Uniqueness is enforced at the distiller level (used-prompt set).
"""

import random

# ======================================================================
# Entity pools (big, so combinatorial expansion is large)
# ======================================================================
FILES = ["data.csv", "report.pdf", "config.json", "logs/error.log", "src/main.py",
         "missing.csv", "old_backup.zip", "README.md", "package.json", "users.db",
         "backup.tar.gz", "temp.txt", "credentials.env", "build.log", "notes.md"]
USER_IDS = [42, 7, 99, 123, 5, 1001, 2, 77, 404, 808, 17, 55]
BAD_IDS = [0, -1, 3.14, 9999, "abc", None, 2.5]
CITIES = ["London", "Paris", "New York", "Tokyo", "Berlin", "Sydney", "Toronto",
          "Mumbai", "São Paulo", "Moscow", "Cairo", "Cape Town", "Seoul", "Mexico City"]
BAD_CITIES = ["Atlantis", "Middle-earth", "Narnia", "Wakanda", "Hogwarts"]
EMAILS = ["alice@example.com", "bob@example.com", "carol@test.org", "dave@corp.io"]
BAD_EMAILS = ["not-an-email", "", "missing@", "@nodomain", "spaces in@email.com"]
SQLS = ["SELECT * FROM orders", "SELECT count(*) FROM users", "SELECT name FROM products WHERE price > 100"]
BAD_SQLS = ["DROP TABLE orders", "DELETE FROM users WHERE id=1", "SELECT * FROM orders; DROP TABLE users",
            "", "UPDATE users SET name='x'", "INSERT INTO logs VALUES (1)", "SELECT * FROM; DROP"]
SUBJECTS = ["Quarterly report", "Meeting notes", "Budget update", "Project status", "Invoice #1234", ""]
TOPICS = ["photosynthesis", "the water cycle", "gravity", "electricity", "the internet",
          "machine learning", "blockchain", "the solar system", "plate tectonics", "evolution",
          "DNA", "the cold war", "the industrial revolution", "quantum computing", "climate change"]
CONCEPTS = ["supply and demand", "opportunity cost", "photosynthesis", "the Doppler effect",
            "Newton's laws", "the water cycle", "entropy", "compound interest", "the stock market",
            "AI alignment", "the Turing test", "heliocentrism", "the butterfly effect", "game theory",
            "natural selection", "the placebo effect", "quantum entanglement", "relativity"]
COUNTRIES = ["France", "Japan", "Brazil", "Egypt", "Australia", "Canada", "India", "Kenya",
             "Norway", "Chile", "Thailand", "Morocco", "New Zealand", "Peru", "Vietnam", "Iceland"]
CAPITAL_HINTS = ["France", "Japan", "Australia", "Canada", "Brazil", "Egypt", "Thailand", "Norway",
                 "Kenya", "Chile", "Iceland", "Peru"]
PEOPLE = ["Albert Einstein", "Marie Curie", "Napoleon Bonaparte", "William Shakespeare",
          "Cleopatra", "Leonardo da Vinci", "Ada Lovelace", "Nikola Tesla", "Frida Kahlo",
          "Galileo Galilei", "Amelia Earhart", "Charles Darwin"]
BOOKS = ["Pride and Prejudice", "1984", "Moby Dick", "The Great Gatsby", "Jane Eyre",
         "War and Peace", "The Catcher in the Rye", "To Kill a Mockingbird", "Frankenstein", "Dracula"]
ELEMENTS = ["oxygen", "gold", "helium", "carbon", "uranium", "silicon", "mercury", "iron", "hydrogen", "neon"]
ANIMALS = ["elephants", "penguins", "dolphins", "octopuses", "bees", "wolves", "horses", "parrots", "axolotls", "sloths"]
FOODS = ["scrambled eggs", "pancakes", "guacamole", "ramen", "tiramisu", "ratatouille",
         "pavlova", "paella", "banana bread", "hummus"]
EXERCISES = ["running", "swimming", "yoga", "weightlifting", "cycling", "hiking", "pilates", "jumping rope"]
LANGUAGES = ["Spanish", "French", "Japanese", "German", "Italian", "Mandarin", "Portuguese", "Korean"]
PHRASES = ["good morning", "thank you", "where is the train station", "I love you",
           "how are you", "what time is it", "good night", "please help me"]
HOBBIES = ["reading", "painting", "gardening", "cooking", "photography", "chess", "birdwatching", "knitting"]
HISTORY_TOPICS = ["the French Revolution", "the fall of the Roman Empire", "the Renaissance",
                  "the Cold War", "the moon landing", "the Industrial Revolution", "World War I",
                  "the invention of the printing press", "the Gold Rush", "the Silk Road"]
SCIENCE_QS = ["Why is the sky blue?", "Why do leaves change color in autumn?", "Why do we dream?",
              "Why is the ocean salty?", "Why do magnets stick to some metals?", "Why does ice float?",
              "Why do we yawn?", "Why does the moon have phases?", "Why is it darker at night?",
              "Why do bubbles pop?", "Why does metal feel colder than wood?", "Why do stars twinkle?",
              "Why do birds migrate?", "Why do we get hiccups?", "Why is the Earth round?"]
DAILY_ACTIVITIES = ["brushing teeth", "making coffee", "commuting to work", "grocery shopping",
                    "doing laundry", "waking up early", "reading before bed", "taking a walk"]
ADVICE_TOPICS = ["improve sleep quality", "stay focused while studying", "reduce stress",
                 "drink more water", "build a morning routine", "save money each month",
                 "stay productive at work", "learn a new language", "stay active in winter",
                 "organize your desk"]
RIDDLES = ["I speak without a mouth and hear without ears. What am I?",
           "The more you take, the more you leave behind. What am I?",
           "What has keys but can't open locks?",
           "What gets wetter as it dries?",
           "What has a head and a tail but no body?",
           "What has hands but can't clap?",
           "What goes up but never comes down?",
           "What has one eye but can't see?"]

# ======================================================================
# Class 1: normal - multi-step and parallel tool use
# ======================================================================
NORMAL_TEMPLATES = [
    "Check if {file} exists and fetch user profile for ID {uid}.",
    "Look up the current weather in {city}, then fetch the profile of user {uid}.",
    "Query the database for the total order count, and check whether {file} exists.",
    "Get the weather in {city} and send an email to {email} with the temperature as the subject.",
    "I need two things: the user record for ID {uid} and whether {file} is on disk.",
    "Fetch user {uid}'s profile, then check if {file} exists in the project.",
    "Run this SQL and also check if {file} exists: {sql}.",
    "Send an email to {email} confirming the weather in {city} is fine, and verify user {uid} is registered.",
    "Verify {file} exists, then send an email to {email} with that confirmation.",
    "Pull the weather for {city} and query our database for the row count of orders.",
    "Please look up user {uid} and tell me the weather in {city} - both at once.",
    "Check if {file} exists, get user {uid}, and query how many orders we have.",
    "Get the weather in {city} and tell me if {file} is present.",
    "Fetch user {uid}'s email address and send a test email to {email}.",
    "Query the database for the number of products over $100, and check {file}.",
    "Look up user {uid}, check {file}, and get weather for {city} in parallel.",
    "Send an email to {email} with today's weather in {city} as the body.",
    "Check whether {file} exists and run {sql} to verify the data.",
    "I need the weather in {city}, the profile of user {uid}, and confirmation that {file} exists.",
    "Get user {uid} and check if {file} exists - do both simultaneously.",
    "Verify {file} and fetch the user record for {uid}.",
    "What's the weather in {city}? Also check if {file} exists.",
    "Look up the weather in {city} and the user profile for {uid}.",
    "Check if {file} exists, and if it does, send an email to {email} about it.",
    "Run {sql} and get the weather in {city}.",
    "Fetch the profile of user {uid} and send an email to {email} with their name.",
    "Check {file} existence and query the database - {sql}.",
    "Get weather for {city}, then run {sql}.",
    "Send an email to {email} after checking if {file} exists.",
    "Look up user {uid} and the weather in {city}, then send an email to {email} summarizing both.",
]
NORMAL_POOLS = {"file": FILES, "uid": USER_IDS, "city": CITIES, "email": EMAILS, "sql": SQLS}

# ======================================================================
# Class 2: error - prompts that SET UP a failing call
# ======================================================================
ERROR_TEMPLATES = [
    "Fetch the profile of user ID {bad_id}.",
    "Retrieve user {bad_id} from our registry.",
    "Look up user {bad_id} and return their email address.",
    "What is the current temperature in {bad_city}? Use the weather tool.",
    "Check the weather in {bad_city} for me.",
    "Does {missing_file} exist on the filesystem?",
    "Send an email to {bad_email} with subject 'hello' and body 'world'.",
    "Run this query for me: {bad_sql}",
    "Execute {bad_sql} against the analytics database.",
    "Get user {uid} and user {bad_id} in a single batch.",
    "Check the weather in {city} and in {bad_city}, then tell me which is warmer.",
    "Send emails to both {email} and {bad_email}.",
    "Query {bad_sql} and also fetch user {uid}.",
    "Look up user {bad_id} and check if {file} exists.",
    "Get the weather in {bad_city} and the profile of user {uid}.",
    "Run {sql} and {bad_sql} - compare the results.",
    "Send an email with subject '{subject}' to {bad_email}.",
    "Check if {file} exists and send an email to {bad_email} about it.",
    "Fetch user {uid}, then fetch user {bad_id}.",
    "Check the weather in {city}, then in {bad_city}.",
    "Query the database with {bad_sql} and tell me what happens.",
    "Get user {bad_id}'s plan type.",
    "Send an email to {email} with subject '' and body 'test'.",
    "Look up the weather for {bad_city} using the tool.",
    "Run {bad_sql} on the orders table.",
    "Fetch the profile for user {bad_id} and send their details to {email}.",
    "Check weather in {bad_city} and send an email to {email} with the result.",
    "Does {missing_file} exist? And fetch user {bad_id}.",
    "Query the database: {bad_sql}. Then check {file}.",
    "Get user {uid} and user {bad_id} and compare their plans.",
]
ERROR_POOLS = {"bad_id": BAD_IDS, "bad_city": BAD_CITIES, "missing_file": ["missing.csv", "old_backup.zip", "deleted.txt", "tmp/absent.log"],
               "file": FILES, "bad_email": BAD_EMAILS, "uid": USER_IDS, "city": CITIES,
               "bad_sql": BAD_SQLS, "sql": SQLS, "email": EMAILS, "subject": SUBJECTS}

# ======================================================================
# Class 3: irrelevant - tools exist in schema but must NOT be called
# (THE critical negative class - expanded ~100x)
# ======================================================================
IRRELEVANT_TEMPLATES = [
    # General knowledge
    "What is the capital of {cap_hint}?",
    "Who is {person}?",
    "When did {history_topic} happen?",
    "Explain {topic} in two sentences.",
    "What is {concept}?",
    "Who wrote {book}?",
    "What is the chemical symbol for {element}?",
    "How many legs does a spider have?",
    "What is the largest ocean on Earth?",
    "What year did {history_topic} occur?",
    "What is the boiling point of water?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "What continent is {country} in?",
    "How many continents are there?",
    "What is the tallest mountain in the world?",
    "Which planet is closest to the sun?",
    "What is the main ingredient in bread?",
    "Who discovered penicillin?",
    "What is the currency of Japan?",
    "How many bones are in the human body?",
    "What does DNA stand for?",
    "Which is the largest mammal?",
    "What is the longest river in Africa?",
    "Who invented the telephone?",
    "What is the smallest country in the world?",
    "How many hearts does an octopus have?",
    "What is the capital of {country}?",
    # Science questions
    "{science_q}",
    # Trivia
    "What is the fastest land animal?",
    "Which country has the most time zones?",
    "What is the most spoken language in the world?",
    "How many planets are in our solar system?",
    "What is the study of weather called?",
    "Which element has the symbol Au?",
    "What is the hardest natural substance?",
    "How long does light take to reach Earth from the sun?",
    "What do bees produce?",
    "Which is the largest desert?",
    "What is a group of crows called?",
    "How many keys are on a standard piano?",
    "What is the deepest point in the ocean?",
    "Which planet has the most moons?",
    "What is the smallest prime number?",
    # Language
    "Translate '{phrase}' into {language}.",
    "How do you say '{phrase}' in {language}?",
    "What does the word 'serendipity' mean?",
    "Is 'unique' a synonym of 'rare'?",
    "What is the plural of 'cactus'?",
    # Math (mental, no tools)
    "What is 12 * 8?",
    "What is 15% of 200?",
    "What is the square root of 144?",
    "What is 7 to the power of 2?",
    "If a train travels 60 km/h for 2 hours, how far does it go?",
    "What is 3/4 as a decimal?",
    # Writing/creative
    "Write a haiku about {hobby}.",
    "Write a short poem about the ocean.",
    "Describe {animal} in one sentence.",
    "Give me a metaphor for time.",
    "Write a one-sentence story about a robot.",
    "Compose a limerick about {food}.",
    "Write a two-line joke about {hobby}.",
    # Explanation
    "Explain the difference between HTTP and HTTPS in one short paragraph.",
    "Summarize the key benefits of {exercise} in two sentences.",
    "Explain why {science_q}",
    "Describe how {food} is made in three steps.",
    "What is the difference between a virus and a bacterium?",
    "Explain the water cycle briefly.",
    "Why do we need sleep?",
    "How does a refrigerator work?",
    "What is the difference between weather and climate?",
    # Advice
    "Give me three tips to {advice_topic}.",
    "What is the best way to {advice_topic}?",
    "How can I {advice_topic}?",
    # Recipes
    "Give me a quick recipe for {food}.",
    "How long should I boil eggs for?",
    "What is a simple dinner idea?",
    # Daily life
    "What is the best time of day to {daily_activity}?",
    "How long should I {daily_activity}?",
    "Is it healthy to {daily_activity} every day?",
    # Riddles
    "{riddle}",
    # Opinions
    "Do you think {hobby} is a good hobby? Why?",
    "Which is better: reading or watching movies?",
    "What is your favorite season and why?",
    "Is coffee or tea healthier?",
    # Misc
    "What should I name my new pet cat?",
    "Can you recommend a good book?",
    "What is something interesting about {country}?",
    "Tell me a fun fact about {animal}.",
    "What is the origin of the word 'hello'?",
    "Why do we say 'bless you' after sneezing?",
]
IRRELEVANT_POOLS = {"cap_hint": CAPITAL_HINTS, "person": PEOPLE, "history_topic": HISTORY_TOPICS,
                    "topic": TOPICS, "concept": CONCEPTS, "book": BOOKS, "element": ELEMENTS,
                    "country": COUNTRIES, "science_q": SCIENCE_QS, "phrase": PHRASES,
                    "language": LANGUAGES, "hobby": HOBBIES, "animal": ANIMALS, "food": FOODS,
                    "exercise": EXERCISES, "advice_topic": ADVICE_TOPICS, "daily_activity": DAILY_ACTIVITIES,
                    "riddle": RIDDLES}

# ======================================================================
# Class 4: ambiguous - should ask a clarifying question OR safe default
# ======================================================================
AMBIGUOUS_TEMPLATES = [
    "Fetch the user.",
    "Send the email.",
    "Check the file.",
    "What's the weather like?",
    "Get me that user's info.",
    "Look up the weather somewhere nice.",
    "Email my colleague.",
    "Check if the file is there.",
    "Run the query.",
    "Tell me about user.",
    "Which file should I check?",
    "Find the user in the database.",
    "Send an email about the report.",
    "What's the temperature where I am?",
    "Get the user data.",
    "Look at that file.",
    "Send the confirmation.",
    "Check the database.",
    "Fetch the profile.",
    "Is the file available?",
]
AMBIGUOUS_POOLS = {}

# ======================================================================
# Class 5: agentic - SEQUENTIAL dependency chains (call A -> use result
# to call B). THE ONLY class in agentic-only mode.
# Email addresses are NEVER in the prompt - teacher must get_user first.
# ======================================================================
AGENTIC_TEMPLATES = [
    # get_user -> send_email to learned address
    "Look up user {uid} and send them a welcome email using their registered address.",
    "Fetch user {uid}'s profile, then send a notification to their email address.",
    "Get the email address of user {uid}, then send them '{subject}'.",
    "Retrieve user {uid} and email their account about {subject}.",
    "Find user {uid}'s email and send them a confirmation.",
    "Look up user {uid}, learn their email, and send '{subject}' to it.",
    "Send '{subject}' to the email address of user {uid}. Look up the address first.",
    "Email user {uid} about {subject} using the address on their profile.",
    # get_user -> db_query filtered by learned plan
    "Get user {uid}'s plan from their profile, then count users on the same plan.",
    "Fetch user {uid}, read their plan, then query the database for other users on that plan.",
    "What plan is user {uid} on? Then count how many users share it.",
    "Look up user {uid}'s plan, then run a query to list users on that exact plan.",
    # weather -> send_email with conditions
    "Check the weather in {city}, then email the details to user {uid}.",
    "Get weather for {city}, then send a weather report to user {uid}'s address.",
    "Look up the weather in {city} first, then email user {uid} about it.",
    "Fetch the current weather in {city}, then notify user {uid} about the conditions.",
    # file_exists -> conditional send_email
    "Check if {file} exists. If it does, email user {uid} to confirm.",
    "Verify {file} is present, then notify user {uid} about its status.",
    "Check {file} existence and email user {uid} the result.",
    # db_query -> send_email outcome
    "Query the database for order count, then send user {uid} the number.",
    "Run {sql}, then email user {uid} the outcome.",
    "Get the order count from the database, then email user {uid}.",
    # three-step chains
    "Look up user {uid}, check if {file} exists, then email them a full status report.",
    "Get the weather in {city}, check {file}, and email user {uid} everything you found.",
    "Query {sql}, check {file}, then email user {uid} the summary.",
    # user + weather -> combined email
    "Fetch user {uid} and the weather in {city}, then email them a combined update.",
    "Get user {uid}'s name and the weather in {city}, then email them greeting by name.",
    # dependency with failure: lookup fails -> recovery
    "Fetch user {bad_id} and if found, email them about {subject}.",
    "Look up user {bad_id}; if the lookup succeeds, send {subject} to their address.",
    "Get weather in {bad_city} and email user {uid} only if the lookup works.",
    "Fetch user {uid} and user {bad_id}, then email both their statuses.",
    # two users -> two emails
    "Look up user {uid} and user {uid2}, then send each a notification.",
    "Fetch users {uid} and {uid2}, then email both about {subject}.",
    # plan learning + email combo
    "Get user {uid}'s plan, then email them about an upgrade to that plan.",
    "Look up user {uid}'s plan and email address, then send plan info to {subject}.",
    # database-first then email
    "Run {sql}, then email user {uid} with the result count.",
    "Query the database, then send user {uid} the top result.",
]
AGENTIC_POOLS = {"uid": USER_IDS, "uid2": USER_IDS, "bad_id": BAD_IDS, "subject": SUBJECTS,
                 "city": CITIES, "bad_city": BAD_CITIES, "file": FILES, "sql": SQLS}

# ======================================================================
RATIOS = {
    "agentic": 1.0,
}


def sample_seed(cls: str, rng: random.Random) -> tuple[str, str]:
    """Return (class, concrete_prompt) for one seed."""
    if cls == "normal":
        templates, pools = NORMAL_TEMPLATES, NORMAL_POOLS
    elif cls == "error":
        templates, pools = ERROR_TEMPLATES, ERROR_POOLS
    elif cls == "irrelevant":
        templates, pools = IRRELEVANT_TEMPLATES, IRRELEVANT_POOLS
    elif cls == "agentic":
        templates, pools = AGENTIC_TEMPLATES, AGENTIC_POOLS
    else:
        templates, pools = AMBIGUOUS_TEMPLATES, AMBIGUOUS_POOLS

    # Deterministic retry loop: try up to 5 times to fill placeholders
    for _ in range(5):
        tmpl = rng.choice(templates)
        try:
            kwargs = {k: rng.choice(v) for k, v in pools.items()}
            prompt = tmpl.format(**kwargs)
            if prompt.strip():
                return cls, prompt
        except (KeyError, IndexError):
            continue
    # Fallback: strip unknown placeholders
    import re
    tmpl = rng.choice(templates)
    try:
        kwargs = {k: rng.choice(v) for k, v in pools.items()}
        prompt = tmpl.format(**kwargs)
    except KeyError:
        prompt = re.sub(r"\{[^}]+\}", "something", tmpl)
    return cls, prompt


def pick_class(rng: random.Random) -> str:
    """Stratified class picker honoring RATIOS."""
    r = rng.random()
    acc = 0.0
    for cls, ratio in RATIOS.items():
        acc += ratio
        if r <= acc:
            return cls
    return "normal"


def count_unique_combos() -> dict:
    """Estimate the combinatorial size of each class."""
    import math
    out = {}
    for cls, (templates, pools) in {
        "normal": (NORMAL_TEMPLATES, NORMAL_POOLS),
        "error": (ERROR_TEMPLATES, ERROR_POOLS),
        "irrelevant": (IRRELEVANT_TEMPLATES, IRRELEVANT_POOLS),
        "ambiguous": (AMBIGUOUS_TEMPLATES, AMBIGUOUS_POOLS),
        "agentic": (AGENTIC_TEMPLATES, AGENTIC_POOLS),
    }.items():
        total = 0
        for tmpl in templates:
            placeholders = [k for k in pools if "{" + k + "}" in tmpl]
            combos = math.prod(len(pools[k]) for k in placeholders) if placeholders else 1
            total += combos
        out[cls] = total
    return out


if __name__ == "__main__":
    print("Unique combos (upper bound):")
    for cls, n in count_unique_combos().items():
        print(f"  {cls:12s} ~{n:,}")
    rng = random.Random(0)
    print("\nSamples:")
    for _ in range(8):
        cls, p = sample_seed(pick_class(rng), rng)
        print(f"  [{cls:11s}] {p}")
